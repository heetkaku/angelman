import numpy as np
from scipy import signal
from itertools import zip_longest, product

from scripting.spectrum.io import paths, dialogs, persist
from scripting.spectrum import filters, masking
from scripting.spectrum.io.eeg import EEG
from scripting.spectrum.io.annotations import pinnacle

def welch(arr, fs, axis, nperseg, **kwargs):
    """Returns the spectrum of array using the welch mean peridogram method.

    Args:
        arr (ndarray):          time-series of measurements
        fs (int):               sampling frequency of arr
        axis (int):             axis along with to compute the spectrum
        nperseg (int):          length of each segment (Default is 16384
                                samples per window)
        **kwargs:               passed to scipy.signal.welch
    """

    return signal.welch(arr, fs=fs, nperseg=nperseg, axis=axis, **kwargs)

def bandstop(arr, fs, axis, order=8, ftype='bandstop', criticals=[58, 62],
             **kwargs):
    """Returns an array that has been bandstop filtered with Butterworth.

    Args:
        arr (ndarray):          time-series of measurements
        fs (int):               sampling frequency of arr
        order (int):            number of filter coeffecients (Default is 8)
        criticals (seq):        a start, stop tuple of bandstop (default is
                                58 to 62 Hz
        axis (int):             axis of data to apply filter
        **kwargs:               passed to filter's apply method
    """

    f = filters.Butterworth(order=order, ftype=ftype, criticals=criticals, fs=fs)
    return f.apply(arr, axis=axis, **kwargs)

def serial_gen(eeg, mask, epoch):
    """Yields masked data of eeg between start and stop samples in epoch.

    Args:
        eeg (obj):          iterable eeg instance
        mask (ndarray):     boolean of values to mask
        epoch (tuple):      start, stop samples between which eeg data will
                            be serially yielded in chunks of eeg.csize

    Assumptions:
        1. stop - start > eeg.csize

    Yields: numpy array of masked data
    """

    starts = range(*epoch, eeg.csize)
    for a, b in zip_longest(starts, starts[1:], fillvalue=starts.stop):
        yield eeg[a:b][mask[a:b]]

def epoch_gen(eeg, mask, epochs):
    """Yields masked data of eeg during start, stop sample tuples.

    Args:
        eeg (obj):          iterable eeg instance
        mask (ndarray):     boolean of values to mask
        epochs (seq):       sequence of start, stop sample tuples

    Assumptions:
        1. Each epoch is smaller than available RAM.

    Yields: numpy array of masked data
    """

    for a, b in epochs:
        yield eeg[a:b][mask[a:b]]

def data_gen(eeg, mask, epochs):
    """Yields masked eeg data for each start, stop sample epoch in epochs.

    This is a dispatch function switching between yielding data from one
    large epoch (see serial_gen) and yielding data over multiple small
    epochs (see epoch_gen).
    """
    
    if epochs.size > 2:
        return epoch_gen(eeg, mask, epochs)
    else:
        return serial_gen(eeg, mask, epochs)

def spectrum(eeg, mask, nperseg, epochs, notch, **kwargs):
    """Returns the Power Spectral Density during hours of a masked eeg.

    Args:
        eeg (obj):          iterable eeg instance
        mask (ndarray):     boolean of values to mask
        nperseg (int):      len of FFT segments
        epochs (seq):       sequence of start, stop sample tuples
        notch (bool):       if True, apply a 60Hz notch filter eeg
        kwargs:             passed to scipy's welch
    
    Returns: array of frequencies and array of psd values
    """
    
    #FFT axis of EEG instance is always 0th
    axis = 0
    #Build data generator
    data = data_gen(eeg, mask, epochs)
    #compute averaged spectrum
    fs = int(eeg.sample_rate)
    avg, n, f = 0, 0, 0
    for segment in data:
        #make sure segment is not empty
        if segment.size > 0:
            filtered = bandstop(segment, fs, axis) if notch else segment
            f, pxx = welch(filtered, fs, axis, nperseg, **kwargs)
            avg  = (n*avg + pxx) / (n+1)
            n = n+1
    return f, avg

def period(path, between, fs, reader, **kwargs):
    """Returns a 2-el array of start & stop samples from an annotation file.

    Args:
        path (str):             filepath to an annotation file
        between (2-seq):        a seq of two strings specifying start and
                                stop time in annotation file
        fs (int):               sampling rate of system 
        reader (funct):         an annotation reader function
        **kwargs:               passed to reader
    """
    
    res = np.array([t0 for t0, _, _ in reader(path, *between, **kwargs)])
    return np.rint(res * fs).astype(int)

def resultant(data, **indices):
    """Returns a dict with a numpy array data field and meta-data.

    Args:
        data (np.array):        an n-dim numpy array
        **indices:              indices of

    Returns: dict

    Python 3.6+ specific as prior vers do not preserve kwarg ordering
    """

    indices.update({'axes': list(indices.keys()), 'data': data})
    return indices

def _build(paths, nperseg, annote, rejects, hours, between, notch, csize,
           reader, **kwargs):
    """Returns a spectrum for each eeg,annotation tuple path in paths. 
    
    See build caller for argument descriptions.

    Returns: chs x freqs x names array of psds, channel sequence,
    frequencies sequence & names sequence.
    """

    pxxs, names = [], []
    for ix, (eeg_path, ann_path) in enumerate(paths):
        print("File {} of {}" .format(ix+1, len(paths)))
        eeg = EEG(str(eeg_path), chunksize=csize)
        fs = int(eeg.sample_rate)
        #compute hourly or endpoint epochs
        endpts = [0, len(eeg)]
        if between: 
            endpts = period(ann_path, between, fs, reader, **kwargs)
        epochs = (np.array(hours)* 3600 * fs) + endpts[0] if hours \
                  else np.array(endpts)
        epochs = epochs.astype(int) #Sample numbers cannot be floats
        #build individual acceptance and rejection mask and combine
        if annote.lower() in {'ignore'}:
            accept = np.ones(len(eeg), dtype=bool)
        else:
            accept = masking.from_csv(ann_path, len(eeg), fs, True, annote, 
                                      reader=reader, **kwargs)
        reject = masking.from_csv(ann_path, len(eeg), fs, False, *rejects,
                                  reader=reader, **kwargs)
        eeg_mask = masking.intersect(accept, reject)
        #compute and store spectra
        if np.count_nonzero(eeg_mask) > 1:
            f, pxx = spectrum(eeg, eeg_mask, nperseg, epochs, notch)
        else:
            #if no samples survive mask-- return nans
            f = np.arange(0, fs/2 + fs/nperseg, fs/nperseg)
            pxx = np.nan * np.ones((len(f), eeg.num_channels))
        pxxs.append(pxx)
        names.append(eeg_path.stem)
        freqs = f 
    #names x freqs x chs -> chs x freqs x names
    arr = np.stack(pxxs, axis=0)
    arr = np.swapaxes(arr, 0, 2)
    return arr, np.arange(eeg.num_channels), freqs, names

def build(nperseg=5000, annotations=['rest', 'grooming', 'exploring'],
          rejects=['artifact', 'water_drinking'], hours=None, 
          between=['start', 'stop'], notch=True, csize=30e6, reader=pinnacle, 
          **kwargs):
    """Computes and returns power spectrums for each annotation across all
    dialog selected eeg and annotation files.

    Args:
        nperseg (int):              len of nfft segments; larger values
                                    yield finer frequency resolution but
                                    increase compute times. (Default: 16384)
        annotations (seq):          seq. of annotation strings to extract
                                    samples from eeg for analysis. Each
                                    annotation will be analyzed separately
                                    and stored along 0th axis of result
                                    instance. The reserved 'ignore'
                                    annotation may be used to ignore all
                                    annotions and analyze complete eeg for
                                    spectrum (Default: ['rest','grooming',
                                    'exploring']).
        rejects (seq):              seq. of annotation strings that exclude
                                    samples of eeg from analysis, If more
                                    than 1 are present, samples from the
                                    union will be exluded.
        hours (seq):                seq of start stop hours to analyze
                                    spectrum over (Default: None)
        between (seq):              2-el seq. of start, stop strings that
                                    are substrings of two annotation strings
                                    specifying sample endpoints of eeg for
                                    analysis (Default: ['start' and 'stop'])
        notch (bool):               boolean specifying if 60 Hz notch filter
                                    should be applied (Default: True)
        csize (int):                number of samples to hold in memory at
                                    any time (Default: 1e6)
        reader (callable):          function for reading annotation file.
                                    Should return (start, duration, name)
                                    tuple for each annotation in file.
                                    (Default: pinnacle, see annotations.py)
        **kwargs:                   optional list of eeg, annotation path
                                    tuples and kwargs for reader.

    Returns: resultant dict instance with data ndarray of shape
    len(annotations) x len(channels) x len(frequencies) x len(eeg files)
    
    Note: Annotation/EEG files missing an annotation are assigned spectrums
    filled with NaNs across chs & frequencies to ensure data is non-ragged.
    """

    #fetch the eeg and annotation paths 
    fpaths = kwargs.pop('paths', None)
    if not fpaths:
        fpaths = dialogs.matched(['Select EDFs', 'Select Annotations'], 
                         initialdir=paths.DATA_DIR)
    #build psds across annotations and files
    results = []
    for annote in annotations:
        print("Annotation: {}" .format(annote))
        result = _build(fpaths, nperseg, annote, rejects, hours, between,
                       notch, csize, reader, **kwargs)
        pxxs, chs, freqs, names = result
        results.append(pxxs)
    #stack results -> annotations x chs x freqs x names
    data = np.stack(results, axis=0)
    result = resultant(data, annotations=annotations, channels=chs, 
                       frequencies=freqs, names=names)
    #include build parameters
    params = {'nperseg':nperseg, 'annotations': annotations,
              'rejects':rejects, 'hours':hours, 'notch': notch,
              'between': between, 'csize': csize, 'reader': reader,
              'filepaths': fpaths}
    params.update(kwargs)
    result['parameters'] = params
    return result

def extract(dset, keepdims=False, **filters):
    """Extracts a subarray from a dataset using listed-axis, value items
    specified in filters.

    A dataset contains a data array and a set of listed axis describing the 
    data at each index. For example, the 'annotations' list contains all the
    annotations that lie along the annotations axes. Extract filters these
    lists and returns an subarray of values for all filtered listed-axes.

    Sample call to return all data for with rest annotation for chs 1 & 3:
    >> extract(dset, annotations=['rest'] channels=[1,3]) 
    
    Args:
        dset (dict):            a dataset dict object
        keepdims (bool):        keep dataset's data dims after filtering
                                (Default False -> squeeze single dims)
        filters:                a collection of named axis and values to
                                extract along axis
    """
    
    arr = dset['data']
    for axis_name, values in filters.items():
        axis = dset['axes'].index(axis_name)
        indices = [list(dset[axis_name]).index(v) for v in values]
        arr = np.take(arr, indices, axis=axis)
    return arr.squeeze() if not keepdims else arr
        

if __name__ == '__main__':

    from scripting.spectrum import metadata
    import time
   
    #build result
    t0 = time.perf_counter()
    result = build(between=[' Start', ' Stop'], annotations=['ignore'], 
                   delimiter='\t')
    print('Build completed in {} s'.format(time.perf_counter() - t0))
    #build metadata from filepaths
    meta_path = dialogs.standard('askopenfilename', title='Select meta CSV',
                                 initialdir=paths.DATA_DIR)
    meta = metadata.from_csv(meta_path, 0, 'Animal ID', *['Treatment'], 
                             delimiter='\t')
    result['metadata'] = meta
    
    #save result
    spath = dialogs.standard('asksaveasfilename')
    persist.pkl_save(spath, result)
