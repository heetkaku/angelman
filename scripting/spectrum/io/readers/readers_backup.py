import copy
import numpy as np
import warnings
import abc

from scripting.williams.spectrum.io.readers import headers

class Reader(abc.ABC):
    """Abstract base class of all readers specfying abstract methods and
    properties expected of all subclasses."""

    @abc.abstractmethod
    def num_samples(self):
        """Computes the number of samples for each channel.
        
        Returns: a sequence of ints one per channel
        """
        
        pass

    @abc.abstractmethod
    def num_channels(self):
        """Returns the number of channels (int)."""
        
        pass
    
    @abc.abstractmethod
    def sample_rates(self):
        """Computes sample rates for each channel.
        
        Returns: sequence of sample rates for each channel
        """

        pass

    @abc.abstractmethod
    def read(self, start, stop):
        """Reads channel data between start and stop samples.
        
        Returns np.arr of shape num_samples X num_channels
        """
        
        pass

class EDF(Reader):
    """Interface to a European Data Format file providing both metadata
    information and fast access to signal data stored in the file. 

    Data access is provided by the read method returning non-annotation
    signals between start and stop sample values. Metadata access is
    provided as attributes of the EDF instance.
    
    Attrs:
        path (str):                 path to EDF file
        annotated (bool):           EDF+ file with possible annotations
        num_channels (int):         number of non-annotation signals in edf
        num_samples (arr):          number of samples for each
                                    non-annotation signal in edf
        file_header (dict):         File Metadata Contatining:
                                    edf version (0 always)
                                    patient (str)
                                    recording (str)
                                    start_date (str)
                                    start_time (str)
                                    header_bytes (int)
                                    reserved (str)
                                    num_records (int)
                                    record_duration (float)
                                    num_signals (int)
        signal_header (dict):       Signal Metadata Containing:
                                    names (list of strs)
                                    transducers (list of strs)
                                    physical_dim (list of strs)
                                    physical_min (list of floats)
                                    physical_max (list of floats)
                                    digital_min (list of floats)
                                    digital_max (list of floats)
                                    prefiltering (list of strs)
                                    samples_per_record (list of ints)
                                    reserved (list of strs)
    """

    def __init__(self, path):
        """Initialize this EDF instance.
        
        Args:
            path (str):         full file path to EDF file
        """

        self.path = path
        #read headers
        self.file_header, self.signal_header = headers.read_header(path)
        #determine if file is annotated
        self.annotated = 'EDF Annotations' in self.signal_header['names']
        #get channel counts (non-annotation signals)
        num_signals = self.file_header['num_signals'] 
        #build list of channel indices
        ann_idx = self.annotation_index()
        self._channel_idxs = np.delete(np.arange(num_signals), ann_idx)
        #compute signal calibrations and offsets
        self._calibrations, self._offsets = self._signal_scales()
        #Determine if read can be optimized
        self._optimized = self._optimized()
        #record section start, size in samples and size in bytes
        self._rec_start = self.file_header['header_bytes']
        self._rec_samples = sum(self.signal_header['samples_per_record'])
        self._rec_bytes = self._rec_samples * 2

    @property
    def num_channels(self):
        """Returns the number of channels (ordinary signals of this EDF."""

        return self.file_header['num_signals'] - 1 if self.annotated else 0

    @property
    def num_samples(self):
        """Computes total number of samples one per channel in this EDF.
        
        Returns: array of number of samples one per non-annation signals
        """
        
        #get the ordinary sinal lens per record
        olens, _ = self._signal_lens()
        #multiply by num of records
        nsamples = self.file_header['num_records'] * np.array(olens)
        return nsamples.astype('int')

    @property
    def sample_rates(self):
        """Computes the sample rates one per channel in this EDF.

        Returns: array of samle rates one per non-annotation signal
        """
        olens, _ = self._signal_lens()
        return np.array(olens) / self.file_header['record_duration'] 

    def _signal_scales(self):
        """Computes the signal calibrations and offsets for this EDF.
        
        Returns: list of calibrations and offsets one per signal
        """

        #compute the signal calibrations and offsets
        pmaxs = np.array(self.signal_header['physical_max'])
        pmins = np.array(self.signal_header['physical_min'])
        dmaxs = np.array(self.signal_header['digital_max'])
        dmins = np.array(self.signal_header['digital_min'])
        #store the calibrations and offsets
        calibrations = (pmaxs - pmins) / (dmaxs - dmins)
        offsets = pmins - calibrations * dmins
        return calibrations, offsets

    def annotation_index(self):
        """Locates index of annotations signal among all signals.
        
        Returns: int
        """
        
        if self.annotated:
            return self.signal_header['names'].index('EDF Annotations')
        else:
            return None

    def _signal_lens(self):
        """Returns the samples per record for the ordinary and annotation
        signals in this EDF.
        
        Returns: list(s) of samples per data record in this EDF. A list is
        returned for ordinary signals and a second list for annotations if
        present.
        """
        
        if self.annotated:
            ann_idx = self.annotation_index()
            olens = copy.copy((self.signal_header['samples_per_record']))
            ann_len = olens.pop(ann_idx)
            return olens, ann_len
        else:
            olens = self.signal_header['samples_per_record']
            return olens, None

    def _optimized(self):
        """Determines if the ordinary signal sample rates are equal.

        If the signal sampling rates are equal we optimize data reading
        of this EDF by fast numpy reshape operations (see _optimized_read)

        Returns: boolean
        """
        
        #get len of ordinary signals in a record
        olens, _ = self._signal_lens()
        #determine if the len is unique
        optimized = len(set(olens)) == 1
        #remove annotation calibration and offset if annotations
        if optimized and self.annotated:
            ann_idx = self.annotation_index()
            self._calibrations = np.delete(self._calibrations, ann_idx)[0]
            self._offsets = np.delete(self._offsets, ann_idx)[0]
        elif optimized and not self.annotated:
            self._calibrations = self._calibrations[0]
            self._offsets = self._offsets[0]
        #return boolean of whether optimization succeeded
        return optimized

    def _split_signals(self, arr):
        """Splits the annotations signals from the ordinary signals.
        
        Args:
            arr (np.arr):   2-D num_records X samples_per_record arr

        Returns: numpy array of ordinary signals
        """
        
        if self.annotated:
            ann_idx = self.annotation_index()
            _, ann_len = self._signal_lens()
            #annotation is first signal
            if ann_idx == 0:
                return arr[:, ann_len:]
            #annotation is last signal
            elif ann_idx == self.file_header['num_signals'] - 1:
                return arr[:, :-ann_len]
            #annotation is somewhere between
            else:
                msg = 'Annotations should be at the beginning or' + \
                'end of data records for optimal read performance!'
                warnings.warn(msg)
                all_lens = self.signal_header['samples_per_record']
                start = sum(all_lens[:ann_idx])
                end = sum(all_lens[:ann_idx+1])
                arr_start, arr_ann, arr_end = arr.split(arr, start, stop)
                return np.concatenate((arr_start, arr_end), axis=1)
        else:
            return arr

    def _optimal_read(self, start, stop):
        """Returns data samples from all non-annotation (ordinary) signals
        between start and stop samples.

        Under the assumption of identical sample rates, fast numpy slicing
        and reshape operations are used to quickly read in all ordinary 
        signals of this EDF.

        Args:
            start (int):        start sample of read
            stop (int):         stop sample of read (exclusive)

        Returns: np array of shape (stop - start) X self.num_channels
        """
        
        #get the ordinary signal len
        o_lens, _ = self._signal_lens()
        olen = o_lens[0]
        #locate nearest start/stop records to requested start/stop samples
        start_rec = start // olen
        stop_rec = np.ceil(stop / olen).astype(int)
        #compute the number of records to read
        num_records = stop_rec - start_rec
        start_byte = start_rec * self._rec_bytes
        with open(self.path, 'rb') as infile:
            #compute offset from the start of records section
            byte_offset = self._rec_start + start_byte
            #compute number of samples to read
            nsamples = num_records * self._rec_samples
            #read little endian 2 bit signed integers (EDF spec)
            arr = np.fromfile(infile, dtype='<i2', count=nsamples,
                              offset=byte_offset).astype(float)
        #reshape and remove annotation signals
        arr = arr.reshape(num_records, self._rec_samples)
        arr = self._split_signals(arr)
        #scale data (calibrations and offsets are float) in-place
        arr = np.multiply(arr, self._calibrations, out=arr)
        arr = np.add(arr, self._offsets, out=arr)
        #reshape to num recs x num sigs x sig len
        arr = arr.reshape(num_records, -1, olen)
        #swap the num_recs (0th axis) with the num_sigs (1st axis) axis
        arr = arr.swapaxes(0,1)
        #reshape the array into num_sigs x (num_recs * sig_len)
        arr = arr.reshape(-1, num_records*olen)
        #transpose so channels are mapped to columns
        arr = arr.T
        #our arr is from start to end of records not samples so slice
        begin = start - start_rec * olen if start >= olen else start
        end = begin + (stop - start)
        #return copy of slice since views retain memory of original
        return arr[begin:end].copy()
        
    def _non_optimal_read(self, start, stop):
        """Returns data samples from non-annotation (ordinary) signals for
        signals with unequal sample rates between start and stop.
        
        Without the assumption of equal sample rates, the signals must be
        read in sequentially since they may require reading a different
        number of data records for each signal.

        Args:
            start (int):        start sample of read
            stop (int):         stop sample of read (exclusive)

        Returns: a np array of shape (stop-start) X self.num_channels

        Implementation Note: if the start or stop exceeds the number of
        samples for a given channel the last values of this channel are
        filled with np.NaNs
        """

        def read_channel(ch_idx, start, stop):
            """Reads data from channel between start and stop indices.
            
            Returns: np.array
            """
        
            #get lens of signals per record and scales
            all_lens = self.signal_header['samples_per_record']
            calibrations, offsets = self._signal_scales()
            #get the num samples per record of this channel 
            olen = all_lens[ch_idx]
            #locate nearest start/stop records to start/stop samples
            start_rec = start // olen
            stop_rec = np.ceil(stop / olen).astype(int)
            #compute the number of records to read and starting byte
            num_records = stop_rec - start_rec
            start_byte = start_rec * self._rec_bytes
            with open(self.path, 'rb') as infile:
                #compute offset from the start of records section
                byte_offset = self._rec_start + start_byte
                #compute number of samples to read
                nsamples = num_records * self._rec_samples
                #read little endian 2 bit signed integers (EDF spec)
                arr = np.fromfile(infile, dtype='<i2', count=nsamples,
                                  offset=byte_offset)
            #reshape
            arr = arr.reshape(num_records, self._rec_samples)
            #get a single signal by slicing the arr
            col_start = np.insert(np.cumsum(all_lens), 0, 0)[ch_idx]
            col_end = col_start + olen
            arr = arr[:, col_start:col_end].flatten()
            #calibrate and offset this channels data
            arr = arr * calibrations[ch_idx] + offsets[ch_idx]
            #arr is from start to end of records not samples so slice
            begin = start - start_rec * olen if start >= olen else start
            end = begin + (stop - start)
            return arr[begin:end].copy()
        #Read the channels sequentially
        result_ls = []
        for ch_idx in self._channel_idxs:
            #if requested stop is less than num_samples just read
            if stop < self.num_samples[ch_idx]:
                arr = read_channel(ch_idx, start, stop)
                result_ls.append(arr)
            elif start < self.num_samples[ch_idx]:
                #perform partial read upto last sample
                new_stop = self.num_samples[ch_idx]
                arr = read_channel(ch_idx, start, new_stop)
                result_ls.append(arr)
            else:
                #if start is greater than num samples return empty
                result_ls.append(np.array([]))
        #get the max len of all the arrays in result_ls
        longest=max(len(arr) for arr in result_ls)
        #pad each array upto longest 
        pad_results = [np.pad(arr, (0, longest-len(arr)), constant_values=np.NaN) 
                        for arr in result_ls]
        return np.stack(pad_results, axis=1) 

    def read(self, start, stop):
        """Read data samples from non-annotation (ordinary) signals from 
        this EDF between start and stop samples.
        
        Args:
            start (int):        start sample of read
            stop (int):         stop sample of read (exclusive)

        Returns: a np array of shape (stop-start) X self.num_channels
        """

        #Dispatch to appropriate reader
        if self._optimized:
            return self._optimal_read(start, stop)
        else:
            return self._non_optimal_read(start, stop)

    def read_annotations(self, sample):
        """This function has not yet been implemented. It will read the
        annotations based on the current sample testing if the sample is in
        the current data record and fetching the annotation if not updating
        as needed"""

        if not self.annotated:
            return None
        else:
            #use the current annotation cache (self.annotation) or fetch
            raise NotImplementedError()



if __name__ == '__main__':

    """
    import pyedflib
    import time
    import matplotlib.pyplot as plt
    
    fname = '/home/giladmeir/python/nri/open_seize/__data__/edf_samples/big_sample.edf'
    
    reader = EDF(fname)
  
    start = 0
    stop= int(1e8)
    t0 = time.perf_counter()
    arr_0 = reader.read(start, stop)
    print('Optimized Read Time {}'.format(time.perf_counter() - t0))

    reader._optimized=False
    t0 = time.perf_counter()
    arr_1 = reader.read(start, stop)
    print('Non optimized Read Time {}'.format(time.perf_counter() - t0))
    """
    
    """
    ch_nums = [0,1,2,3]
    pyedf_arr = np.zeros((stop-start, len(ch_nums)))
    t0 = time.perf_counter()
    with pyedflib.EdfReader(fname) as stream:
        for num in ch_nums:
            pyedf_arr[:, num] = stream.readSignal(num, start, stop-start)
    elapsed = time.perf_counter() - t0
    print('PYEDF Read Time {}'.format(elapsed))
    
    print('EDF == pyEDF? {}'.format(np.allclose(arr_0, pyedf_arr)))
    """

