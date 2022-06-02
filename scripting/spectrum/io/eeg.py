import os, inspect, numbers
import numpy as np

from itertools import zip_longest

from scripting.spectrum.io.readers import readers
from scripting.spectrum.io.mixins.view_instance import ViewInstance


class EEG(ViewInstance):
    """A memory map object with data accessor returning numpy ndarrays.
    
    Attrs:
        path (str):         abs path to a datafile
        chunksize (int):    number of samples to fetch during iteration
        reader (obj):       readers.Reader instance

    Implementation Notes: 
        This interface provides two data accessor methods that return
        numpy ndarrays:

        __getitem__: supports integer indices and slices.

        __iter__: supports iteration protocols returning ndarrays of length
                  chunksize

        In the future, this will be turned into a proper subclass of
        ndarrays memmap so that it will support numpy ufuncs.
    """
   
    def __init__(self, path, chunksize=30e6, **kwargs):
        """Initialize this EEG with path to data and cache size."""

        self.path = path
        self.reader = self._fetch_reader()
        self.csize = int(chunksize)

    def _fetch_reader(self):
        """Return a concrete reader instance using path extension."""
        
        #get extension and build dict of available readers
        ext = os.path.splitext(self.path)[1].lstrip('.').upper()
        classes = dict(inspect.getmembers(readers, inspect.isclass))
        #attempt reader build
        try:
            return classes[ext](self.path)
        except KeyError:
            msg = 'Files of type {} not supported'
            raise TypeError(msg.format(ext))

    @property
    def num_channels(self):
        """Returns the number of channels in this EEG."""

        return self.reader.num_channels

    @property
    def sample_rate(self):
        """Returns the sample rate of this EEG channels.
        
        Currently assumes consistent sample rate across chs.
        """

        return np.unique(self.reader.sample_rates)[0]
           

    def __len__(self):
        """Returns the length of this EEG."""
            
        return max(self.reader.num_samples)

    def __getitem__(self, idx):
        """Indexing and slicing method for this EEG.
        
        Args:
            idx (int or slice):     index to retrieve samples from

        Returns:
        """

        if isinstance(idx, numbers.Integral):
            #test if in range and handle negative indices
            idx = range(len(self))[idx]
            return self.reader.read(start=idx, stop=idx+1).squeeze()
        elif isinstance(idx, (slice, tuple)):
            #FIXME when we subclass this hack will be fixed
            #convert idx to 2-tuple of slices
            slices = np.index_exp[idx]
            if len(slices) > 1:
                rows, cols = slices
            else:
                rows, cols = slices[0], slice(None)
            start, stop, step = rows.indices(len(self))
            return self.reader.read(start, stop)[0:stop-start:step, cols]
        else:
            #This msg should be consistent with numpy when we subclass EEG
            msg = '{} indices must be integers or slices not {}'
            cls = type(self)
            idx_cls = type(idx)
            raise IndexError(msg.format(cls, idx_cls))

    def __iter__(self):
        """Returns an iterator producing arrays of length chunksize.
        
        Yields arrays of shape (chunksize X num_channels).
        """
       
        #get the start of each epoch
        starts = range(0, len(self), self.csize)
        epochs = zip_longest(starts, starts[1:], fillvalue=len(self))
        #read and slice each each epoch yielding sliced data
        for epoch in epochs:
            yield self.reader.read(*epoch)
    
    def __contains__(self, sample):
        """Returns True if sample is found in instance."""

        for chunk in self:
            if (sample==chunk).all(axis=1).any():
                return True


if __name__ == '__main__':

    #FIXME replace this with a tkinter dialog
    path = '/media/heet/Data_A/heet/projects/angelman/pinnacle files/EDF/3 weeks rescue post 6 weeks/DL0AA3_P067_nUbe3a_16_54_2dayEEG_2019-09-13_13_01_33.edf'
    
    #create an eeg object
    eeg = EEG(path)
   
    def show_usage():
        """Demonstrates basic usage of eeg instances."""

        #print info about the eeg
        print(eeg)
        print('\n---Usage---\n')

        #print the length of this eeg
        print('eeg length = ', len(eeg))
        
        #get sample of this eeg (see __getitem__ )
        print('picking random sample...')
        rand_sample = np.random.randint(0, len(eeg))
        value = eeg[rand_sample]
        print('sample {} has value {}'.format(rand_sample, value))

        #slice the eeg
        arr_0 = eeg[0:20]
        arr_1 = eeg[0:20:2, 0:2]
        print(arr_0)
        print(arr_1)
        
        #determine if the 3rd row of arr is in sliced_eeg
        print('Is 3rd row of data in the sliced eeg? ', arr_0[3] in eeg)
