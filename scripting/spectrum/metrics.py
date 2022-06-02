import numpy as np
from scipy.integrate import simps
import pickle

def freq_index(frequencies, freq):
    """Returs the index of frequencies sequence closest to freq.

    Args:
        frequencies (array_like):   frequencies to index from
        freq (int / float):         frequency of interest to look for
    """

    return np.argmin(np.abs(frequencies-freq))

def band_amplitudes(arr, freqs, bands=[(1,4),(4,8),(8,13),(13,30),(30,100)], 
                    relative=False, cutoffs=[1, 100], axis=-1):
    """Returns the average spectrum amplitude of an array in each band.

    Args:
        arr (ndarray):      array over which band average will be computed
        freqs (array_like): frequencies to index from
        bands (seq.):       sequence of start, stop frequency tuples in which
                            average amplitude will be computed
        relative (bool):    boolean for computing relative band ampltiude
        cutoffs (seq):      pair of low and high values to normalize to if
                            relative amplitude requested (Defaults to 1-100Hz)
        axis (int):         frequency axis of array

    Returns:
    """

    #preallocate
    shape = list(arr.shape)
    shape[axis] = len(bands)
    result = np.zeros(shape)
    for idx, band in enumerate(bands):
        #get average value of amplitude in band
        start, stop = [freq_index(freqs, f) for f in band]
        vals = np.take(arr, np.arange(start, stop), axis=axis)
        avg = np.mean(vals, axis=axis, keepdims=True)
        if relative: #compute normalizer and noramalize
            a, b = [freq_index(freqs, cutoff) for cutoff in cutoffs]
            v = np.take(arr, np.arange(a, b), axis=axis)
            n_avg = np.mean(v, axis=axis, keepdims=True)
            avg = avg / n_avg
        #construct slice tuple to place avg into
        slc = [slice(None)] * result.ndim
        slc[axis] = slice(idx, idx+1)
        result[tuple(slc)] = avg
    return result

def band_powers(arr, freqs, bands=[(1,4),(4,8),(8,13),(13,30),(30,100)], 
                relative=False, cutoffs=[1, 100], axis=-1):
    """Returns the power for each band in array.

    Args:
        arr (ndarray):      array over which band power will be computed
        freqs (array_like): frequencies to index from
        bands (seq.):       sequence of start, stop frequency tuples in which
                            average will be computed
        relative (bool):    flag for computing relative band power
        cutoffs (seq):      pair of start and stop freqs to normalize power if
                            relative is True (Defaults to 1-100Hz)
        axis (int):         frequency axis of array

    Returns np.array
    """
    #preallocate
    shape = list(arr.shape)
    shape[axis] = len(bands)
    result = np.zeros(shape)
    #get freq resolution
    resolution = freqs[1] - freqs[0]
    for idx, band in enumerate(bands):
        #get average value of power in band
        start, stop = [freq_index(freqs, f) for f in band]
        vals = np.take(arr, np.arange(start, stop), axis=axis)
        power = simpsons(vals, dx=resolution, axis=axis)
        if relative: #compute normalizer and normalize
            a, b = [freq_index(freqs, cutoff) for cutoff in cutoffs]
            v = np.take(arr, np.arange(a, b), axis=axis)
            n_power = simpsons(v, dx=resolution, axis=axis)
            power = power / n_power
        power = np.expand_dims(power, axis=axis)
        #construct slice tuple to place power into
        slc = [slice(None)] * result.ndim
        slc[axis] = slice(idx, idx+1)
        result[tuple(slc)] = power
    return result

def simpsons(array, **kwargs):
    """Computes the area below array using Simpson's quadrature rule.

    Args:
        array (np.arr):  array to be integrated
        kwargs (dict):   kwargs passed to scipy.simps

    Returns: a float of the area below array
    """

    return simps(array, **kwargs)


if __name__ == '__main__':
    pass
