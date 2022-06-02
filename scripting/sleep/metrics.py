import numpy as np
from itertools import zip_longest
from scripting.spectrum import metadata
from scripting.spectrum.dataset import extract

def get_epochs(data, num_seg):
    """Get (start, stop) ranges for data.shape[1] divided into 'num_seg'
    
    Args:
        data (ndarray):     score data from dset. (dset['data'])
        num_seg (int):      number of segments
    
    Returns a list of (start, stop) ranges
    """
    stop = data.shape[1]
    step = stop // num_seg
    starts = range(0, stop, step)
    return list(zip_longest(starts, starts[1:], fillvalue=stop))
    

def distribution(data, art_threshold=0.5):
    """Calculate proportion of time spent in artifact, Wake, REM & NREM 
    
    Args:
        data (ndarray):         score array (names x epochs x scores). Columns
                                (last dimension) is in oder of epoch #, score, 
                                artifact probability
        art_threshold (float):  probability greater than which to consider
                                artifact
    
    Returns ndarray of dims --> state x animal
    states are always artifact, wake, NREM, REM (i.e result.shape[0] = 4)
    """
    t = data.shape[1] #total num of epochs
    a, w, n, r = [], [], [], [] #one list for artifact, wake, REM, NREM
    for arr in data.astype(str):
        _a = len(np.where(arr[:, -1].astype(float) > art_threshold)[0]) / t
        _w = len(np.where((arr[:, 1] == 'w'))[0]) / t
        _n = len(np.where((arr[:, 1] == 'n'))[0]) / t
        _r = len(np.where((arr[:, 1] == 'r'))[0]) / t
        a.append(_a)
        w.append(_w)
        n.append(_n)
        r.append(_r)
    return np.array([a, w, n, r])
    
    
def get_distribution(dset, group, num_seg=1, art_threshold=0.5):
    """Calculate proportion of time spent in artifact, Wake, REM & NREM for 
    each segment for group.
    
    Args:
        dset (dict):            sleep dataset (using build.py)
        group (str):            group to extract. 'a', 'b', 'c' or 'd'
        num_seg (int):          # of segments to divide dset['data']
        art_threshold (float):  probability greater than which to consider
                                artifact
    
    Returns ndarray of dims --> state x segment x animal
    states are always artifact, wake, NREM, REM (i.e result.shape[0] = 4)
    """
    result = []
    names = metadata.find(dset['metadata'], group)
    arr = extract(dset, names=names)
    epochs = get_epochs(arr, num_seg)
    for a, b in epochs:
        d = distribution(arr[:, a:b, :], art_threshold)
        result.append(d)
    return np.array(result).swapaxes(0, 1)
