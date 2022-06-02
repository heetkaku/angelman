import numpy as np
import pandas as pd

from scripting.spectrum.io import dialogs, paths
from scripting.spectrum import metadata
from scripting.utils import replace

def get_paths(t1="Scores", t2="Artifact Probs", matched_chars=11):
    """Get matched paths for scores and artifact probabilities text files along
    with filenames
    
    Args:
        t1 (str):            Title for first askopenfilenames dialog
        t2 (str):            Title for second askopenfilenames dialog
        matched_chars (int): number of characters to use for matching filenames
    
    Returns a list of matched tuples containing score and artifact filepaths & 
    a list of corresponding file names
    
    NOTE: Not using dialogs.matched because filenames are not an exact match.
    Instead, matching by ID and age here which is unique enough for this purpose.
    """
    names = []
    fpaths = []
    spaths = dialogs.standard('askopenfilenames', title=t1, 
                              initialdir=paths.DATA_DIR)
    apaths = dialogs.standard('askopenfilenames', title=t2, 
                              initialdir=paths.DATA_DIR)
    for apath in apaths:
        name = apath.split('/')[-1].split('.')[0][:matched_chars] 
        idx = [ix for ix, el in enumerate(spaths) if name in el][0]
        fpaths.append((spaths[idx], apath))
        names.append(name)
    return fpaths, names

def read(fpaths):
    """Read in matched scores and artifacts probabilities and return a merged
    array.
    
    Args:
        fpaths (list):      list of tuples containing path to scores and 
                            corresponding artifact probabilities
    Returns an array with dimensions of names x epochs x score
    
    NOTE: Replaces 1, 2 & 3 with w, n & r respectively. This is to ensure SPINDLE
    did not consider an artifact threshold probability while generating labels
    since their website function for artifact slider has bugs. Use result[:, 2]
    to look for artifacts if needed.
    """
    result = []   
    tokens = [('1', 'w'), ('2', 'n'), ('3', 'r')]
    for spath, apath in fpaths:
        scores = pd.read_csv(spath, sep=',', header=None) #read scores
        probs = pd.read_csv(apath, sep=',', header=None) #read artifact probs
        scores['probs'] = probs.iloc[:, 1] #concat probs to scores
        arr = scores.to_numpy() #convert to numpy ndarray
        arr[:, 1] = replace(arr[:, 1], tokens)
        result.append(arr)
    return np.array(result)

def build():
    """Build dataset for sleep analysis. Uses output files from SPINDLE"""
    fpaths, names = get_paths()
    result = read(fpaths)
    meta_path = dialogs.standard('askopenfilename', title='Select Metadata', 
                                 initialdir=paths.DATA_DIR)
    meta = metadata.from_csv(meta_path, 0, 'Animal ID', *['Treatment'], 
                             delimiter='\t')
    #Esnure dset['names'] & dset['metadata'].keys() match so that subsequent data 
    #extraction can be done using spectrum.dataset.extract()
    for ix, name in enumerate(names):
        keys = list(meta.keys())
        idx = [i for i, key in enumerate(keys) if name in key][0]
        names[ix] = keys[idx]
        
    dset = {'data': result, 'fpaths': fpaths, 'names': names, 'metadata': meta,
            'axes': ['names', 'epochs', 'score']}
    return dset
