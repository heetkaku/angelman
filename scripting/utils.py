# Utility scripts used by sleep, spectrum & spindle
import pandas as pd
import numpy as np
import copy
import datetime

from scipy.interpolate import interp1d
from scripting.spectrum.metrics import freq_index
from scripting.spectrum.dataset import extract

def spindle_to_pinnacle(spath, pinpath, between=None, fmt='%m/%d/%y %H:%M:%S.%f'):
    """Merge SPINDLE sleep scores & Pinnacle annotation files. 
    Can then be used with spectrum to compute PSD of brain states.
    
    Args:
        spath (str):        path to SPINDLE scores CSV
        pinpath (str):      path to corresponding pinnacle annotation file.
        fmt (str):          datetime format for converted dataframe
        between (2-seq):    Optional. Analysis start & stop labels in pinpath CSV. 
        
    
    Returns DataFrame in Pinnacle format
    """
    offset = 0
    res = pd.read_csv(pinpath, sep='\t', skiprows=6) #read pinnacle annotation
    start = datetime.datetime.strptime(res['Start Time'].iloc[0], fmt)
    t = res['Time From Start'].iloc[0]
    start = start - datetime.timedelta(seconds=t)
    start = start.strftime(fmt)
    if between: #get start timepoint to match SPINDLE score timestamps   
        s = res.loc[res['Annotation'].str.contains(between[0], case=False)]
        offset = s.iloc[0]['Time From Start']
        start = s.iloc[0]['Start Time']
    df = pd.read_csv(spath, header=None, sep=',') #read SPINDLE scores
    num_rows = len(df.index)
    df = df.rename(columns={0: "Number", 1: "Annotation"})
    #replace scores incase SPINDLE outputted artifacts
    df.Annotation.replace(['1','2','3'], ['w','n','r'] , inplace=True)
    #SPINDLE epoch length is always 4s
    df["Time From Start"] = (df["Number"] * 4) + offset 
    df["Channel"] = "ALL" #add column to match pinnacle format
    df["Start Time"] = pd.date_range(start=start, periods=num_rows, freq="4s")
    df['End Time'] = pd.date_range(start=df["Start Time"][1], periods=num_rows,
                                   freq='4s')
    df["Start Time"] = df["Start Time"].dt.strftime(fmt)
    df["End Time"] = df["End Time"].dt.strftime(fmt)
    
    df = df[list(res.columns)] #re-arrange columns to match pinnacle format
    df = pd.concat([df, res], ignore_index=True) #concat SPINDLE and pinnacle
    df = df.sort_values('Time From Start')
    df['Number'] = range(0, len(df.index))
    return df

def interpolate(dset, annote='ignore', criticals=(56, 64), nbpts=10, axis=1,
                **kwargs):
    """Linearly interpolate PSD between two frequencies using scipy's interp1d
    
    Args:
        dset (dataset):         a dict containing a 'data' field containing
                                PSDs
        annote (str):           Annotation to interpolate (should be present
                                in dset['annotations'])
        criticals (list):       pair of freqs to interpolate between
        nbpts (int):            number of points to consider before and after
                                criticals for interpolation
        axis (int):             frequency axis in dset['data']
        kwargs:                 kwargs passed to scipy.interpolate.interp1d
    """
    msg = ('WARNING, setting values in the dataset is PERMANENT.' 
            'Press "y/yes/Y/Yes" to proceed.\n')
    user = input(msg)
    if not user.lower() in ('y', 'yes'):
        print('dataset unchanged')
        return
    
    ann = dset['annotations'].index(annote) #get annote index
    freqs = dset['frequencies']
    start, stop = [freq_index(freqs, f) for f in criticals]
    arr = extract(dset, annotations=[annote]) #extract array
    a = freqs[start-nbpts:start] 
    b = freqs[stop:stop+nbpts]
    x = np.concatenate((a, b)) #freqs to use for interpolation
    ix = [freq_index(freqs, f) for f in x] #indices of those freqs
    y = np.take(arr, indices=ix, axis=axis) #extract array at those freqs
    f = interp1d(x, y, kind='linear', axis=axis, **kwargs) #interpolation func
    newx = freqs[start+1:stop] #freqs to interpolate
    newy = f(newx) #interpolated y-values for newx
    arr.swapaxes(0, axis)[start+1:stop] = newy.swapaxes(0, axis) #replace old vals in arr
    dset['data'][ann] = arr #replace in dset

def replace(arr, tokens):
    """Replace element(s) in 1D array with new value(s) 
    
    Args:
        arr (1d array):     1-d numpy array
        tokens (list):      list of tuples containing old and new value
    NOTE: Makes a deep copy of arr. Does NOT replace in place.
    """
    res = copy.deepcopy(arr)
    for a, b in tokens:
        res[np.where(arr == a)[0]] = b
    return res
