"""
Create confusion matrices to evaluate sleep scoring
build_spindle(): evaluates SPINDLE (Predicted) and Pinnacle sleep (Manual/True)
build_expert(): evaluates two experts, both using Pinnacle sleep to score
"""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score

from scripting.spectrum.io import dialogs, paths
from scripting.spectrum.io.annotations import pinnacle_sleep
from scripting.utils import replace

def build_spindle(labels=['a','w','n','r'], **kwargs):
    """Build dataset containing confusion matrices comparing SPINDLE to an
    expert. Expects manual scores from pinnacle sleep module & predicted scores
    from SPINDLE.
    
    Args:
        labels (seq):    confusion matrix labels
        kwargs:          kwargs passed to pandas.read_csv()
    
    Returns dict with confusion matrices stored under 'matrix'
    """
    fpaths = dialogs.matched(titles=['Select Manual Scores', 
                                     'Select Predicted Scores'],
                             initialdir = paths.DATA_DIR)
    result, names, accuracy = [], [], []
    annotations = ['1', '2', '3', '0', '129', '130', '131']
    tokens = [('1', 'w'), ('2', 'n'), ('3', 'r'), ('0', 'a'), ('129', 'w'),
              ('130', 'n'), ('131', 'r')]  
    for manual_path, predicted_path in fpaths:
        name = str(manual_path).split('/')[-1].split(".")[0]
        names.append(name)
        manual = pinnacle_sleep(manual_path, 4, *annotations, 
                                delimiter='\t')
        predicted = pd.read_csv(predicted_path, sep=',', **kwargs)
        y_true = np.array([score for _, _, score in manual])
        y_pred = predicted.iloc[:, 1].to_numpy()
        
        #map scores to make them consistent
        y_true = replace(y_true, tokens)
        y_pred = replace(y_pred, tokens)
        
        accuracy.append(accuracy_score(y_true, y_pred))
        c = confusion_matrix(y_true, y_pred, labels=labels)
        result.append(c)
        
    return {'matrix': np.array(result), 'names': names, 'matrix labels': labels,
            'accuracy': np.array(accuracy)}


def build_expert(labels=['a','w','n','r']):
    """Build dataset containing confusion matrices comparing two experts. 
    Expects scores output from pinnacle sleep module
    
    Args:
        labels (seq):    confusion matrix labels
    
    Returns dict with confusion matrices stored under 'matrix'
    """
    fpaths = dialogs.matched(titles=['Select Expert1 Scores', 
                                     'Select Expert2 Scores'],
                             initialdir=paths.DATA_DIR)
    result, names, accuracy = [], [], []
    annotations = ['1', '2', '3', '0', '129', '130', '131']
    tokens = [('1', 'w'), ('2', 'n'), ('3', 'r'), ('0', 'a'), ('129', 'w'),
              ('130', 'n'), ('131', 'r')]  
    for manual_path, predicted_path in fpaths:
        name = str(manual_path).split('/')[-1].split(".")[0]
        names.append(name)
        manual = pinnacle_sleep(manual_path, 4, *annotations, 
                                delimiter='\t')
        predicted = pinnacle_sleep(predicted_path, 4, *annotations, 
                                delimiter='\t')
        y_true = np.array([score for _, _, score in manual])
        y_pred = np.array([score for _, _, score in predicted])
              
        #map scores to make them consistent
        y_true = replace(y_true, tokens)
        y_pred = replace(y_pred, tokens)

        accuracy.append(accuracy_score(y_true, y_pred))
        c = confusion_matrix(y_true, y_pred, labels=labels)
        result.append(c)
        
    return {'matrix': np.array(result), 'names': names, 'matrix labels': labels,
            'accuracy': np.array(accuracy)}


def metrics(matrix, metric='precision', kind='percentage'):
    """Calculate precision, recall or f1-score for a single confusion matrix
    
    Args:
        matrix (2d array):      confusion matrix
        metric (str):           precision, recall or f1
        kind (str):             if kind is percentage, values will be multiplied
                                by 100
    Returns a 2d matrix of desired metric
    """
    result = np.empty(shape=matrix.shape)
    if  metric.lower() in 'precision':
        for ix in range(matrix.shape[1]):
            col = matrix[:, ix]
            denominator = np.sum(col)
            col = col / denominator
            result[:, ix] = col
    
    elif metric.lower() in 'recall':
        for ix in range(matrix.shape[0]):
            row = matrix[ix, :]
            denominator = np.sum(row)
            row = row / denominator
            result[ix, :] = row
    
    elif metric.lower() in 'f1-score':
        pr = metrics(matrix, metric='precision', kind='fraction')
        re = metrics(matrix, metric='recall', kind='fraction')
        result = (2*pr*re) / (pr+re)
    
    else: 
        print('Invalid metric entry. Returning None.')
        return None
    
    result = result*100 if 'percent' in kind.lower() else result
    return np.nan_to_num(result, copy=False)
