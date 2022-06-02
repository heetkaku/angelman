import os
import pickle

def pkl_save(path, obj, **kwargs):
    """Saves a serializable obj to a pickle (.pkl) file."""

    path = os.path.splitext(path)[0] + '.pkl'
    with open(path, 'wb') as outfile:
            pickle.dump(obj, outfile, **kwargs)

def pkl_load(path, **kwargs):
    """Loads a pickled obj from path."""

    with open(path, 'rb') as infile:
        result = pickle.load(infile)
    return result

