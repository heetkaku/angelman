import os
import re

from scripting.spectrum.io.annotations import read_csv
from scripting.spectrum.io import dialogs

def from_csv(path, start, key_column, *columns, **kwargs):
    """Returns a dict of metadata from a csv file with key_column as
    keys and columns as dict values.

    Metadata for a dataset is stored to a csv file with a column 
    specifying the animal/experiment (i.e. a "key_column") and
    columns of metadata to associate with the key column. from_csv
    constructs a dict of this metadata.
    
    Args:
        path (str):                 path to csv file
        start (int):                line number of file header row
        name_column (str):          name column to use as dict keys
        *columns:                   seq of column names to use as dict
                                    values
        **kwargs:                   passed to read_csv
    """

    tuples = read_csv(path, [key_column]+list(columns), start=start, **kwargs)
    return dict([(tup[0], list(tup[1:])) for tup in tuples])

def from_paths(paths, regex, *regexes):
    """Returns a dict of metadata from a list of file paths with regex pair
    match as key and regexes pair matches as values.

    Animal/Experiment ID and associated metadata will is stored into 
    filename. The regex expression tuple extracts the animal/experiment id
    and the regexes extract the associated metadata. A dict of this data is
    returned.
    
    Args:
        paths (seq):            seq. of paths to extract metadata from
        regex (tuple):          a re expression and group integer tuple
                                whose match is a key for metadata dict
        *regexes (seq):         seq. of regex tuples whose matches are
                                values for metadata dict
    """

    meta = {}
    for path in paths:
        #split off dirs
        fname=os.path.split(path)[1]
        #perform regular expression searches
        try:
            key = re.search(regex[0], fname).group(regex[1])
            vals = [re.search(r[0], fname).group(r[1]) for r in regexes]
            meta[key] = vals
        except Exception as e:
            print('regex error for file {}'.format(fname))
            raise e
    return meta

def find(metadata, *tokens):
    """Returns the keys of metadata for keys containing vals.
    
    Args:
        metadata (dict):        collection of animal/experiment ids and
                                associated metadata.
        *tokens:                values to filter metadata.values() by
    """

    result = []
    for key, values in metadata.items():
        if all(token in values for token in tokens):
            result.append(key)
    return result


if __name__ == '__main__':


    path = dialogs.standard('askopenfilename')
    meta = from_csv(path, 0, 'Animal ID', *['Treatment'], delimiter='\t')
    # meta = from_paths(paths, (r'(.+)\.', 1), (r'[^_]+_([^_]+)', 1),
    #                    (r'\w+\s(\w)', 1), (r'\w+\s\w\-(\w)', 1))
    print(meta)
    result = find(meta, 'a')

