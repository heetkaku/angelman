import csv
import sys
import inspect
import itertools
import numpy as np
import scipy.io as sio
from itertools import zip_longest
from datetime import datetime

from scripting.spectrum.io import dialogs
from scripting.spectrum.io import paths

class CSVReader(csv.DictReader):
    """An Extended DictReader object that can start at an arbitrary row of a
    file obj or start at the first row containing a token string."""

    def __init__(self, csvfile, token=None, start=None, **kwargs):
        """Initialize this Reader.

        Args:
            csvfile (file):         a file obj with a next protocol yielding
                                    row strings
            token (str):            a str that defines the start row for
                                    this Reader
            start (int):            row index to begin read
            **kwargs                passed to csv.DictReader

        Note: one of token or start must be specified
        """

        self.validate(token, start)
        start = start if start is not None else self.find(csvfile, token)
        [next(csvfile) for _ in range(start)]
        super().__init__(csvfile, **kwargs)

    def validate(self, token, start):
        """Raise TypeError if neither token or start is specified."""

        if token is None and start is None:
            print(token, start)
            msg = 'one of token or start must be specified.'
            raise TypeError(msg)

    def find(self, csvfile, token):
        """Returns the first row containing token in csvfile."""

        for idx, row in enumerate(csvfile):
            if token.lower() in row.lower():
                #seek back to start row
                csvfile.seek(0)
                return idx

def instring(substrings, string):
    """Returns True if any substrings are in string and False otherwise."""

    for substr in substrings:
        if substr.lower() in string.lower():
            return True
    return False

def read_csv(path, names, token=None, start=None, **kwargs):
    """Reads the csv file at path with a CSVReader instance restricted to
    file columns containing names.

    Args:
        path (str):         path to csv file. If None open dialog
        names (seq):        seq of substrs specifying columns to return
        token (str):        a str that defines the start row of this reader.
                            Reading begins directly after this row
        start (int):        row index of header row

    Note: One of token or start must be specified
    """

    indir = paths.DATA_DIR
    if not path:
        path = dialogs.standard('askopenfilename', initialdir=indir)
    results = []
    with open(path, 'r') as csvfile:
        if start is not None:
            reader = CSVReader(csvfile, start=start, **kwargs)
        else:
            reader = CSVReader(csvfile, token=token, **kwargs)
        for row in reader:
            res = {k:row[k] for k in row.keys() if instring(names, k)}
            results.append(tuple(res.values()))
    return results

def pinnacle(path, *annotations, **kwargs):
    """Reads a pinnacle formatted CSV file and returns an event start time
    and duration for each annotation in annotations.

    path (str):         path to a pinnacle formatted CSV file
    *annotations:       annotation name/names for which timing info
                        is returned
    """

    indir = paths.DATA_DIR
    if not path:
        path = dialogs.standard('askopenfilename', initialdir=indir)
    results = []
    names = annotations if annotations else ['']
    with open(path, 'r') as csvfile:
        for row in CSVReader(csvfile, token='Annotation', **kwargs):
            #get annotation
            annotation = row['Annotation'].lower()
            if not any([n.lower() in annotation for n in names]):
                continue
            #create a format for datetime objs
            fmt = '%m/%d/%y %H:%M:%S.%f'
            start = datetime.strptime(row['Start Time'], fmt)
            stop = datetime.strptime(row['End Time'], fmt)
            #return duration of event and event time from trace start
            event_time = float(row['Time From Start'])
            duration = (stop - start).total_seconds()
            results.append((event_time, duration, annotation))
    return results

def pinnacle_sleep(path, epoch_length, *annotations, **kwargs):
    """Reads a pinnacle sleep formatted TSV file and returns an event start time
    and duration for each annotation in annotations.

    path (str):         path to a pinnacle sleep formatted TSV file
    epoch_length (int): length of each epoch in seconds
    *annotations:       annotation name/names for which timing info
                        is returned
    """
    indir = paths.DATA_DIR
    if not path:
        path = dialogs.standard('askopenfilename', initialdir=indir)
    results = []
    names = annotations if annotations else ['']
    with open(path, 'r') as csvfile:
        for row in CSVReader(csvfile, token='Date', **kwargs):
            #get annotation
            #The column name/key is not fixed here but always ends with Numeric
            #Added an extra "if key" check because one of the keys/columns
            #(the last one) is of None type. Its just how the sleep software
            #exports the TSV file
            annotation = [float(val) for key, val in row.items()
                          if key if key.endswith('Numeric')][0]
            #Added this to get rid of the decimal and remove whitespaces
            annotation = str(annotation).lower().split('.')[0].strip()

            if not any([n.lower() == annotation for n in names]):
                continue
            event_time = float(row['Time from Start'])
            results.append((event_time, epoch_length, annotation))
    return results

def mat(path, name='DEL_ts', annotation='rpsp', **kwargs):
    """Reads a Matlab file containing an array with start, stop times in
    secs for a specific annotation.

    Args:
        path (str):         path to a matlab annotation file
        name (str):         name of times array stored in mat file
        annotation (str):   annotation associated with each start/stop time
        kwargs (dict):      passed to scipy loadmat
    """

    indir = paths.DATA_DIR
    if not path:
        path = dialogs.standard_dialog('askopenfilename', initialdir=indir)
    data_dict = sio.loadmat(path, **kwargs)
    times = data_dict[name]
    starts = np.around(times[:, 0], decimals=3)
    durations = np.around(np.squeeze(np.diff(times, axis=1)), decimals=3)
    res = zip_longest(starts, durations, annotation, fillvalue=annotation)
    return list(res)



if __name__ == '__main__':

    fp='/home/giladmeir/python/nri/data/eeg/annotations/heets_samples/'+\
    'group_info_3week.csv'

    #res = pinnacle(None, 'rest', delimiter='\t')
    res = read_csv(fp, names=['mouse', 'aso'], token='mouse', start=None)

