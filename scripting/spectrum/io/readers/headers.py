import collections
import os

def read_header(file_path):
    """Returns a dict of all header items for an EEG data file.
    
    Args:
        file_path (str):        full file path to an EEG data file
    """
    
    #use the files extension to dispatch to reader
    if file_path.endswith('edf'):
        return EDF_Header(file_path).read()
    else:
        _, ext = os.path.splitext(file_path)
        msg = 'Files of type {} not currently supported'.format(ext)
        raise NotImplementedError(msg)

class EDF_Header:
    """EDF specific file header reader.
    
    Attrs:
        path (str):        path to EDF file
    """

    def __init__(self, file_path):
        """Initialize this header reader with the path to the data."""

        self.path = file_path

    def _file_map(self):
        """Constructs a mapping of header item locations.

        Locations are specified with a start byte location, the number of
        bytes to read and a function to apply to the decoded bytes.

        Returns dict of locations keyed on header names
        """

        file_map = {'version': (0, 8, str), 
                    'patient': (8, 80, str), 
                    'recording': (88, 80, str),
                    'start_date': (168, 8, str),
                    'start_time': (176, 8, str),
                    'header_bytes': (184, 8, int),
                    'reserved': (192, 44, str),
                    'num_records': (236, 8, int),
                    'record_duration': (244, 8, float),
                    'num_signals': (252, 4, int)}

        return file_map

    def _signal_map(self, file_head):
        """Constructs a mapping of signal header item locations.
        
        Args:
            file_head (dict):       decoded file_header mapping (see
                                    _read_header)

        Signal header item locations are a list of tuples specifying where
        to start reading in the file, how many bytes to read and what func
        to apply to the decoded bytes. Since the starts depend on the number
        of signals we must have the file_header ready.

        Returns (dict):         map of list of tuples (one per signal) with
                                each tuple containing the start, read bytes 
                                and func
        """

        #get the number of signals in file
        ns = file_head['num_signals']
        #initial section start byte is end of file header
        section_start = 256
        def sections(start, inc, ns, func):
            """Returns list of sublist sections to read, one per signal.
            
            Args:
                start (int):        location to start read
                inc (int):          number of bytes to read for each signal
                ns (int):           number of signals in edf
                func (callable):    callable to apply to decoded bytes
            """

            #compute the starts one per signal in a section
            starts = [start + inc * i for i in range(ns)]
            #update the section_start
            nonlocal section_start
            section_start = start + inc * ns
            #return the start, bytes to read and func to apply
            return [(start, inc, func) for start in starts]

        signal_map = {
               'names': sections(section_start, 16, ns, str),
               'transducers': sections(section_start, 80, ns, str),
               'physical_dim': sections(section_start, 8, ns, str),
               'physical_min': sections(section_start, 8, ns, float),
               'physical_max': sections(section_start, 8, ns, float),
               'digital_min': sections(section_start, 8, ns, float),
               'digital_max': sections(section_start, 8, ns, float),
               'prefiltering': sections(section_start, 80, ns, str),
               'samples_per_record': sections(section_start, 8, ns, int),
               'reserved': sections(section_start, 32, ns, str)}
        
        return signal_map

    def _read_header(self, header_map):
        """Reads a file or signal header.
        
        Args:
            header_map (dict):      map of locations to read
        """

        def read_attr(infile, start, nbytes, func=None):
            """Reads and decodes nbytes from infile and applies func."""

            s = infile.seek(start)
            value = infile.read(nbytes).strip().decode()
            return func(value) if func else value
        
        result = dict()
        with open(self.path, 'rb') as infile:
            for name, vals in header_map.items():
                if isinstance(vals, collections.MutableSequence):
                    #header_map is signal map (list of tups as vals)
                    result[name] = [read_attr(infile, *val) for val in vals]
                else:
                    #header_map is a file mapping (single tuple as value)
                    result[name] = read_attr(infile, *vals)
        return result

    def read(self):
        """Returns a header map including file metadata and signals
        metadata."""

        #get file_map and read file header
        file_map = self._file_map()
        file_head = self._read_header(file_map)
        #build signal map and read signal header
        signal_map = self._signal_map(file_head)
        signal_head = self._read_header(signal_map)
        return file_head, signal_head 






if __name__ == '__main__':

    fpath = '/home/giladmeir/python/nri/open_seize/__data__/' + \
            'edf_samples/big_sample.edf'

    header = EDF_Header(fpath)




