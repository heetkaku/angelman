import numpy as np
from scripting.spectrum.io.annotations import pinnacle

def from_csv(path, span, fs, value, *annotations, reader=pinnacle, **kwargs):
    """Creates a boolean masking array from an annotation in a csv at path.

    If annotations at path do not contain annotation the returned mask is
    ~value at all mask indices.
    
    Args:
        path (str):         path to annotation file containing annotations
        span (int):         length of mask in samples
        fs (int):           sampling rate of recording system
        value (bool):       boolean specifying if annotation should be
                            included (True) or excluded (False) from mask
        annotations (str):  annotations to mask
        reader (callable):  func returning start and duration of annotation
        **kwargs:           passed to reader

    Returns: boolean numpy array of shape (span,)
    """
    
    #read annotations and convert to start, stop event samples
    annotes = reader(path, *annotations, **kwargs)
    events = [(start, start + duration) for start, duration, _ in annotes]
    events = (np.array(events) * fs).astype(int)
    #build boolean logical
    arr = np.logical_not(value) * np.ones(span, dtype=bool)
    if events.size > 0:
        idxs = np.concatenate([np.arange(*row) for row in events])
        arr[idxs] = value
    return arr

def intersect(*masks):
    """Returns the intersection of a sequence of masks."""

    return np.logical_and(*masks)

def union(masks):
    """Returns the union of a sequence of masks."""

    return np.logical_or(*masks)


if __name__ == '__main__':

    from scripting.spectrum.io import dialogs
    from scripting.spectrum.io.eeg import EEG
    import matplotlib.pyplot as plt

    ANNOTE_1 = ['rest', 'exploring']
    ANNOTE_2 = ['grooming']

    ann_path = dialogs.standard('askopenfilename')
    eeg_path = dialogs.standard('askopenfilename')

    eeg = EEG(eeg_path, csize=1e5)
    masks = []
    events = []
    for ANNOTE in [ANNOTE_1, ANNOTE_2]:
        annotes = pinnacle(ann_path, *ANNOTE, delimiter='\t')
        tups = [(start, start + duration) for start, duration, _ in annotes]
        events.append((np.array(tups) * eeg.sample_rate).astype(int))

        masks.append(from_csv(ann_path, len(eeg), eeg.sample_rate, True,
                              *ANNOTE, delimiter='\t'))
    unioned = union(masks)
    print(events)
    plt.plot(unioned)
