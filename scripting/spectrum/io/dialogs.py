import tkinter as tk
import tkinter.filedialog as tkdialogs
import os
from pathlib import Path

def root_deco(dialog):
    """Decorates a dialog with a toplevel that is destroyed when the dialog
    is closed."""

    def decorated(*args, **kwargs):
        #create root and withdraw from screen
        root = tk.Tk()
        root.withdraw()
        #open dialog returning result and destroy root
        result = dialog(*args, parent=root, **kwargs)
        root.destroy()
        return result
    return decorated

@root_deco
def standard(kind, **options):
    """Opens a standard tkinter modal file dialog and returns a result.

    Args:
        kind (str):             name of a tkinter dialog
    **options:
        parent (widget):        ignored
        title (str):            title of the dialog window
        initialdir (str):       dir dialog starts in 
        initialfile (str):      file selected on dialog open
        filetypes (seq):        sequence of (label, pattern tuples) '*'
                                wildcard allowed
        defaultextension (str): default ext to append during save dialogs
        multiple (bool):        when True multiple selection enabled
    """

    return getattr(tkdialogs, kind)(**options)

def matched(titles=['', ''], **options):
    """Opens two standard dialogs and matches the results the path stems.
    
    Args:
        titles (seq):           2-el seq of string titles one per dialog
        **options:              passed to standard dialog
    """

    res = []
    t0, t1 = titles
    #dialog for paths
    fpaths = standard('askopenfilenames', title=t0, **options)
    opaths = standard('askopenfilenames', title=t1, **options)
    #convert to path objects and get stems
    fpaths = [Path(el) for el in fpaths]
    opaths = [Path(el) for el in opaths]
    ostems = [op.stem for op in opaths]
    #match stems
    for fpath in fpaths:
        idx = ostems.index(fpath.stem)
        res.append((fpath, opaths[idx]))
    return res



if __name__ == '__main__':

    from scripting.spectrum.io.paths import DATA_DIR
    
    paths = matched(['Select EDFs', 'Select Annotations'],
                    initialdir=DATA_DIR)


