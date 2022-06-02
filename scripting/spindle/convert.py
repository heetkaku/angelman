"""
Merge SPINDLE scores and Pinnacle annotation files and save as CSV 
(delimiter = \t). These CSVs can be used with spectrum.dataset to compute
brain state wise PSD while still removing artifacts. 
This is a preprocessing step for computing brain state wise PSD.
"""

from scripting.utils import spindle_to_pinnacle
from scripting.sleep.dataset import get_paths

def run_conversion(save_dir='./', between=[' Start', ' Stop'], **kwargs):
    """Merge a batch of SPINDLE and pinnalce annotations and save with the same
    name as pinnacle annotations.
    
    Args:
        save_dir (str):     path to saving merged CSV files
        between (2-el-seq): Annotation labels in Pinnacle annotations for start
                            and stop points of analysis.
        kwargs:             kwargs passed to pandas DataFrame.to_csv()
    """
    fpaths, _ = get_paths(t1='Select SPINDLE scores', 
                          t2='Select Pinnacle Annotations')
    for spath, pinpath in fpaths:
        df = spindle_to_pinnacle(spath, pinpath, between=between)
        fname = pinpath.split('/')[-1].split('.')[0] + '.csv'
        save_name = save_dir + fname
        df.to_csv(save_name, sep='\t', index=False, **kwargs)
    
    
if __name__ == '__main__':
    save_dir = '/media/heet/Data_A/heet/projects/angelman/sleep analysis/'\
               '8 weeks rescue post 10 weeks/annotations/annotations/'
    between = [' Start', ' Stop']
    run_converstion(save_dir, between)
