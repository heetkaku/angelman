"""
AS sleep build using SPINDLE output.
"""
import time
from scripting.sleep import dataset
from scripting.spectrum.io import persist

SAVE_PATH = './sleep_dataset.pkl'

#Build dataset
t0 = time.perf_counter() #initiate timer
dset = dataset.build() #see dataset.py
delta_t = time.perf_counter() - t0
print('Dataset built in {} s.'.format(delta_t))

#save the dataset
try:
    persist.pkl_save(SAVE_PATH, dset)
    print('Dataset successfully saved to {}'.format(SAVE_PATH))
except Exception as e:
    raise(e)

