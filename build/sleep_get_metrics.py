from scripting.spectrum.io import persist
from scripting.sleep.metrics import get_distribution

#path to the saved dataset pickle
DPATH = '/home/guest/Desktop/sleep_dataset.pkl'
GROUP = 'd' #'a','b','c', or 'd' 
NUM_SEG = 12  

dset = persist.pkl_load(DPATH)
#dist --> state x segment x animal
#states are always artifact, wake, NREM, REM
#ignoring artifact (hence [1:])
dist = get_distribution(dset, group=GROUP, num_seg=NUM_SEG)[1:]

#NOTE: We average segments later to give results for Light and Dark
#cycles. Light = ZT6 to ZT20 (14 hours). Rest is Dark.
