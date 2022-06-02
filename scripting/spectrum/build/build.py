"""
AS build using annotations text file to reject.
"""
import time
from scripting.spectrum import metadata
from scripting.spectrum.io import persist
from scripting.spectrum import dataset
from scripting.utils import interpolate

BETWEEN = [' Start', ' Stop'] #start and stop labels in annotation text file
HOURS = [(0, 1), (4, 5), (8, 9), (12, 13), (16, 17), (20, 21)] #hours to use relative to start label
ANNOTATIONS = ['ignore'] #annotations to include. Ignore means include everthing except REJECTS
REJECTS = ['Artifact', 'water_drinking'] #annotations to exclude
DELIMITER = '\t' #delimiter for annotations text file
START = 0 #start row for reading metadata from CSV
NPERSEG = 10000 #nperseg for scipy.welch()

SAVE_PATH = './dataset.pkl'
META_PATH = '/home/guest/Desktop/group info/group_info_8week_post3week.csv'
KEY_COLUMN = 'Animal ID' #column in csv holding animal ID 
COLUMNS = ['Treatment'] #column in csv holding metadata info

#Build dataset
t0 = time.perf_counter() #initiate timer
dset = dataset.build(nperseg=NPERSEG, annotations=ANNOTATIONS, rejects=REJECTS, 
                     hours=HOURS, between=BETWEEN, delimiter=DELIMITER) #see dataset.py

delta_t = time.perf_counter() - t0
print('Dataset built in {} s.'.format(delta_t))

#Add metadata to dataset from csv
meta = metadata.from_csv(META_PATH, 0, KEY_COLUMN, *COLUMNS, delimiter='\t')
dset['metadata'] = meta
#Interpolate data around 60Hz to account for attenuation by notch filtering
interpolate(dset) #linear interpolation

#save the dataset
try:
    persist.pkl_save(SAVE_PATH, dset)
    print('Dataset successfully saved to {}'.format(SAVE_PATH))
except Exception as e:
    raise(e)

