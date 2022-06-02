from scripting.spectrum.dataset import extract
from scripting.spectrum import metadata
from scripting.spectrum.metrics import band_powers
from scripting.spectrum.io import persist

DPATH = '/home/guest/Desktop/dataset_pickles/new/8wp3/dataset.pkl'
GROUP = 'b' #or 'b','c','d'
CHS = [0, 1, 2]
BANDS = [(1,4),(4,8),(8,13),(13,18),(18,25),(25,50),(50,100)]
ANNOTATION = 'ignore'

dset = persist.pkl_load(DPATH)
freqs = dset['frequencies']
names = metadata.find(dset['metadata'], GROUP)
#arr -> freqs X names
abs_psd = extract(dset, annotations=[ANNOTATION], names=names,
                  channels=CHS)

#normalize psds to power in 1-100Hz
normalizer = band_powers(abs_psd, freqs, bands=[(1, 100)], axis=1)
normalized_psd = abs_psd / normalizer
#see scripting.metrics for different options.
#this is what was used for the manuscript.
absolute_powers = band_powers(abs_psd, freqs, bands=BANDS, axis=1)
relative_powers = band_powers(abs_psd, freqs, bands=BANDS, relative=True, axis=1)
band_ratios = band_powers(abs_psd, freqs, bands=[(8, 25)], relative=True,
                          cutoffs=[50,100], axis=1)
