import uproot
import numpy as np
from scipy.stats import expon

def rate_estim(path, run_number, file_name, channel_number, Nmeds=20, left_shift=50):
    file = uproot.open(f'{path}run{run_number}/tier1/{file_name}')
    timestamp = np.array(file['eventTree']['timestamp'].array())
    
    diffs = np.diff(timestamp[left_shift:, channel_number])
    upper_cut = Nmeds*np.median(diffs[diffs>0])
    mask = np.logical_and(diffs < upper_cut, diffs > 0)
    diffs_masked = diffs[mask]
    
    try:
        res = expon.fit(diffs_masked * 8e-9)
        rate = 1 / res[1]
    except Exception as e:
        print(e)
        res = None
        rate = np.nan
        
    return rate, res, diffs_masked