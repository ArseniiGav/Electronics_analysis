import numpy as np


def charge_comp(wfs_array, baselines, channel_number, left_b, right_b, baseline_samples, coef=1.5e-9):
    baselines_wf = []
    charges = []

    for evt in range(wfs_array.shape[0]):
        baselines_wf.append(np.mean(wfs_array[evt, channel_number, :baseline_samples]))
        signal = baselines[evt, channel_number] - wfs_array[evt, channel_number, left_b:right_b]
        charges.append(coef*np.sum(signal))
        
    baselines_gcu = baselines[:, channel_number]
    return np.array(charges), np.array(baselines_wf), baselines_gcu