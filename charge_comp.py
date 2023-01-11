import numpy as np


def charge_comp(wfs_array, baselines, channel_number, left_b, right_b, evt_range, coef=1.5e-9):
    baselines_wf = []
    charges = []

    for evt in range(evt_range[0], evt_range[1]):
        baselines_wf.append(np.mean(wfs_array[evt, channel_number, :15]))
        signal = baselines[evt, channel_number] - wfs_array[evt, channel_number, left_b:right_b]
        charges.append(coef*np.sum(signal))
        
    baselines_gcu = baselines[evt_range[0]:evt_range[1], channel_number]
    return np.array(charges), np.array(baselines_wf), baselines_gcu