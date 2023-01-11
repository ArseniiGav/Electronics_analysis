import uproot
import numpy as np


def collect_run_data(path, keys='all'):
    run_data = dict()
    with uproot.open(path) as file:
        if keys == 'all':
            keys = file['eventTree'].keys()
        for param in keys:
            if param not in ['waveforms']:
                run_data[param] = np.array(file['eventTree'][param].array())

    return run_data


def join_by_run(paths, keys='all'):
    data = []
    for path in paths:
        data.append(collect_run_data(path, keys))
    return data