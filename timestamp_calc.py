import uproot
import numpy as np

def timestamp_calc(run_number, dir_name, left_shift=5000,
                   path = "/data/48PMT/calib/20221207/"):
    
    file = uproot.open(f'{path}run{run_number}/{dir_name}/output_dt2.root')
    timestamp = np.array(file['eventTree']['timestamp'].array())
    diffs = np.array(
        [8e-9*(timestamp[-1, i] - timestamp[left_shift, i]) for i in range(timestamp.shape[1])]
    )

    for i in range(diffs.shape[0]):
        if diffs[i] > 1000:
            print(f'**************Channel with incorrect value: {i}')
            print(f'**************Value: {8e-9*timestamp[5000, i]}')

    return diffs

