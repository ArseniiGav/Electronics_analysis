import numpy as np
import uproot
import os


def shapes_cumsum_calc(path, run_number):
    full_path = f'{path}run{run_number}/tier1_wf'
    shapes = []
    for file_name in os.listdir(full_path):   
        path_wfs = f'{full_path}/{file_name}'
        timestamp_array = np.array(uproot.open(path_wfs)['eventTree']['timestamp'].array())
        shapes.append(timestamp_array.shape[0])
    shape_cumsum = np.cumsum(shapes)
    
    return shape_cumsum


def find_file_and_ent_number(EvtNumber, shape_cumsum):
    if EvtNumber < shape_cumsum[0]:
        FileNumber = ''
    else:
        for i in range(len(shape_cumsum)-1):            
            if EvtNumber >= shape_cumsum[i] and EvtNumber < shape_cumsum[i+1]:
                FileNumber = f"_{i+1}"
                EvtNumber -= shape_cumsum[i]
                break 
    
    return EvtNumber, FileNumber