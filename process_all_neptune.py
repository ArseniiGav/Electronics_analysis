import uproot
import os
import json
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import neptune.new as neptune
from vis_functions import *
from collect_data import *
from timestamp_calc import *
from rate_estim import *

params_file_path = sys.argv[1]
params_file = open(params_file_path)
PARAMS = json.load(params_file)

run = neptune.init_run(
    project="legnaro/LegnaroSetup",
    api_token=PARAMS['api_token'],
    tags=[
        PARAMS['path'],
        PARAMS['run_number'],
        PARAMS['source_name'],
        PARAMS['output_file'],
        PARAMS['output_file_wf'],
        "all"
    ],
    name="LNL",
    # with_id="LNL-19"
)

del PARAMS['api_token']
run['PARAMS'] = PARAMS

path = PARAMS['path']
output_file = PARAMS['output_file']
output_file_wf = PARAMS['output_file_wf']
run_number = PARAMS['run_number']
run_numbers = [run_number]
source_name = PARAMS['source_name']
Nsigmas = int(PARAMS['Nsigmas'])
baseline_samples = int(PARAMS['baseline_samples'])
evt_step = int(PARAMS['evt_step'])
left_b = int(PARAMS['left_b'])
right_b = int(PARAMS['right_b'])
rates_bins = int(PARAMS['rates_bins'])
left_shift = int(PARAMS['left_shift'])
graph_height = int(PARAMS['graph_height'])
graph_width = int(PARAMS['graph_width'])
source_names = [source_name]
runs_list = ["Run "+run_number+". "+source_name]

paths = [f'{path}run{i}/tier1/{output_file}' for i in [run_number]]
path_wfs = f'{path}run{run_number}/tier1_wf/{output_file_wf}'
file = uproot.open(f'{path}run{run_number}/tier1/{output_file}')
timestamps = np.array(file['eventTree']['timestamp'].array())
baseline_array = np.array(uproot.open(path_wfs)['eventTree']['baseline'].array())
charge_array = np.array(uproot.open(path_wfs)['eventTree']['charge'].array())
active_ch = np.array(file['eventTree']['active_ch'].array())
wfs_array = np.array(
    uproot.open(
        path_wfs, file_handler=uproot.MultithreadedFileSource, num_workers=10
    )['eventTree']['waveforms'].array()
)
shp = wfs_array.shape
print(f"Number of events: {shp[0]}, number of channels: {shp[1]}, N. of time points: {shp[2]}")

dfs = []
for i in range(shp[1]):
    df = pd.DataFrame(
        np.array([np.tile(np.arange(shp[2]), shp[0]), wfs_array[:, i, :].flatten()]).T,
        columns=['t', 'wfs']
    )
    dfs.append(df)
    
rates_array = []
fit_params_array = []
distrs_array = []

channel_number = np.random.randint(45)
rate_estim_outputs = np.array(
    [rate_estim(path, run_number, output_file, channel_number, Nmeds=10, left_shift=left_shift) for run_number in run_numbers],
    dtype=object
)

rates = rate_estim_outputs[:, 0]
fit_params = rate_estim_outputs[:, 1]
distrs = rate_estim_outputs[:, 2]

active_chs = join_by_run(paths, ['active_ch'])
plot_distrs_one_param(active_chs, 'active_ch', source_names,
                      save_plot=False, Nbins=np.linspace(0, shp[1], shp[1]+1), xaxis_title="Active channels",
                      neptune_run=True, run=run, run_plot_name="Active channels")

multiplicity = join_by_run(paths, ['multiplicity'])
plot_distrs_one_param(multiplicity, 'multiplicity', source_names,
                      save_plot=False, Nbins=np.linspace(0, shp[1], shp[1]+1), xaxis_title="Multiplicity", 
                      neptune_run=True, run=run, run_plot_name="Multiplicity")

plot_timestamps(timestamps, width=graph_width, height=graph_height, left_shift=left_shift,
            vertical_spacing=0.05, horizontal_spacing=0.05, evt_step=1000,
            neptune_run=True, run=run, run_plot_name="Timestamps")

rates_fit_plots(runs_list, distrs, fit_params, rates, channel_number,
                Nbins=rates_bins, line_width=0.1, neptune_run=True, run=run,
                run_plot_name=f"Rate for {PARAMS['run_number']} run")

wfs_2d_plot_by_channels(dfs, 't', 'wfs', plot_width=120,
                        plot_height=120, height=graph_height, width=graph_width,
                        neptune_run=True, run=run,
                        run_plot_name="wfs 2d hist by channel")

for evt_num in [2000, 5000, 10000]:
    plot_wf_diff_channels_same_evt(wfs_array, EvtNumber=evt_num, height=graph_height,
                                   width=graph_width, range_y_max=11800, 
                                   Nsigmas=Nsigmas, baseline_samples=baseline_samples,
                                   neptune_run=True, run=run,
                                   run_plot_name=f"Diff. channels, same evt. EvtNumber: {evt_num}")
for channel_num in [0, 14, 28, 42]:
    plot_wf_same_channel_diff_evts(wfs_array, ChannelNumber=channel_num, nrows=7, ncols=5,
                                   height=graph_height, width=graph_width, left_shift=left_shift, 
                                   range_y_max=11800, Nsigmas=Nsigmas,
                                   baseline_samples=baseline_samples,
                                   neptune_run=True, run=run,
                                   run_plot_name=f"Same channel, diff. evts. Channel {channel_num}")

plot_baselines_diffs(wfs_array, baseline_array, horizontal_spacing=0.05,
                     vertical_spacing=0.05, height=graph_height, width=graph_width, left_b=left_b,
                     right_b=right_b, baseline_samples=baseline_samples,
                     neptune_run=True, run=run, run_plot_name="Baselines' differences distributions")

plot_charges_hist(wfs_array, baseline_array, horizontal_spacing=0.05,
                  vertical_spacing=0.05, height=graph_height, width=graph_width,
                  left_b=left_b, right_b=right_b, baseline_samples=baseline_samples, neptune_run=True, run=run,
                  run_plot_name="Charge distributions (where the charge is calculated manually)")

plot_charges_scatter(wfs_array, charge_array, baseline_array,
                     horizontal_spacing=0.08, vertical_spacing=0.05, height=graph_height,
                     width=graph_width, left_b=left_b, right_b=right_b,
                     baseline_samples=baseline_samples,
                     evt_step=evt_step, neptune_run=True, run=run,
                     run_plot_name="Manually calculated charge (x axis) vs. GCU calculated charge (y axis)")

paths = [f'{path}run{i}/tier1/{output_file}' for i in run_numbers]
total_charges = join_by_run(paths, ['total_charge'])
plot_distrs_one_param(total_charges, 'total_charge', source_names, "Total charge",
                      bkg_subtract=False, Nbins=np.linspace(0, 1e-4, 500), bar_step_x_shift=0.475,
                      range_x=[0, 1e-4], line_width=0.0, opacity=0.7, return_values=False,
                      neptune_run=True, run=run, run_plot_name=f"Total charge for {PARAMS['run_number']} run")

for i in run_numbers:
    print(f"-_____________________run number {i}_____________________")
    timestamp_list = timestamp_calc(path, run_number=i, dir_name="tier1", left_shift=left_shift)
    print(timestamp_list)
    print(f"**************Number of channels: {len(timestamp_list)}")
    print("\n \n")

run.stop()

