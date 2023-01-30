import numpy as np
import copy
import neptune.new as neptune
from scipy.stats import expon
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.express as px
from charge_comp import charge_comp
pio.templates.default = 'plotly_white'    

xaxis = dict(
    showline=True,
    ticks='outside',
    mirror=True,
    linecolor='black',
    showgrid=True,
    gridcolor='grey',
    gridwidth=0.25,
)

yaxis = dict(
    showline=True,
    ticks='outside',
    mirror=True,
    linecolor='black',
    showgrid=True,
    gridcolor='grey',
    gridwidth=0.25,
    zeroline=True,
    zerolinecolor='black',
    zerolinewidth=0.25
)

def add_buttons(args1, args2, label1, label2, x_shift, method, y_shift=1.15):
    buttons_dict = dict( 
        type = "buttons",
        direction = "left",
        buttons=list([
            dict(
                args=[args1],
                label=label1,
                method=method
            ),
            dict(
                args=[args2],
                label=label2,
                method=method
            )
        ]),
        showactive=True,
        x=x_shift,
        xanchor="left",
        y=y_shift,
        yanchor="top",
    )
    return buttons_dict

def mean_absolute_percentage_error(y_true, y_fit):
    return np.mean(np.abs((y_true - y_fit)/y_true))*100


def plot_baselines_diffs(wfs_array, baseline_array,
                         nrows=8, ncols=6,
                         Nbins=100, vertical_spacing=0.1,
                         horizontal_spacing=0.1,
                         height=1000, width=1200, 
                         left_b=70, right_b=250,
                         baseline_samples=60,
                         neptune_run=False, **kwargs):
    
    fig = make_subplots(rows=nrows, cols=ncols,
                        vertical_spacing=vertical_spacing,
                        horizontal_spacing=horizontal_spacing)

    for i in range(nrows*ncols):
        try:
            charges, baselines_wf, baselines_gcu = charge_comp(
                wfs_array, baseline_array, i, left_b, right_b, baseline_samples)

            counts, bins = np.histogram(baselines_wf-baselines_gcu, bins=Nbins)
            bins = 0.5 * (bins[:-1] + bins[1:])

            if np.sum(baselines_wf-baselines_gcu) == 0:
                bar_width = 1
            else:
                bar_width = None

            fig.add_trace(
                go.Bar(
                    x=bins,
                    y=counts,
                    name=f"Channel: {i}",
                    width=bar_width,
                    marker=dict(
                        line=dict(width=0.0)
                    )
                ), row=int(i/ncols)+1, col=i%ncols+1
            )

            fig.update_xaxes(title="bsln_wf - bsln_gcu",
                             row=int(i/ncols)+1, col=i%ncols+1)
        except Exception as e:
            pass
        
    axis_params = {}
    for i in range(1, nrows*ncols+1):
        try:
            axis_params['xaxis{}'.format(i)] = xaxis
            axis_params['yaxis{}'.format(i)] = yaxis
        except:
            pass

    fig.update_layout(
        height=height,
        width=width,
        barmode='overlay',
        bargap=0.0,
        **axis_params,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1
        )
    )

    fig.show()
    if neptune_run:
        run = kwargs['run']
        run_plot_name = kwargs['run_plot_name']
        run[run_plot_name].upload(neptune.types.File.as_html(fig))
        

def plot_charges_hist(wfs_array, baseline_array,
                      nrows=8, ncols=6, Nbins=100,
                      vertical_spacing=0.1,
                      horizontal_spacing=0.1,
                      height=1000, width=1200, 
                      left_b=90, right_b=220,
                      baseline_samples=60,
                      neptune_run=False, **kwargs):

    fig = make_subplots(rows=nrows, cols=ncols,
                        vertical_spacing=vertical_spacing,
                        horizontal_spacing=horizontal_spacing)
    
    for i in range(nrows*ncols):
        try:
            charges, baselines_wf, baselines_gcu = charge_comp(
                wfs_array, baseline_array, i, left_b, right_b, baseline_samples)

            counts, bins = np.histogram(charges, bins=Nbins)
            bins = 0.5 * (bins[:-1] + bins[1:])

            fig.add_trace(
                go.Bar(
                    x=bins,
                    y=counts,
                    name=f"Channel: {i}",
                    marker=dict(
                        line=dict(width=0),
                    )
                ), row=int(i/ncols)+1, col=i%ncols+1
            )

            fig.add_trace(
                go.Scatter(
                    x=bins,
                    y=counts,
                    showlegend=True,
                    visible=False,
                    name=f"Channel: {i}",
                    line=dict(shape='hvh'),
                    marker=dict(
                        line=dict(
                            color='black',
                        )
                    )
                ), row=int(i/ncols)+1, col=i%ncols+1
            )
                            
            fig.update_xaxes(title="Charge", row=int(i/ncols)+1, col=i%ncols+1)
        except:
            pass
        
    axis_params = {}
    yaxis_linear = {"yaxis.type": "linear"}
    yaxis_log = {"yaxis.type": "log"}
    for i in range(1, nrows*ncols+1):
        try:
            axis_params[f'xaxis{i}'] = xaxis
            axis_params[f'yaxis{i}'] = yaxis
            if i > 1:
                yaxis_linear[f"yaxis{i}.type"] = "linear"
                yaxis_log[f"yaxis{i}.type"] = "log"
        except:
            pass
    
    updatemenus = []
    updatemenus.append(add_buttons(
            yaxis_linear, yaxis_log, "Linear scale", "Log scale", 0.0, "relayout", 1.18
        )
    )
    
    visibles = [True, False]*nrows*ncols
    updatemenus.append(
        add_buttons(
            {"visible": visibles},
            {"visible": visibles[::-1]},
            "Bar", "Step", 0.75, "update", 1.18
        )
    )
    
    fig.update_layout(
        height=height,
        width=width,
        barmode='overlay',
        bargap=0.0,
        **axis_params,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1
        ),
        updatemenus=updatemenus
    )

    fig.show()
    if neptune_run:
        run = kwargs['run']
        run_plot_name = kwargs['run_plot_name']
        run[run_plot_name].upload(neptune.types.File.as_html(fig))


def plot_charges_scatter(wfs_array, charge_array,
                         baseline_array,
                         nrows=8, ncols=6,
                         vertical_spacing=0.1,
                         horizontal_spacing=0.1,
                         height=1000, width=1200, 
                         left_b=90, right_b=220,
                         baseline_samples=60, evt_step=1,
                         neptune_run=False, **kwargs): 
    
    fig = make_subplots(rows=nrows, cols=ncols,
                        vertical_spacing=vertical_spacing,
                        horizontal_spacing=horizontal_spacing)
    
    for i in range(nrows*ncols):
        try:
            charges, baselines_wf, baselines_gcu = charge_comp(
                wfs_array, baseline_array, i, left_b, right_b, baseline_samples)

            fig.add_trace(
                go.Scattergl(
                    x = charges[::evt_step],
                    y = charge_array[::evt_step, i],
                    name=f"Channel: {i}",
                    mode='markers',
                ), row=int(i/ncols)+1, col=i%ncols+1
            )

            fig.update_xaxes(title="Manuanlly calc. Q", row=int(i/ncols)+1, col=i%ncols+1)
            fig.update_yaxes(title="GCU calc. Q", row=int(i/ncols)+1, col=i%ncols+1)
        except Exception as e:
            print(e)
            pass

    axis_params = {}
    for i in range(1, nrows*ncols+1):
        try:
            axis_params['xaxis{}'.format(i)] = xaxis
            axis_params['yaxis{}'.format(i)] = yaxis
        except:
            pass

    fig.update_layout(
        height=height,
        width=width,
        **axis_params,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1
        )
    )

    fig.show()
    if neptune_run:
        run = kwargs['run']
        run_plot_name = kwargs['run_plot_name']
        run[run_plot_name].upload(neptune.types.File.as_html(fig))

    
def wfs_2d_plot_by_channels(data, xlabel, ylabel, neptune_run=False,
                            plot_width=200, plot_height=200,
                            col_wrap=6, height=500, width=950, **kwargs):
    
    aggs = []
    for i in range(len(data)):
        agg, xedges, yedges = np.histogram2d(x=data[i][xlabel], y=data[i][ylabel],
                                   bins=(plot_width, plot_height))
        
        x_centers = (xedges[1:] + xedges[:-1]) / 2
        y_centers = (yedges[1:] + yedges[:-1]) / 2
        agg = agg.T
        zero_mask = agg == 0
        agg = np.log10(agg, where=np.logical_not(zero_mask))
        agg[zero_mask] = np.nan
        aggs.append(agg)
    aggs = np.array(aggs)
        
    fig = px.imshow(
            aggs,
            x=x_centers,
            y=y_centers,
            origin='lower',
            labels={
                'color':'Log10(count)',
                'facet_col':'Channel',
                'x': 't, ns',
                'y': 'V, ADC counts',
            },
            color_continuous_scale='inferno',
            height=height,
            width=width,
            facet_col=0,
            facet_col_wrap=col_wrap,
            aspect='auto'
        )
    
    axis_params = {}
    for i in range(1, len(data)+1):
        axis_params['xaxis{}'.format(i)] = xaxis
        axis_params['yaxis{}'.format(i)] = yaxis

    fig.update_layout(
        coloraxis_colorbar=dict(
            title='Log10',
            tickprefix='10^'
        ),
        showlegend=True,
        **axis_params,
        font=dict(
            family="Times New Roman",
            size=20,
            color='black'
        ),
    )

    fig.show()
    if neptune_run:
        run = kwargs['run']
        run_plot_name = kwargs['run_plot_name']
        run[run_plot_name].upload(neptune.types.File.as_html(fig))

def plot_wf_diff_channels_same_evt(
    wfs_array, EvtNumber, range_y_max=11800, range_y_min_step=200, neptune_run=False,
    nrows=8, ncols=6, height=800, width=1400, Nsigmas=25, baseline_samples=60, **kwargs):
    
    fig = make_subplots(rows=nrows, cols=ncols)
    
    minis = []
    maxis = []
    for i in range(nrows*ncols):
        try:
            fig.add_trace(
                go.Scattergl(
                    x = np.arange(wfs_array.shape[2]),
                    y = wfs_array[EvtNumber, i, :].flatten(),
                    name=f"Channel {i}"
                ), row=int(i/ncols)+1, col=i%ncols+1
            )
            
            baseline_mean = wfs_array[EvtNumber, i, :baseline_samples].mean()
            baseline_std = wfs_array[EvtNumber, i, :baseline_samples].std()
            
            fig.add_hline(
                y=baseline_mean,
                line=dict(dash='dash'),
                row=int(i/ncols)+1,
                col=i%ncols+1
            )

            fig.add_hrect(
                y0=baseline_mean-Nsigmas*baseline_std,
                y1=baseline_mean+Nsigmas*baseline_std,
                fillcolor="darkred",
                line_width=1,
                opacity=0.25,
                row=int(i/ncols)+1,
                col=i%ncols+1
            )
            minis.append(min(baseline_mean-Nsigmas*baseline_std,
                             min(wfs_array[EvtNumber, i, :].flatten())))
            maxis.append(max(baseline_mean+Nsigmas*baseline_std,
                             max(wfs_array[EvtNumber, i, :].flatten())))
        except:
            pass
        
    axis_params = {}
    for i in range(1, nrows*ncols+1):
        try:
            axis_params['xaxis{}'.format(i)] = xaxis
            axis_params['yaxis{}'.format(i)] = yaxis
        except:
            pass
        
    buttons = []
    for range_y_min in range(0, range_y_max, range_y_min_step):
        yaxis_range = {"yaxis.range": [range_y_min, range_y_max]}
        for i in range(2, nrows*ncols+1):
            yaxis_range[f"yaxis{i}.range"] = [range_y_min, range_y_max]
        buttons.append(
            dict(
                args=[yaxis_range],
                label=f"y min limit: {range_y_min}",
                method="relayout"
            )
        )
        
    default_button = []
    default_yaxis_range = {"yaxis.range": [minis[0], maxis[0]]}
    for i in range(2, nrows*ncols+1):
        default_yaxis_range[f"yaxis{i}.range"] = [minis[i-1], maxis[i-1]]
    default_button.append(
        dict(
            args=[default_yaxis_range],
            label="Default",
            method="relayout"
        )
    )
        
    updatemenus = []
    updatemenus.append(
        dict(
            buttons=buttons,
            x=0.0,
            xanchor="left",
            y=1.15,
            yanchor="top",
        )
    )
    updatemenus.append(
        dict(
            type = "buttons",
            buttons=default_button,
            x=0.12,
            xanchor="left",
            y=1.15,
            yanchor="top",
            showactive=True
        )
    )


    fig.update_layout(
        title=dict(
            text=f"Event number: {EvtNumber}",
            xanchor='center',
            yanchor='top',
            x=0.5,
            y=0.99
        ),
        **axis_params,
        height=height,
        width=width,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1
        ),
        updatemenus=updatemenus
    )

    fig.show()
    if neptune_run:
        run = kwargs['run']
        run_plot_name = kwargs['run_plot_name']
        run[run_plot_name].upload(neptune.types.File.as_html(fig))


def plot_wf_same_channel_diff_evts(
    wfs_array, ChannelNumber, range_y_max=11800, range_y_min_step=200,
    nrows=10, ncols=6, height=1600, width=1400, neptune_run=False,
    left_shift=1000, Nsigmas=25, baseline_samples=60, **kwargs):

    evtIds = np.random.randint(left_shift, wfs_array.shape[0], size=nrows*ncols)
    fig = make_subplots(rows=nrows, cols=ncols)

    minis = []
    maxis = []
    for i in range(nrows*ncols):
        fig.add_trace(
            go.Scattergl(
                x = np.arange(wfs_array.shape[2]),
                y = wfs_array[evtIds[i], ChannelNumber, :].flatten(),
                name=f"EvtN: {evtIds[i]}",
            ), row=int(i/ncols)+1, col=i%ncols+1
        )

        baseline_mean = wfs_array[evtIds[i], ChannelNumber, :baseline_samples].mean()
        baseline_std = wfs_array[evtIds[i], ChannelNumber, :baseline_samples].std()

        fig.add_hline(
            y=baseline_mean,
            line=dict(dash='dash'),
            row=int(i/ncols)+1,
            col=i%ncols+1
        )

        fig.add_hrect(
            y0=baseline_mean-Nsigmas*baseline_std,
            y1=baseline_mean+Nsigmas*baseline_std,
            fillcolor="darkred",
            line_width=1,
            opacity=0.25,
            row=int(i/ncols)+1,
            col=i%ncols+1
        )
        
        minis.append(min(baseline_mean-Nsigmas*baseline_std,
                         min(wfs_array[evtIds[i], ChannelNumber, :].flatten())))
        maxis.append(max(baseline_mean+Nsigmas*baseline_std,
                         max(wfs_array[evtIds[i], ChannelNumber, :].flatten())))

    axis_params = {}
    yaxis_linear = {"yaxis.range": "linear"}
    for i in range(1, nrows*ncols+1):
        try:
            axis_params[f'xaxis{i}'] = xaxis
            axis_params[f'yaxis{i}'] = yaxis
            if i > 1:
                yaxis_linear[f"yaxis{i}.type"] = "linear"
        except:
            pass
            
    buttons = []
    for range_y_min in range(0, range_y_max, range_y_min_step):
        yaxis_range = {"yaxis.range": [range_y_min, range_y_max]}
        for i in range(2, nrows*ncols+1):
            yaxis_range[f"yaxis{i}.range"] = [range_y_min, range_y_max]
        buttons.append(
            dict(
                args=[yaxis_range],
                label=f"y min limit: {range_y_min}",
                method="relayout"
            )
        )
        
    default_button = []
    default_yaxis_range = {"yaxis.range": [minis[0], maxis[0]]}
    for i in range(2, nrows*ncols+1):
        default_yaxis_range[f"yaxis{i}.range"] = [minis[i-1], maxis[i-1]]
    default_button.append(
        dict(
            args=[default_yaxis_range],
            label="Default",
            method="relayout"
        )
    )
        
    updatemenus = []
    updatemenus.append(
        dict(
            buttons=buttons,
            x=0.0,
            xanchor="left",
            y=1.15,
            yanchor="top",
        )
    )
    updatemenus.append(
        dict(
            type="buttons",
            buttons=default_button,
            x=0.12,
            xanchor="left",
            y=1.15,
            yanchor="top",
            showactive=True
        )
    )

    fig.update_layout(
        title=dict(
            text=f"Channel number: {ChannelNumber}",
            xanchor='center',
            yanchor='top',
            x=0.5,
            y=0.99
        ),
        **axis_params,
        height=height,
        width=width,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1
        ),
        updatemenus=updatemenus
    )

    fig.show()
    if neptune_run:
        run = kwargs['run']
        run_plot_name = kwargs['run_plot_name']
        run[run_plot_name].upload(neptune.types.File.as_html(fig))


def rates_fit_plots(
    run_numbers, runs_list, distrs_array, fit_params_array, rates_array, neptune_run=False,
    Nbins=100, line_width=0.0, height=1600, width=1400, nrows=8, ncols=6,
    vertical_spacing=0.05, horizontal_spacing=0.05, **kwargs
):
    fig = make_subplots(rows=nrows, cols=ncols,
                        vertical_spacing=vertical_spacing,
                        horizontal_spacing=horizontal_spacing)

    size = len(distrs_array)
    for i in range(len(runs_list)):
        try:
            for k in range(size):
                if len(distrs_array[k][i]) == 0:
                    x = [0]
                    y = [0]
                    bins = [0]
                    counts = [0]
                    name = f"Channel {k}. τ: {np.round(rates_array[k][i], 2)}"
                else:
                    x = np.linspace(0, distrs_array[k][i].max()*8e-9, 10000)
                    y = expon.pdf(x, *fit_params_array[k][i])
                    name = f"Channel {k}. τ: {np.round(rates_array[k][i], 2)}"

                    counts, bins = np.histogram(distrs_array[k][i]*8e-9, density=True, bins=Nbins)
                    bins = 0.5 * (bins[:-1] + bins[1:]) 

                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        name=name,
                        mode='lines',
                        visible=(i==0)
                    ), row=int(k/ncols)+1, col=k%ncols+1
                )

                fig.add_trace(
                    go.Bar(
                        x=bins,
                        y=counts,
                        showlegend=False,
                        marker=dict(
                            color='darkred',
                            line=dict(
                                color='black',
                                width=line_width,
                            )
                        ),
                        visible=(i==0)
                    ), row=int(k/ncols)+1, col=k%ncols+1
                )

                fig.update_xaxes(title="Timestamp diffs",
                                 row=int(i/ncols)+1, col=i%ncols+1)

                
        except Exception as e:
            print(e)
            pass

    axis_params = {}
    yaxis_linear = {"yaxis.type": "linear"}
    yaxis_log = {"yaxis.type": "log"}
    for i in range(1, size+1):
        try:
            axis_params[f'xaxis{i}'] = xaxis
            axis_params[f'yaxis{i}'] = yaxis
            if i > 1:
                yaxis_linear[f"yaxis{i}.type"] = "linear"
                yaxis_log[f"yaxis{i}.type"] = "log"
        except:
            pass
        
    updatemenus = []
    updatemenus.append(add_buttons(
            yaxis_linear, yaxis_log, "Linear scale", "Log scale", 0.4, "relayout", y_shift=1.18
        )
    )
    
    buttons = []
    for N in range(len(runs_list)): 
        buttons.append(
            dict(
                 args=['visible', [False]*2*N*size + [True]*2*size + [False]*2*size*(len(runs_list)-1-N)],
                     label=f'{runs_list[N].capitalize()}',
                 method='restyle'
            )
        )
        
    updatemenus.append(
            dict(
                x=0.0,
                y=1.18,
                xanchor="left",
                yanchor="top",
                buttons=buttons
            )
    )
        
    fig.update_layout(
        showlegend=True,
        **axis_params,
        bargap=0.0,
        height=height,
        width=width,
        updatemenus=updatemenus,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1
        )
    )

    fig.show()
    if neptune_run:
        run = kwargs['run']
        run_plot_name = kwargs['run_plot_name']
        run[run_plot_name].upload(neptune.types.File.as_html(fig))
    
def plot_distrs_one_param(
    params_list, param, names, xaxis_title, bar_step_x_shift=0.75,
    colors_flag=False, scale_factor=1, bkg_n=-1, Nbins=200,
    bkg_subtract=False, line_width=0.1, save_plot=False,
    opacity=0.75, height=500, width=950, neptune_run=False,
    left_shift=0, return_values=False, **kwargs
):
    
    if colors_flag:
        colors = kwargs['colors']
    else:
        colors=[None]*len(params_list)
    
    
    fig = go.Figure()
    
    if not bkg_subtract:
        counts_list = []
        for i in range(len(params_list)):
            
            counts, bins = np.histogram(params_list[i][param][left_shift:],
                                        bins=Nbins)
            bins = 0.5 * (bins[:-1] + bins[1:])
            
            counts_list.append(counts)
            
            fig.add_trace(
                go.Bar(
                    x=bins,
                    y=counts,
                    opacity=opacity,
                    name=names[i],
                    text=f"Total entries: {sum(counts)}",
                    textposition="none",
                    showlegend=True,
                    marker=dict(
                        color=colors[i],
                        line=dict(
                            color='black',
                            width=line_width,
                        )
                    )
                )
            )
                
            fig.add_trace(
                go.Scatter(
                    x=bins,
                    y=counts,
                    opacity=opacity,
                    name=names[i],
                    showlegend=True,
                    visible=False,
                    line=dict(shape='hvh'),
                    marker=dict(
                        color=colors[i],
                        line=dict(
                            color='black',
                            width=line_width,
                        )
                    )
                )
            )
            
        visibles = [True, False] * len(params_list)
    else:
        counts_bkg, bins = np.histogram(params_list[bkg_n][param][left_shift:],
                                        bins=Nbins)
        bins = 0.5 * (bins[:-1] + bins[1:])

        for i in range(len(params_list)):
            if i != bkg_n:
                counts, bins = np.histogram(params_list[i][param][left_shift:],
                                            bins=Nbins)
                bins = 0.5 * (bins[:-1] + bins[1:])
                
                fig.add_trace(
                    go.Bar(
                        x=bins,
                        y=counts - scale_factor*counts_bkg,
                        opacity=opacity,
                        name=names[i],
                        text=f"Total entries: {sum(counts)}",
                        textposition="none",
                        showlegend=True,
                        marker=dict(
                            color=colors[i],
                            line=dict(
                                color='black',
                                width=line_width,
                            )
                        )
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=bins,
                        y=counts - scale_factor*counts_bkg,
                        opacity=opacity,
                        name=names[i],
                        showlegend=True,
                        visible=False,
                        line=dict(
                            shape='hvh'
                        ),
                        marker=dict(
                            color=colors[i],
                            line=dict(
                                color='black',
                                width=line_width,
                            )
                        )
                    )
                )
                
            visibles = [True, False] * (len(params_list)-1)

                    
    updatemenus = []
    updatemenus.append(
        add_buttons(
            {"yaxis.type": "linear"},
            {"yaxis.type": "log"},
            "Linear scale", "Log scale", 0.0, "relayout"
        )
    )
    updatemenus.append(
        add_buttons(
            {"visible": visibles},
            {"visible": visibles[::-1]},
            "Bar", "Step", bar_step_x_shift, "update"
        )
    )

    fig.update_layout(
        xaxis_title=xaxis_title,
        xaxis=xaxis,
        yaxis=yaxis,
        font=dict(
            family="Times New Roman",
            size=20,
            color="black"
        ),
        height=height,
        width=width,
        bargap=0.0,
        barmode='overlay',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        updatemenus=updatemenus
    )

    fig.show()
    if neptune_run:
        run = kwargs['run']
        run_plot_name = kwargs['run_plot_name']
        run[run_plot_name].upload(neptune.types.File.as_html(fig))
    if save_plot:
        file_name = kwargs['file_name']
        pio.write_image(fig, f'plots/calib_12_2022/{file_name}.pdf', width=width, height=height)
    if return_values:
        return bins, scale_factor*counts_list[0], counts_list[1]
    
    
def plot_charges_by_channels_src_and_bkg(charges, timestamps, source_names, Nbins,
                                     nrows=8, ncols=6, left_shift=1000,
                                     bkg_subtract=False, line_width=0.0,
                                     opacity=0.75, height=1000, width=1600,
                                     neptune_run=False, **kwargs):

    bkg_times = timestamps[0]
    cs_times = timestamps[1]
    max_dim = min(bkg_times.shape[0], cs_times.shape[0])
    scale_factors = cs_times[:max_dim] / bkg_times[:max_dim]

    fig = make_subplots(rows=nrows, cols=ncols)

    for i in range(nrows*ncols):
        try:
            if not bkg_subtract:
                counts_bkg, bins = np.histogram(charges[0]['charge'][left_shift:, i],
                                            density=True, bins=Nbins)
                counts_src, bins = np.histogram(charges[1]['charge'][left_shift:, i],
                                            density=True, bins=Nbins)
                bins = 0.5 * (bins[:-1] + bins[1:]) 
                
                fig.add_trace(
                    go.Bar(
                        x=bins,
                        y=scale_factors[i]*counts_bkg,
                        opacity=opacity,
                        showlegend=False,
                        marker=dict(
                            color='grey',
                            line=dict(
                                width=line_width,
                            )
                        ),
                        name=f"Channel: {i}",
                        text=f"Channel: {i}. Bkg",
                        textposition="none",
                    ), row=int(i/ncols)+1, col=i%ncols+1
                )

                fig.add_trace(
                    go.Bar(
                        x=bins,
                        y=counts_src,
                        opacity=opacity,
                        marker=dict(
                            color='darkred',
                            line=dict(
                                width=line_width,
                            )
                        ),
                        name=f"Channel: {i}",
                        text=f"Channel: {i}. Src",
                        textposition="none",
                        showlegend=True,
                    ), row=int(i/ncols)+1, col=i%ncols+1
                )

                fig.add_trace(
                    go.Scatter(
                        x=bins,
                        y=scale_factors[i]*counts_bkg,
                        opacity=opacity,
                        showlegend=False,
                        visible=False,
                        line=dict(shape='hvh'),
                        marker=dict(
                            color='grey',
                            line=dict(
                                width=line_width,
                            )
                        ),
                        name=f"Channel: {i}",
                        text=f"Channel: {i}. Bkg",
                    ), row=int(i/ncols)+1, col=i%ncols+1
                )

                fig.add_trace(
                    go.Scatter(
                        x=bins,
                        y=counts_src,
                        opacity=opacity,
                        visible=False,
                        line=dict(shape='hvh'),
                        marker=dict(
                            color='darkred',
                            line=dict(
                                width=line_width,
                            )
                        ),
                        name=f"Channel: {i}",
                        text=f"Channel: {i}. Src",
                        showlegend=True,
                    ), row=int(i/ncols)+1, col=i%ncols+1
                )
                visibles = [True, True, False, False] * nrows*ncols
            else:
                counts_bkg, bins = np.histogram(charges[0]['charge'][left_shift:, i],
                                            density=True, bins=Nbins)
                counts_src, bins = np.histogram(charges[1]['charge'][left_shift:, i],
                                            density=True, bins=Nbins)
                bins = 0.5 * (bins[:-1] + bins[1:]) 
                
                fig.add_trace(
                    go.Bar(
                        x=bins,
                        y=counts_src - scale_factors[i]*counts_bkg,
                        opacity=opacity,
                        marker=dict(
                            color='darkred',
                            line=dict(
                                width=line_width,
                            )
                        ),
                        text=f"Channel: {i}. Src, bkg subtracted",
                        textposition="none",
                        name=f"Channel: {i}",
                        showlegend=True,
                    ), row=int(i/ncols)+1, col=i%ncols+1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=bins,
                        y=counts_src - scale_factors[i]*counts_bkg,
                        opacity=opacity,
                        visible=False,
                        line=dict(
                            shape='hvh'
                        ),
                        marker=dict(
                            color='darkred',
                            line=dict(
                                width=line_width,
                            )
                        ),
                        text=f"Channel: {i}. Src, bkg subtracted",
                        name=f"Channel: {i}",
                        showlegend=True,
                    ), row=int(i/ncols)+1, col=i%ncols+1
                )
                
                visibles = [True, False] * nrows*ncols
        except:
            pass

    axis_params = {}
    yaxis_linear = {"yaxis.type": "linear"}
    yaxis_log = {"yaxis.type": "log"}
    for i in range(1, nrows*ncols+1):
        try:
            axis_params[f'xaxis{i}'] = xaxis
            axis_params[f'yaxis{i}'] = yaxis
            if i > 1:
                yaxis_linear[f"yaxis{i}.type"] = "linear"
                yaxis_log[f"yaxis{i}.type"] = "log"
        except:
            pass
        
    updatemenus = []
    updatemenus.append(add_buttons(
            yaxis_linear, yaxis_log, "Linear scale", "Log scale", 0.0, "relayout", 1.18
        )
    )
    
    updatemenus.append(
        add_buttons(
            {"visible": visibles},
            {"visible": visibles[::-1]},
            "Bar", "Step", 0.75, "update", 1.18
        )
    )

    
    fig.update_layout(
        height=height,
        width=width,
        **axis_params,
        barmode='overlay',
        bargap=0.0,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1
        ),
        updatemenus=updatemenus
    )

    fig.show()
    if neptune_run:
        run = kwargs['run']
        run_plot_name = kwargs['run_plot_name']
        run[run_plot_name].upload(neptune.types.File.as_html(fig))


def plot_timestamps(timestamps, nrows=8, ncols=6,
                    vertical_spacing=0.1, horizontal_spacing=0.1,
                    height=1000, width=1200, left_shift=1000, neptune_run=False,
                    outlier_value=1e16, evt_step=False, **kwargs): 
    
    fig = make_subplots(rows=nrows, cols=ncols,
                        vertical_spacing=vertical_spacing,
                        horizontal_spacing=horizontal_spacing)
    
    if evt_step:
        if len(timestamps.shape) == 1:
            timestamps = copy.deepcopy(timestamps)
            for i in range(len(timestamps)):
                timestamps[i] = timestamps[i][::evt_step]
        else:
            timestamps = timestamps[::evt_step, :]
            
        left_shift /= evt_step
        left_shift = int(left_shift)
        x_title = f"Evt. number / {evt_step}"
    else:
        x_title = "Evt. number"
        
    for i in range(nrows*ncols):
        try:
            if len(timestamps.shape) == 1:
                shifted_timestamp = timestamps[i][left_shift:]
                x = np.arange(len(timestamps[i]))
            else:
                shifted_timestamp = timestamps[left_shift:, i]
                x = np.arange(len(timestamps[:, i]))
            shifted_x = x[left_shift:]
            outliers_mask = (shifted_timestamp < outlier_value)
            
            fig.add_trace(
                go.Scattergl(
                    x=shifted_x,
                    y=shifted_timestamp,
                    name=f"Channel: {i}",
                    mode='lines',
                ), row=int(i/ncols)+1, col=i%ncols+1
            )

            fig.add_trace(
                go.Scattergl(
                    x=shifted_x[outliers_mask],
                    y=shifted_timestamp[outliers_mask],
                    name=f"Channel: {i}",
                    mode='lines',
                    visible=False,
                ), row=int(i/ncols)+1, col=i%ncols+1
            )

            fig.update_xaxes(title=x_title, row=int(i/ncols)+1, col=i%ncols+1)
            fig.update_yaxes(title="Timestamp", row=int(i/ncols)+1, col=i%ncols+1)
        except Exception as e:
            print(e)
            pass
        
    axis_params = {}
    yaxis_linear = {"yaxis.type": "linear"}
    yaxis_log = {"yaxis.type": "log"}
    for i in range(1, nrows*ncols+1):
        try:
            axis_params[f'xaxis{i}'] = xaxis
            axis_params[f'yaxis{i}'] = yaxis
            if i > 1:
                yaxis_linear[f"yaxis{i}.type"] = "linear"
                yaxis_log[f"yaxis{i}.type"] = "log"
        except:
            pass
        
    updatemenus = []
    updatemenus.append(add_buttons(
            yaxis_linear, yaxis_log, "Linear scale", "Log scale", 0.0, "relayout", 1.18
        )
    )
        
    visibles = [True, False]*nrows*ncols
    updatemenus.append(
        add_buttons(
            {"visible": visibles},
            {"visible": visibles[::-1]},
            "Default", "Skip outliers", 0.65, "update", 1.18
        )
    )
            
    fig.update_layout(
        height=height,
        width=width,
        **axis_params,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1
        ),
        updatemenus=updatemenus
    )

    fig.show()
    if neptune_run:
        run = kwargs['run']
        run_plot_name = kwargs['run_plot_name']
        run[run_plot_name].upload(neptune.types.File.as_html(fig))
