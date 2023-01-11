import numpy as np
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


def mean_absolute_percentage_error(y_true, y_fit):
    return np.mean(np.abs((y_true - y_fit)/y_true))*100


def plot_baselines_diffs(wfs_array, baseline_array,
                         evt_range, nrows=4, ncols=6,
                         Nbins=100, vertical_spacing=0.1,
                         horizontal_spacing=0.1,
                         height=1000, width=1200): 
    
    fig = make_subplots(rows=nrows, cols=ncols,
                        vertical_spacing=vertical_spacing,
                        horizontal_spacing=horizontal_spacing)

    for i in range(nrows*ncols):
        charges, baselines_wf, baselines_gcu = charge_comp(
            wfs_array, baseline_array, i, 150, 250, evt_range)
    
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
            ), row=int(i/ncols)+1, col=i%ncols+1
        )

        fig.update_xaxes(title="bsln_wf - bsln_gcu",
                         row=int(i/ncols)+1, col=i%ncols+1)

    axis_params = {}
    for i in range(1, nrows*ncols+1):
        axis_params['xaxis{}'.format(i)] = xaxis
        axis_params['yaxis{}'.format(i)] = yaxis

    fig.update_layout(
        height=height,
        width=width,
        barmode='overlay',
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
    

def plot_charges_hist(wfs_array, baseline_array, evt_range,
                      nrows=4, ncols=6, Nbins=100,
                      vertical_spacing=0.1,
                      horizontal_spacing=0.1,
                      height=1000, width=1200): 
    
    fig = make_subplots(rows=nrows, cols=ncols,
                        vertical_spacing=vertical_spacing,
                        horizontal_spacing=horizontal_spacing)
    
    for i in range(nrows*ncols):
        charges, baselines_wf, baselines_gcu = charge_comp(
            wfs_array, baseline_array, i, 150, 250, evt_range)
        
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

        fig.update_yaxes(type="log", row=int(i/ncols)+1, col=i%ncols+1)
        fig.update_xaxes(title="Charge", row=int(i/ncols)+1, col=i%ncols+1)

    axis_params = {}
    for i in range(1, nrows*ncols+1):
        axis_params['xaxis{}'.format(i)] = xaxis
        axis_params['yaxis{}'.format(i)] = yaxis
                
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


def plot_charges_scatter(wfs_array, charge_array,
                         baseline_array, evt_range,
                         nrows=4, ncols=6,
                         vertical_spacing=0.1,
                         horizontal_spacing=0.1,
                         height=1000, width=1200): 
    
    fig = make_subplots(rows=nrows, cols=ncols,
                        vertical_spacing=vertical_spacing,
                        horizontal_spacing=horizontal_spacing)
    
    for i in range(nrows*ncols):
        charges, baselines_wf, baselines_gcu = charge_comp(
            wfs_array, baseline_array, i, 150, 250, evt_range)

        fig.add_trace(
            go.Scattergl(
                x = charges,
                y = charge_array[evt_range[0]:evt_range[1], i],
                name=f"Channel: {i}",
                mode='markers',
            ), row=int(i/ncols)+1, col=i%ncols+1
        )
        
        fig.update_xaxes(title="Manuanlly calc. Q", row=int(i/ncols)+1, col=i%ncols+1)
        fig.update_yaxes(title="GCU calc. Q", row=int(i/ncols)+1, col=i%ncols+1)

    axis_params = {}
    for i in range(1, nrows*ncols+1):
        axis_params['xaxis{}'.format(i)] = xaxis
        axis_params['yaxis{}'.format(i)] = yaxis

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

    
def wfs_2d_plot_by_channels(data, xlabel, ylabel,
                plot_width=200, plot_height=200,
                col_wrap=6, height=500, width=950):
    
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


def plot_wf_diff_channels_same_evt(
    wfs_array, EvtNumber, range_y=[11200, 11800],
    nrows=4, ncols=6, height=800, width=1400):
    
    fig = make_subplots(rows=nrows, cols=ncols)

    for i in range(nrows*ncols):
        fig.add_trace(
            go.Scattergl(
                x = np.arange(wfs_array.shape[2]),
                y = wfs_array[EvtNumber, i, :].flatten(),
                name=f"Channel {i}"
        ), row=int(i/ncols)+1, col=i%ncols+1
    )

    axis_params = {}
    for i in range(1, nrows*ncols+1):
        axis_params['xaxis{}'.format(i)] = xaxis
        axis_params['yaxis{}'.format(i)] = yaxis

    for i in range(nrows*ncols):
        fig.update_yaxes(range=range_y, row=int(i/nrows)+1, col=i%ncols+1)

    fig.update_layout(
        **axis_params,
        height=height,
        width=width,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1
        )
    )

    fig.show()
    

def plot_wf_same_channel_diff_evts(
    wfs_array, ChannelNumber, range_y=[11200, 11800],
    nrows=10, ncols=6, height=1600, width=1400):

    evtIds = np.random.randint(2000, wfs_array.shape[0], size=nrows*ncols)
    fig = make_subplots(rows=nrows, cols=ncols)

    for i in range(nrows*ncols):
        fig.add_trace(
            go.Scattergl(
                x = np.arange(wfs_array.shape[2]),
                y = wfs_array[evtIds[i], ChannelNumber, :].flatten(),
                name=f"EvtN: {evtIds[i]}",
        ), row=int(i/ncols)+1, col=i%ncols+1
    )

    axis_params = {}
    for i in range(1, nrows*ncols+1):
        axis_params['xaxis{}'.format(i)] = xaxis
        axis_params['yaxis{}'.format(i)] = yaxis

    for i in range(nrows*ncols):
        fig.update_yaxes(range=[11200, 11800], row=int(i/ncols)+1, col=i%ncols+1)

    fig.update_layout(
        **axis_params,
        height=height,
        width=width,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1
        )
    )

    fig.show()
    

def rates_fit_plots(
    runs_list, distrs, fit_params, rates, expected_rates,
    colors=['royalblue', 'darkred'], log_scale=False
):
    
    fig = go.Figure()

    for i in range(len(runs_list)):
        x = np.linspace(0, distrs[i].max()*8e-9, 10000)
        y = expon.pdf(x, *fit_params[i])
                
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name=runs_list[i] + f". Fit rate λ: {np.round(rates[i], 2)}" + \
                                    f". IPbus appr. rate λ: {expected_rates[i]}",
                mode='lines',
                line=dict(
                    color=colors[0],
                ),
                visible=(i==0)
            )
        )

    for i in range(len(runs_list)):
        counts, bins = np.histogram(distrs[i]*8e-9, density=True, bins=100)
        bins = 0.5 * (bins[:-1] + bins[1:]) 
        
        # gof = mean_absolute_percentage_error(counts, expon.pdf(bins, *fit_params[i]))

        fig.add_trace(
            go.Bar(
                x=bins,
                y=counts,
                showlegend=False,
                # name=f"MAPE: {np.round(gof, 2)}",
                marker=dict(
                    color=colors[1],
                    line=dict(
                        color='black',
                        width=1.5,
                    )
                ),
                visible=(i==0)
            )
        )

    buttons = []
    for N in range(len(runs_list)): 
        buttons.append(
            dict(
                 args=['visible', [False]*N + [True] + [False]*(len(runs_list)-1-N)],
                     label=f'Run {N+1}',
                 method='restyle'
            )
        )
        
    if log_scale:
        fig.update_yaxes(type="log")
        title = "Log. scale. Press the button:"
        x_button=0.35
    else:
        title="Press the button:"
        x_button=0.25

    fig.update_layout(
        title=title,
        xaxis_title="Timestamp diffs",
        showlegend=True,
        xaxis = xaxis,
        yaxis = yaxis,
        height=500,
        width=950,
        updatemenus=list([
            dict(
                x=x_button,
                y=1.23,
                yanchor='top',
                buttons=buttons
            ),
        ]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1
        )
    )

    fig.show()
    
    
def plot_distrs_one_param(
    params_list, param, names, file_name, colors_flag=False,
    scale_factor=1, bkg_n=-1, Nbins=200,
    range_x=[0, 0.005], log=True,
    bkg_subtract=False, line_width=0.1,
    opacity=0.75, height=500, width=950,
    left_shift=0, **kwargs
):
    
    if colors_flag:
        colors = kwargs['colors']
    else:
        colors=[None]*len(params_list)
    
    
    fig = go.Figure()
    
    if not bkg_subtract:
        for i in range(len(params_list)):
            counts, bins = np.histogram(params_list[i][param][left_shift:],
                                        bins=Nbins, range=range_x)
            bins = 0.5 * (bins[:-1] + bins[1:])

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
    else:
        counts_bkg, bins = np.histogram(params_list[bkg_n][param][left_shift:],
                                        bins=Nbins, range=range_x)
        bins = 0.5 * (bins[:-1] + bins[1:])

        for i in range(len(params_list)):
            if i != bkg_n:
                counts, bins = np.histogram(params_list[i][param][left_shift:],
                                            bins=Nbins, range=range_x)
                bins = 0.5 * (bins[:-1] + bins[1:])

                fig.add_trace(
                    go.Bar(
                        x=bins,
                        y=counts - scale_factor*counts_bkg,
                        opacity=opacity,
                        name=names[i],
                        colors=colors[i],
                        text=f"Total entries: {sum(counts)}",
                        textposition="none",
                        showlegend=True,
                        marker=dict(
                            line=dict(
                                color='black',
                                width=line_width,
                            )
                        )
                    )
                )

    if log:
        fig.update_yaxes(type="log",)
    fig.update_xaxes(range=range_x,)

    fig.update_layout(
        xaxis_title="Total charge",
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
        )
    )

    fig.show()
    pio.write_image(fig, f'plots/calib_12_2022/{file_name}.pdf', width=900, height=600)
    
def plot_charges_by_channels_src_and_bkg(charges, timestamps, source_names, Nbins,
                                     nrows=4, ncols=6, left_shift=1000,
                                     bkg_subtract=False, line_width=0.0,
                                     opacity=0.75, log_scale=True, 
                                     height=1000, width=1600):

    bkg_times = timestamps[0]
    cs_times = timestamps[1]
    scale_factors = cs_times / bkg_times

    fig = make_subplots(rows=nrows, cols=ncols)

    for i in range(nrows*ncols):
        if not bkg_subtract:
            counts_bkg, bins = np.histogram(charges[0]['charge'][left_shift:],
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
                ), row=int(i/ncols)+1, col=i%ncols+1
            )

            counts_src, bins = np.histogram(charges[1]['charge'][left_shift:],
                                        density=True, bins=Nbins)
            bins = 0.5 * (bins[:-1] + bins[1:]) 

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
                    showlegend=True,
                ), row=int(i/ncols)+1, col=i%ncols+1
            )

            if log_scale:
                fig.update_yaxes(type="log", row=int(i/ncols)+1, col=i%ncols+1)
        else:
            counts_bkg, bins = np.histogram(charges[0]['charge'][left_shift:],
                                        density=True, bins=Nbins)
            counts_src, bins = np.histogram(charges[1]['charge'][left_shift:],
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
                    name=f"Channel: {i}",
                    showlegend=True,
                ), row=int(i/ncols)+1, col=i%ncols+1
            )

            if log_scale:
                fig.update_yaxes(type="log", row=int(i/ncols)+1, col=i%ncols+1)

    axis_params = {}
    for i in range(1, nrows*ncols+1):
        axis_params['xaxis{}'.format(i)] = xaxis
        axis_params['yaxis{}'.format(i)] = yaxis

    fig.update_layout(
        title=f"{source_names[0]} + {source_names[1]}",
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
        )
    )

    fig.show()