import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots
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


def plot_charges_by_channels(nHits_array, colors, sources,
                             pmt_x, pmt_y, pmt_z, Nbins, nrows=8, ncols=6,
                             bkg_subtract=False,  line_width=0.0, 
                             opacity=0.75, height=1800, 
                             width=1600, **kwargs):

    fig = make_subplots(rows=nrows, cols=ncols)

    for i in range(nrows*ncols):
        for k in range(len(sources)):
                counts, bins = np.histogram(nHits_array[k][:, i], bins=Nbins)
                bins = 0.5 * (bins[:-1] + bins[1:])
                
                if k==0:
                    showlegend=True
                else:
                    showlegend=False
                    
                fig.add_trace(
                    go.Bar(
                        x=bins,
                        y=counts,
                        opacity=opacity,
                        showlegend=showlegend,
                        marker=dict(
                            color=colors[k],
                            line=dict(
                                width=line_width,
                            )
                        ),
                        name=f"PMT: {i}",
                        text=f"PMT: {i}, source: {sources[k]}. PMT's coors: [{str(pmt_x[i])[:4]}, {str(pmt_y[i])[:4]}, {str(pmt_z[i])[:4]}]",
                        textposition="none",
                    ), row=int(i/ncols)+1, col=i%ncols+1
                )

                fig.add_trace(
                    go.Scatter(
                        x=bins,
                        y=counts,
                        opacity=opacity,
                        showlegend=showlegend,
                        visible=False,
                        line=dict(shape='hvh'),
                        marker=dict(
                            color=colors[k],
                            line=dict(
                                width=line_width,
                            )
                        ),
                        name=f"PMT: {i}",
                        text=f"PMT: {i}, source: {sources[k]}. PMT's coors: [{str(pmt_x[i])[:4]}, {str(pmt_y[i])[:4]}, {str(pmt_z[i])[:4]}]",
                    ), row=int(i/ncols)+1, col=i%ncols+1
                )

    visibles = [True, False] * nrows*ncols * len(counts)
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
            yaxis_linear, yaxis_log, "Linear scale", "Log scale", 0.0, "relayout", 1.15
        )
    )
    
    updatemenus.append(
        add_buttons(
            {"visible": visibles},
            {"visible": visibles[::-1]},
            "Bar", "Step", 0.75, "update", 1.15
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


def plot_distr(source_counts, source_bins, sources, colors, title, xtitle,
               xbut1=0.25, xbut2=0.8):
    
    fig = go.Figure()

    for i in range(len(sources)):
        fig.add_trace(
            go.Bar(
                x=source_bins[i],
                y=source_counts[i],
                opacity=0.7,
                name=sources[i],
                marker=dict(
                    color=colors[i],
                    line=dict(
                        width=0.0,
                    )
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=source_bins[i],
                y=source_counts[i],
                line=dict(shape='hvh'),
                opacity=0.7,
                visible=False,
                name=sources[i],
                marker=dict(
                    color=colors[i],
                    line=dict(
                        width=0.0,
                    )
                )
            )
        )
    
    visibles = [True, False]
    yaxis_linear = {"yaxis.type": "linear"}
    yaxis_log = {"yaxis.type": "log"}
    
    updatemenus = []
    updatemenus.append(add_buttons(
            yaxis_linear, yaxis_log, "Linear scale", "Log scale", xbut1, "relayout", 1.22
        )
    )
    
    updatemenus.append(
        add_buttons(
            {"visible": visibles},
            {"visible": visibles[::-1]},
            "Bar", "Step", xbut2, "update", 1.22
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title=xtitle,
        xaxis=xaxis,
        yaxis=yaxis,
        bargap=0.0,
        barmode='overlay',
        height=500,
        width=950,
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


def plot_fit(xs_data, ys_data, ys_fit, masks_fit, labels, colors):
    
    fig = go.Figure()
    
    for i in range(len(labels)):
    
            fig.add_trace(
                go.Scatter(
                    x=xs_data[i],
                    y=ys_data[i],
                    line=dict(shape='hvh'),
                    opacity=0.7,
                    name=labels[i],
                    marker=dict(
                        color=colors[i],
                        line=dict(
                            width=0.0,
                        )
                    )
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=xs_data[i][masks_fit[i]],
                    y=ys_fit[i][masks_fit[i]],
                    opacity=0.7,
                    showlegend=False,
                    name=labels[i],
                    marker=dict(
                        color='black',
                        line=dict(
                            width=0.0,
                        )
                    )
                )
            )
    
    fig.update_yaxes(type="log", range=[0, 3.5])
    
    fig.update_layout(
        height=600,
        width=950,
        xaxis_title="totalPE",
        xaxis=xaxis,
        yaxis=yaxis,
        barmode='overlay',
        bargap=0.0,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1
        ),
    )

    fig.show()


def plot_calib_curve(mus, mu_errors, exps,  xs):

    params, cov = np.polyfit(mus, exps, 1, cov=True)
    k, b = params

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            y=exps,
            x=mus,
            mode='markers',
            name="Sources",
            marker=dict(
                color='black'
            ),
            error_x=dict(
                    type='data',
                    width=5,
                    array=mu_errors,
                    visible=True
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=k*xs+b,
            mode='lines',
            name='Fit',
            marker=dict(
                color='royalblue'
            ),
        )
    )

    fig.update_layout(
        xaxis_title="PEs",
        yaxis_title="E [MeV]",
        xaxis=xaxis,
        yaxis=yaxis,
        height=500,
        width=950,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
    )

    fig.show()
    return k, b, cov
