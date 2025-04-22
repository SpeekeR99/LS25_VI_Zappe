import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import datetime

import plotly.graph_objects as go
import plotly_resampler
from plotly_resampler import FigureResampler

from dash import Dash, dcc, html
import dash_bootstrap_components as dbc


def load_data(date, market_segment_id, security, level_depth=1):
    lobster_fp = f"data/{date}_{market_segment_id}_{security}_lobster_augmented.csv"
    data = pd.read_csv(lobster_fp, sep=",")
    # Throw away Ask Price i and Bid Price i if the ith level > level_depth
    i = level_depth + 1
    while True:
        if f"Ask Price {i}" not in data.columns and f"Bid Price {i}" not in data.columns:
            break
        data = data.drop(columns=[f"Ask Price {i}", f"Bid Price {i}"], errors='ignore')
    return data


def interpolate_color(color1, color2, factor):
    return tuple(int(color1[i] + factor * (color2[i] - color1[i])) for i in range(3))


def main():
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    # Instrument, security, date
    security = "4128839"
    market_segment_id = "688"
    date = "20191202"

    level_depth = 5
    data = load_data(date, market_segment_id, security, level_depth=level_depth)

    timestamps = data["Time"].values
    ask_prices = [data[f'Ask Price {i}'].values for i in range(1, level_depth+1)]
    bid_prices = [data[f'Bid Price {i}'].values for i in range(1, level_depth+1)]
    imbalance_indices = data["Imbalance Index"].values
    freqs = data["Frequency of Incoming Messages"].values
    cancels = data["Cancellations Rate"].values

    # Convert timestamps to HH:MM:SS format
    timestamps_graph_labels = [datetime.datetime.fromtimestamp(int(ts) / 1e9).strftime("%H:%M:%S.%f") for ts in timestamps]
    # Convert to 0 - n
    timestamps_graph = list(range(len(timestamps_graph_labels)))
    # Tick every 10000 timestamps
    tickvals = list(range(0, len(timestamps), 10000))
    ticklabels = [timestamps_graph_labels[i] for i in tickvals]
    # Go from HH:MM:SS.nnnnnn to truly HH:MM:SS
    ticklabels = [ts[:8] for ts in ticklabels]

    price_graph_fig = FigureResampler(go.Figure(), default_downsampler=plotly_resampler.MinMaxLTTB(parallel=True))

    # Dummy scatter for hover
    price_graph_fig.add_trace(
        go.Scattergl(
            x=timestamps_graph,
            y=ask_prices[0],
            yaxis="y1",
            text=timestamps_graph_labels,
            opacity=0.0,
            showlegend=False,
            hovertemplate="%{text}<extra></extra>"
        )
    )

    for i, ask_price in enumerate(ask_prices, 1):
        price_graph_fig.add_trace(
            go.Scattergl(
                name=f'Ask {i}',
                yaxis="y1"
            ),
            hf_x=timestamps_graph,
            hf_y=ask_price,
            hf_marker_color=f'rgb' + str(interpolate_color((230, 31, 7), (255, 255, 255), (i - 1) / len(ask_prices)))
        )

    for i, bid_price in enumerate(bid_prices, 1):
        price_graph_fig.add_trace(
            go.Scattergl(
                name=f'Bid {i}',
                yaxis="y1"
            ),
            hf_x=timestamps_graph,
            hf_y=bid_price,
            hf_marker_color=f'rgb' + str(interpolate_color((94, 163, 54), (255, 255, 255), (i - 1) / len(bid_prices)))
        )

    price_graph_fig.add_trace(
        go.Scattergl(
            name='Imbalance index',
            yaxis="y2",
            opacity=0.1
        ),
        hf_x=timestamps_graph,
        hf_y=imbalance_indices,
        hf_marker_color="rgb(0, 0, 255)"
    )

    price_graph_fig.add_trace(
        go.Scattergl(
            name='Incoming messages (per sec)',
            yaxis="y3",
            opacity=0.25
        ),
        hf_x=timestamps_graph,
        hf_y=freqs,
        hf_marker_color="rgb(255, 0, 215)"
    )

    price_graph_fig.add_trace(
        go.Scattergl(
            name="Cancellations rate",
            yaxis="y4",
            opacity=0.25
        ),
        hf_x=timestamps_graph,
        hf_y=cancels,
        hf_marker_color="rgb(255, 215, 0)"
    )

    price_graph_fig.update_layout(
        title='Price Graph',
        xaxis={'title': 'Timestamp', "tickmode": "array", "tickvals": tickvals, "ticktext": ticklabels},
        yaxis={'title': 'Price', 'side': 'left'},
        yaxis2={'title': 'Imbalance index', 'side': 'right', 'overlaying': 'y', "anchor": "free", "autoshift": True, "range": [-1, 1]},
        yaxis3={'title': 'Incoming messages (per sec)', 'side': 'right', 'overlaying': 'y', "anchor": "free", "autoshift": True},
        yaxis4={'title': 'Cancellations rate', 'side': 'right', 'overlaying': 'y', "anchor": "free", "autoshift": True},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        clickmode='event+select',
        hovermode="x unified"
    )

    price_graph_fig.register_update_graph_callback(app=app, graph_id="price_graph")

    app.layout = html.Div([
        dcc.Graph(
            id='price_graph',
            figure=price_graph_fig,
            config={'displayModeBar': True, 'displaylogo': False}
        ),
    ])

    app.run_server(host="127.0.0.1", port=8080, debug=True)


if __name__ == '__main__':
    main()
