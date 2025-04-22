import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import datetime

import plotly.graph_objects as go
import plotly_resampler
from plotly_resampler import FigureResampler

from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc


def load_data(date, market_segment_id, security, level_depth=1):
    lobster_fp = f"data/{date}_{market_segment_id}_{security}_lobster_augmented.csv"
    data = pd.read_csv(lobster_fp, sep=",")
    # Throw away Ask Price i and Bid Price i if the ith level > level_depth
    i = level_depth + 1
    while True:
        if f"Ask Price {i}" not in data.columns and f"Bid Price {i}" not in data.columns:
            break
        data = data.drop(columns=[f"Ask Price {i}", f"Bid Price {i}"], errors="ignore")
    return data


def load_all_data(level_depth=1):
    data = []
    names = []

    for file in os.listdir("data"):
        if file.endswith("_lobster_augmented.csv"):
            date, market_segment_id, security = file.split("_")[:3]
            names.append(f"{date}_{market_segment_id}_{security}")
            data.append(load_data(date, market_segment_id, security, level_depth=level_depth))

    return data, names


def aggregate_data(all_data, metric="Ask Price 1", aggregation=np.mean, time_window=3600):
    # X axis is time (timestamps), Y axis are different data files (all_data[0], all_data[1], ...)
    # Color is the price at that given time-window (aggregated -- e.g. mean, median, etc.)
    # Aggregate the whole day into 1 hour intervals -- 24 bins
    aggregated_data = []
    day_sec = 60 * 60 * 24
    for data in all_data:
        tmp_agg = []
        timestamps = pd.to_datetime(data["Time"].values, unit="ns")
        timestamps_series = pd.Series(timestamps)
        seconds_since_midnight = (timestamps_series - timestamps_series.dt.normalize()).dt.total_seconds()

        for i in range(0, day_sec, time_window):
            start_time = i
            end_time = i + time_window
            # Get the data in the time window
            data_in_window = data[(seconds_since_midnight >= start_time) & (seconds_since_midnight < end_time)]
            data_in_window = data_in_window[metric].dropna()
            # Aggregate the metric in the time window
            if data_in_window.empty:
                tmp_agg.append(np.nan)
                continue
            # Use the aggregation function passed as argument
            tmp_agg.append(aggregation(data_in_window.values))

        # Normalize the metric to [0, 1]
        tmp_agg = np.array(tmp_agg)
        tmp_agg = (tmp_agg - np.nanmin(tmp_agg)) / (np.nanmax(tmp_agg) - np.nanmin(tmp_agg))
        aggregated_data.append(tmp_agg)

    # Convert to numpy array
    aggregated_data = np.array(aggregated_data)

    return aggregated_data


def interpolate_color(color1, color2, factor):
    return tuple(int(color1[i] + factor * (color2[i] - color1[i])) for i in range(3))


def main():
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    level_depth = 5
    chosen_aggregation = "Mean"
    aggregation_functions_map = {
        "Mean": np.mean,
        "Median": np.median,
        "Max": np.max,
        "Min": np.min,
        "Std": np.std
    }
    time_window_aggregation = 3600
    metric = "Ask Price 1"
    print("Loading data...")
    # data = load_data("20191202", "688", "4128839", level_depth=level_depth)
    # data_2 = load_data("20210319", "1176", "2299728", level_depth=level_depth)
    # all_data = [data, data_2]
    # names = ["20191202_688_4128839", "20210319_1176_2299728"]
    all_data, names = load_all_data(level_depth=level_depth)
    data = all_data[0]
    aggregated_data = aggregate_data(all_data, metric=metric, aggregation=aggregation_functions_map[chosen_aggregation], time_window=time_window_aggregation)
    print("Data loaded.")

    timestamps = data["Time"].values
    ask_prices = [data[f"Ask Price {i}"].values for i in range(1, level_depth+1)]
    bid_prices = [data[f"Bid Price {i}"].values for i in range(1, level_depth+1)]
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

    # Price graph
    price_graph_fig = FigureResampler(go.Figure(), default_downsampler=plotly_resampler.MinMaxLTTB(parallel=True))

    for i, ask_price in enumerate(ask_prices, 1):
        price_graph_fig.add_trace(
            go.Scattergl(
                name=f"Ask {i}",
                yaxis="y1"
            ),
            hf_x=timestamps_graph,
            hf_y=ask_price,
            hf_marker_color=f"rgb" + str(interpolate_color((230, 31, 7), (255, 255, 255), (i - 1) / len(ask_prices)))
        )

    for i, bid_price in enumerate(bid_prices, 1):
        price_graph_fig.add_trace(
            go.Scattergl(
                name=f"Bid {i}",
                yaxis="y1"
            ),
            hf_x=timestamps_graph,
            hf_y=bid_price,
            hf_marker_color=f"rgb" + str(interpolate_color((94, 163, 54), (255, 255, 255), (i - 1) / len(bid_prices)))
        )

    price_graph_fig.add_trace(
        go.Scattergl(
            name="Imbalance index",
            yaxis="y2",
            opacity=0.1
        ),
        hf_x=timestamps_graph,
        hf_y=imbalance_indices,
        hf_marker_color="rgb(0, 0, 255)"
    )

    price_graph_fig.add_trace(
        go.Scattergl(
            name="Incoming messages (per sec)",
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
        title="Price Graph",
        xaxis={"title": "Timestamp", "tickmode": "array", "tickvals": tickvals, "ticktext": ticklabels},
        yaxis={"title": "Price", "side": "left"},
        yaxis2={"title": "Imbalance index", "side": "right", "overlaying": "y", "anchor": "free", "autoshift": True, "range": [-1, 1]},
        yaxis3={"title": "Incoming messages (per sec)", "side": "right", "overlaying": "y", "anchor": "free", "autoshift": True},
        yaxis4={"title": "Cancellations rate", "side": "right", "overlaying": "y", "anchor": "free", "autoshift": True},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        clickmode="event+select",
        hovermode="x unified"
    )

    price_graph_fig.register_update_graph_callback(app=app, graph_id="price_graph")

    # Heatmap
    heatmap_fig = go.Figure()

    # Create the heatmap
    heatmap_fig.add_trace(
        go.Heatmap(
            z=aggregated_data,
            x=[f"{i // 3600}:00" for i in range(0, 24 * 3600, time_window_aggregation)],
            y=names,
            colorscale="Viridis",
            colorbar=dict(title=metric),
            hoverongaps=False,
            zmin=np.min(aggregated_data),
            zmax=np.max(aggregated_data),
        )
    )

    # Update layout
    heatmap_fig.update_layout(
        title=f"{chosen_aggregation} of {metric} Heatmap",
        xaxis=dict(title="Time"),
        yaxis=dict(title="Day/Product"),
        clickmode="event+select",
        hovermode="x unified"
    )

    # HTML webpage of the visualizations
    app.layout = html.Div([
        html.Div([
            dcc.Loading(
                dcc.Graph(
                    id="price_graph",
                    figure=price_graph_fig,
                    config={
                        "toImageButtonOptions": {
                            "format": "png",
                            "filename": "price_graph",
                            "width": 1920,
                            "height": 1080,
                            "scale": 3
                        }
                    }
                ), type="circle")
        ], style={
            "display": "block",
            "justify-content": "center",
            "align-items": "center",
        }),
        html.Div([
            # DropDownMenu of Metrics
            dcc.Dropdown(
                id="metric_dropdown",
                options=[
                    {"label": "Ask Price", "value": "Ask Price 1"},
                    {"label": "Bid Price", "value": "Bid Price 1"},
                    {"label": "Ask Volume", "value": "Ask Volume 1"},
                    {"label": "Bid Volume", "value": "Bid Volume 1"},
                    {"label": "Imbalance Index", "value": "Imbalance Index"},
                    {"label": "Frequency of Incoming Messages", "value": "Frequency of Incoming Messages"},
                    {"label": "Cancellations Rate", "value": "Cancellations Rate"},
                    {"label": "High Quoting Activity", "value": "High Quoting Activity"},
                    {"label": "Unbalanced Quoting", "value": "Unbalanced Quoting"},
                    {"label": "Low Execution Probability", "value": "Low Execution Probability"},
                    {"label": "Trades Oppose Quotes", "value": "Trades Oppose Quotes"},
                    {"label": "Cancels Oppose Trades", "value": "Cancels Oppose Trades"}
                ],
                value=metric,
                clearable=False,
                style={"width": "50%"}
            ),
            # DropDownMenu of Aggregation Functions
            dcc.Dropdown(
                id="aggregation_dropdown",
                options=[
                    {"label": "Mean", "value": "Mean"},
                    {"label": "Median", "value": "Median"},
                    {"label": "Max", "value": "Max"},
                    {"label": "Min", "value": "Min"},
                    {"label": "Std", "value": "Std"}
                ],
                value=chosen_aggregation,
                clearable=False,
                style={"width": "50%"}
            ),
            dcc.Loading(
                dcc.Graph(
                    id="heatmap_graph",
                    figure=heatmap_fig,
                    config={
                        "toImageButtonOptions": {
                            "format": "png",
                            "filename": "heatmap_graph",
                            "width": 1920,
                            "height": 1080,
                            "scale": 3
                        }
                    }
                ), type="circle")
        ], style={
            "display": "block",
            "justify-content": "center",
            "align-items": "center",
        })
    ])


    # Callback to update the heatmap based on the selected metrics and aggregation functions
    @app.callback(
        Output("heatmap_graph", "figure"),
        Input("metric_dropdown", "value"),
        Input("aggregation_dropdown", "value"),
        State("heatmap_graph", "figure")
    )
    def update_heatmap(selected_metric, selected_aggregation, fig):
        # Update the heatmap data
        fig["data"][0]["z"] = aggregate_data(all_data, metric=selected_metric, aggregation=aggregation_functions_map[selected_aggregation], time_window=time_window_aggregation)
        fig["data"][0]["colorbar"]["title"]["text"] = selected_metric
        fig["layout"]["title"]["text"] = f"{selected_aggregation} of {selected_metric} Heatmap"
        return fig


    # Run the Dash app
    app.run_server(host="127.0.0.1", port=8080, debug=False)


if __name__ == "__main__":
    main()
