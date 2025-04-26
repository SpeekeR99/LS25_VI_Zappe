import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import datetime
import copy

import plotly.graph_objects as go
import plotly_resampler
from plotly_resampler import FigureResampler

import dash
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc


timestamps_graph_labels = None


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

    return data, np.array(names)


def aggregate_data(all_data, metric="Ask Price 1", aggregation=np.mean, time_window=3600):
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


def create_price_graph(timestamps, ask_prices, bid_prices, imbalance_indices, freqs, cancels, name, how_many_x_ticks=75):
    # Convert timestamps to HH:MM:SS format
    global timestamps_graph_labels
    timestamps_graph_labels = [datetime.datetime.fromtimestamp(int(ts) / 1e9).strftime("%H:%M:%S.%f") for ts in timestamps]
    # Convert to 0 - n
    timestamps_graph = list(range(len(timestamps_graph_labels)))
    # Tick every 10000 timestamps
    tickvals = list(range(0, len(timestamps), len(timestamps) // how_many_x_ticks))
    ticklabels = [timestamps_graph_labels[i] for i in tickvals]
    # Go from HH:MM:SS.nnnnnn to truly HH:MM:SS
    ticklabels = [ts[:8] for ts in ticklabels]

    def interpolate_color(color1, color2, factor):
        return tuple(int(color1[i] + factor * (color2[i] - color1[i])) for i in range(3))

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

    price_graph_fig.add_trace(  # Highlight trace for the future hovering on the HeatMap
        go.Scattergl(
            name="Highlight",
            yaxis="y1",
            mode="lines",
            fill="toself",
            line=dict(width=2, color="rgba(25, 25, 100, 1)"),
            fillcolor="rgba(185, 215, 255, 0.3)",
            hoverinfo="skip",
            showlegend=False,
        ),
        hf_x=[],
        hf_y=[],
    )

    price_graph_fig.update_layout(
        title=f"{name}",
        xaxis={"title": "Timestamp", "tickmode": "array", "tickvals": tickvals, "ticktext": ticklabels},
        yaxis={"title": "Price", "side": "left"},
        yaxis2={"title": "Imbalance index", "side": "right", "overlaying": "y", "anchor": "free", "autoshift": True, "range": [-1, 1]},
        yaxis3={"title": "Incoming messages (per sec)", "side": "right", "overlaying": "y", "anchor": "free", "autoshift": True},
        yaxis4={"title": "Cancellations rate", "side": "right", "overlaying": "y", "anchor": "free", "autoshift": True},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        clickmode="event+select",
        hovermode="x unified",
        plot_bgcolor="#f9f9f9",
    )

    return price_graph_fig


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
    # data = load_data("20210319", "1176", "2299728", level_depth=level_depth)
    # all_data = [data]
    # names = ["20210319_1176_2299728"]
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

    # Heatmap
    heatmap_fig = go.Figure()

    # Create the heatmap
    heatmap_fig.add_trace(
        go.Heatmap(
            z=aggregated_data,
            x=[f"{i // 3600:02d}:{i % 3600 // 60:02d}" for i in range(0, 24 * 3600, time_window_aggregation)],
            y=names,
            colorscale="Viridis",
            colorbar=dict(),
            hoverongaps=False,
            zmin=np.min(aggregated_data),
            zmax=np.max(aggregated_data),
        )
    )

    day_sec = 60 * 60 * 24
    # Update layout
    heatmap_fig.update_layout(
        title=f"{chosen_aggregation} of {metric} Heatmap (Normalized Values)",
        xaxis={"title": "Time"},
        yaxis={"title": "Day/Product"},
        clickmode="event+select",
        hovermode="x unified",
        plot_bgcolor="#f9f9f9",
    )

    app.layout = html.Div([
        # Full-width Price Graph
        html.Div([
            dcc.Loading(
                dcc.Graph(
                    id="price_graph",
                    figure=go.Figure(),  # When I had a real figure here, the reset axes button was buggy...
                    config={
                        "toImageButtonOptions": {
                            "format": "png",
                            "filename": "price_graph",
                            "width": 1920,
                            "height": 1080,
                            "scale": 3
                        }
                    }
                ),
                type="circle"
            )
        ], style={
            "marginBottom": "1rem",
            "boxShadow": "0 4px 8px rgba(0,0,0,0.1)",
            "borderRadius": "10px",
            "padding": "0.5rem",
            "backgroundColor": "#f9f9f9"
        }),

        # FLEX CONTAINER: Heatmap + Settings
        html.Div([
            # SETTINGS - LEFT COLUMN
            html.Div([
                html.H4("Heatmap Settings", style={"marginBottom": "1rem"}),

                html.P("Select a metric:", style={"fontWeight": "bold", "marginBottom": "0.5rem"}),
                dcc.Dropdown(
                    id="metric_dropdown",
                    options=[{"label": label, "value": value} for label, value in [
                        ("Ask Price", "Ask Price 1"),
                        ("Bid Price", "Bid Price 1"),
                        ("Ask Volume", "Ask Volume 1"),
                        ("Bid Volume", "Bid Volume 1"),
                        ("Imbalance Index", "Imbalance Index"),
                        ("Frequency of Incoming Messages", "Frequency of Incoming Messages"),
                        ("Cancellations Rate", "Cancellations Rate"),
                        ("High Quoting Activity", "High Quoting Activity"),
                        ("Unbalanced Quoting", "Unbalanced Quoting"),
                        ("Low Execution Probability", "Low Execution Probability"),
                        ("Trades Oppose Quotes", "Trades Oppose Quotes"),
                        ("Cancels Oppose Trades", "Cancels Oppose Trades")
                    ]],
                    value=metric,
                    clearable=False,
                    style={"marginBottom": "0.5rem"}
                ),
                html.P(id="metric_description", style={"marginBottom": "1rem", "fontStyle": "italic"}),

                html.P("Select an aggregation function:", style={"fontWeight": "bold", "marginBottom": "0.5rem"}),
                dcc.Dropdown(
                    id="aggregation_dropdown",
                    options=[{"label": x, "value": x} for x in ["Mean", "Median", "Max", "Min", "Std"]],
                    value=chosen_aggregation,
                    clearable=False,
                    style={"marginBottom": "1rem"}
                ),

                html.P("Select a time window for the heatmap (in seconds):", style={"fontWeight": "bold", "marginBottom": "0.5rem"}),
                dcc.Input(
                    id="time_window_input",
                    type="number",
                    value=time_window_aggregation,
                    min=1,
                    max=86400,
                    step=1,
                    placeholder="Time Window (seconds)",
                    style={"width": "100%", "marginBottom": "1rem"}
                ),

                html.Button("Apply", id="update_heatmap_button", n_clicks=0, style={
                    "marginTop": "1rem",
                    "width": "100%",
                    "padding": "0.75rem 1rem",
                    "fontSize": "1rem",
                    "fontWeight": "bold",
                    "color": "#fff",
                    "backgroundColor": "#007BFF",  # Modern blue
                    "border": "none",
                    "borderRadius": "8px",
                    "boxShadow": "0 4px 6px rgba(0, 123, 255, 0.3)",
                    "cursor": "pointer",
                    "transition": "background-color 0.3s ease-in-out",
                }),
            ], style={
                "flex": "0 0 300px",  # Fixed width
                "padding": "0.5rem",
                "boxShadow": "0 4px 8px rgba(0,0,0,0.1)",
                "borderRadius": "10px",
                "backgroundColor": "#ffffff",
                "minWidth": "250px"
            }),

            # HEATMAP - RIGHT COLUMN
            html.Div([
                dcc.Loading(
                    dcc.Graph(
                        id="heatmap_graph",
                        figure=heatmap_fig,
                        clear_on_unhover=True,
                        config={
                            "toImageButtonOptions": {
                                "format": "png",
                                "filename": "heatmap_graph",
                                "width": 1920,
                                "height": 1080,
                                "scale": 3
                            }
                        }
                    ),
                    type="circle"
                )
            ], style={
                "flex": "1",
                "marginLeft": "1rem",
                "padding": "0.5rem",
                "boxShadow": "0 4px 8px rgba(0,0,0,0.1)",
                "borderRadius": "10px",
                "backgroundColor": "#f9f9f9",
                "minWidth": "0"  # important to prevent overflow
            })
        ], style={
            "display": "flex",
            "flexWrap": "nowrap",  # force side-by-side
            "alignItems": "flex-start",
            "gap": "1rem",
        }),
    ], style={"padding": "0.5rem", "fontFamily": "Arial, sans-serif", "backgroundColor": "#f0f2f5"})

    last_hover_label = None
    last_heatmap_click_count = None


    @app.callback(
        Output("metric_description", "children"),
        Input("metric_dropdown", "value")
    )
    def update_description(selected_metric):
        descriptions = {
            "Ask Price 1": "The lowest price a seller is willing to accept.",
            "Bid Price 1": "The highest price a buyer is willing to pay.",
            "Ask Volume 1": "The total number of offers available at the best ask price.",
            "Bid Volume 1": "The total number of offers available at the best bid price.",
            "Imbalance Index": "Measures the difference between buy and sell interest.",
            "Frequency of Incoming Messages": "Moving average (5 minutes window) of how often order book updates are received.",
            "Cancellations Rate": "Moving average (5 minutes window) of the rate at which orders are canceled.",
            "High Quoting Activity": "Indicates rapid updates in order quotes (changes in volumes).",
            "Unbalanced Quoting": "Shows bias towards one side of the market (buy or sell).",
            "Low Execution Probability": "\"Chance\" of execution given the current quoting activity.",
            "Trades Oppose Quotes": "Binary. 1 if the trade is on the opposite side of the more recently quoted side, 0 otherwise.",
            "Cancels Oppose Trades": "Binary. 1 if the trade is on the opposite side of the more recently canceled side, 0 otherwise."
        }
        return descriptions[selected_metric] if selected_metric in descriptions else "No description available."


    @app.callback(
        Output("price_graph", "figure"),
        Input("heatmap_graph", "clickData"),
        Input("heatmap_graph", "hoverData"),
        State("time_window_input", "value"),
        State("price_graph", "figure"),
    )
    def update_price_graph(heatmap_click, heatmap_hover, selected_time_window, price_fig):
        nonlocal timestamps, ask_prices, bid_prices, imbalance_indices, freqs, cancels, last_heatmap_click_count, last_hover_label

        updated = False

        # Change the price graph based on the heatmap click
        if heatmap_click and last_heatmap_click_count != heatmap_click:
            last_heatmap_click_count = heatmap_click
            # Get the clicked point
            clicked_point = heatmap_click["points"][0]
            # Get the index of the clicked point
            clicked_index = np.where(names == clicked_point["y"])[0][0]
            # Get the data for the clicked point
            clicked_data = all_data[clicked_index]

            timestamps = clicked_data["Time"].values
            ask_prices = [clicked_data[f"Ask Price {i}"].values for i in range(1, level_depth + 1)]
            bid_prices = [clicked_data[f"Bid Price {i}"].values for i in range(1, level_depth + 1)]
            imbalance_indices = clicked_data["Imbalance Index"].values
            freqs = clicked_data["Frequency of Incoming Messages"].values
            cancels = clicked_data["Cancellations Rate"].values

            # Price graph
            price_fig = create_price_graph(timestamps, ask_prices, bid_prices, imbalance_indices, freqs, cancels, names[clicked_index])
            price_fig.register_update_graph_callback(app=app, graph_id="price_graph")  # Resampler callback

            last_hover_label = None  # Click resets the hover as well
            updated = True

        if heatmap_hover and last_hover_label != heatmap_hover:
            updated = True
            hovered_label = heatmap_hover["points"][0]["x"]

            last_hover_label = hovered_label

            h, m = map(int, hovered_label.split(":"))
            hovered_sec = h * 3600 + m * 60

            # Calculate start and end of the range (e.g., ±30 min = 1800 sec)
            highlight_start = hovered_sec - 3600
            highlight_end = hovered_sec + selected_time_window - 3600

            def find_index_for_sec(sec, ts_array):
                for i, ts in enumerate(ts_array):
                    if ts >= sec:
                        return i
                return len(ts_array) - 1

            timestamps_series = pd.Series(pd.to_datetime(timestamps, unit="ns"))
            seconds_since_midnight = (timestamps_series - timestamps_series.dt.normalize()).dt.total_seconds()
            x0 = find_index_for_sec(highlight_start, seconds_since_midnight)
            x1 = find_index_for_sec(highlight_end, seconds_since_midnight)

            y_min = np.nanmin(np.array(bid_prices + ask_prices))
            y_max = np.nanmax(np.array(bid_prices + ask_prices))

            # Define rectangle corners
            highlight_x = [
                x0,  # Top-left
                x1,  # Top-right
                x1,  # Bottom-right
                x0,  # Bottom-left
                x0,  # Closing the path
            ]

            highlight_y = [
                y_max,  # Top-left
                y_max,  # Top-right
                y_min,  # Bottom-right
                y_min,  # Bottom-left
                y_max,  # Closing the path
            ]

            for trace in price_fig["data"]:
                if "Highlight" in trace["name"]:
                    trace["x"] = highlight_x
                    trace["y"] = highlight_y

        if heatmap_hover is None and last_hover_label is not None:
            updated = True
            last_hover_label = None
            for trace in price_fig["data"]:
                if "Highlight" in trace["name"]:
                    trace["x"] = []
                    trace["y"] = []

        return price_fig if updated else dash.no_update


    @app.callback(
        Output("heatmap_graph", "figure"),
        Input("update_heatmap_button", "n_clicks"),
        State("metric_dropdown", "value"),
        State("aggregation_dropdown", "value"),
        State("time_window_input", "value"),
        State("heatmap_graph", "figure"),
    )
    def update_heatmap(update_heatmap_button,  selected_metric, selected_aggregation, selected_time_window, heatmap_fig):
        if not update_heatmap_button:
            return dash.no_update

        new_x = [f"{int(i // 3600):02d}:{int(i % 3600 // 60):02d}" for i in range(0, 24 * 3600, selected_time_window)]
        new_z = aggregate_data(all_data, metric=selected_metric, aggregation=aggregation_functions_map[selected_aggregation], time_window=selected_time_window)

        if new_x == heatmap_fig["data"][0]["x"] and np.array_equal(new_z, heatmap_fig["data"][0]["z"]):
            return dash.no_update

        heatmap_fig["data"][0]["x"] = new_x
        heatmap_fig["data"][0]["z"] = new_z
        heatmap_fig["layout"]["title"]["text"] = f"{selected_aggregation} of {selected_metric} Heatmap (Normalized Values)"

        return heatmap_fig


    # Run the Dash app
    app.run(host="127.0.0.1", port=8080, debug=False)


if __name__ == "__main__":
    main()
