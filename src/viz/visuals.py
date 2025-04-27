import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import datetime

import plotly.graph_objects as go
import plotly_resampler
from plotly_resampler import FigureResampler

import dash
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

from src.viz.dataloader import load_data, load_all_data, aggregate_data


# Networking settings for the Dash app
HOST_ADDRESS = "127.0.0.1"
PORT = 8080

# Minute in seconds (60 seconds)
MINUTE_SEC = 60
# Hour in seconds (60 minutes)
HOUR_SEC = 60 * MINUTE_SEC
# Day in seconds (24 hours)
DAY_SEC = 24 * HOUR_SEC

# Global variables to store graph settings across callbacks
timestamps = None  # Timestamps of the data
ask_prices = None  # Ask prices
bid_prices = None  # Bid prices

timestamps_graph_labels = None  # Timestamps for the graph

last_hover_label = None  # Last hovered label in the heatmap
last_heatmap_click_count = None  # Last clicked point in the heatmap
last_update_heatmap_click_count = None  # Last clicked "Apply" button in the heatmap

chosen_aggregation = "Mean"  # Default aggregation function
aggregation_functions_map = {  # Map aggregation function names to actual function pointers (Python and pointers?)
    "Mean": np.mean,
    "Median": np.median,
    "Max": np.max,
    "Min": np.min,
    "Std": np.std
}
metric = "Ask Price 1"  # Default metric for aggregation
metric_descriptions_map = {  # Map metric names to their respective descriptions
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
time_window_aggregation = 3600  # Default time window for aggregation (1 hour)



def create_price_graph(timestamps, ask_prices, bid_prices, imbalance_indices, freqs, cancels, name, how_many_x_ticks=75):
    """
    Create the price graph with the given data
    :param timestamps: Timestamps of the data
    :param ask_prices: Ask prices
    :param bid_prices: Bid prices
    :param imbalance_indices: Imbalance index for each timestamp
    :param freqs: Frequency of incoming messages for each timestamp
    :param cancels: Cnaccellations rate for each timestamp
    :param name: Name of the product (Day_MarketSegmentID_SecurityID)
    :param how_many_x_ticks: Number of x ticks to show on the graph
    :return: The price graph figure
    """
    # Convert timestamps to HH:MM:SS format
    global timestamps_graph_labels
    timestamps_graph_labels = [datetime.datetime.fromtimestamp(int(ts) / 1e9).strftime("%H:%M:%S.%f") for ts in timestamps]
    # Convert to 0 - n
    timestamps_graph = list(range(len(timestamps_graph_labels)))
    # Ticks
    tickvals = list(range(0, len(timestamps), len(timestamps) // how_many_x_ticks))
    ticklabels = [timestamps_graph_labels[i] for i in tickvals]
    # Go from HH:MM:SS.nnnnnn to truly HH:MM:SS
    ticklabels = [ts[:8] for ts in ticklabels]


    def interpolate_color(color1, color2, factor):
        """
        Utility function to interpolate between two colors
        :param color1: Color 1
        :param color2: Color 2
        :param factor: Factor to interpolate between the two colors
        :return: Interpolated color
        """
        return tuple(int(color1[i] + factor * (color2[i] - color1[i])) for i in range(3))


    # Price graph
    price_graph_fig = FigureResampler(go.Figure(), default_downsampler=plotly_resampler.MinMaxLTTB(parallel=True))

    # Add traces for ask prices, bid prices, imbalance indices, frequencies, and cancellations
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

    # Highlight trace for the future hovering on the HeatMap
    price_graph_fig.add_trace(
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
        hf_x=[],  # Empty x and y to avoid showing the trace
        hf_y=[],  # Empty x and y to avoid showing the trace
    )

    price_graph_fig.update_layout(
        title=f"{name}",
        xaxis={"title": "Timestamp", "tickmode": "array", "tickvals": tickvals, "ticktext": ticklabels, "range": [0, len(timestamps_graph_labels) - 1]},
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
    """
    Main function to run the Dash app
    """
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    print("Loading data...")
    level_depth = 5
    all_data, names = load_all_data(level_depth=level_depth)
    aggregated_data = aggregate_data(all_data, metric=metric, aggregation=aggregation_functions_map[chosen_aggregation], time_window=time_window_aggregation)
    print("Data loaded.")

    # Placeholder for the price graph
    placeholder_fig = FigureResampler(go.Figure(), default_downsampler=plotly_resampler.MinMaxLTTB(parallel=True))
    placeholder_fig.update_layout(
        annotations=[
            {
                "text": "Select Day/Product in the heatmap to view the price graph",
                "xref": "paper", "yref": "paper",
                "x": 0.5, "y": 0.5,
                "showarrow": False,
                "font": { "size": 20 }
            }
        ]
    )
    placeholder_fig.register_update_graph_callback(app=app, graph_id="price_graph")  # Resampler callback

    # Heatmap
    heatmap_fig = go.Figure()
    heatmap_fig.add_trace(
        go.Heatmap(
            z=aggregated_data,
            x=[f"{i // HOUR_SEC:02d}:{i % HOUR_SEC // MINUTE_SEC:02d}" for i in range(0, DAY_SEC, time_window_aggregation)],
            y=names,
            colorscale="Viridis",
            colorbar=dict(),
            hoverongaps=False,
            zmin=np.min(aggregated_data),
            zmax=np.max(aggregated_data),
        )
    )
    heatmap_fig.update_layout(
        title=f"{chosen_aggregation} of {metric} Heatmap (Normalized Values)",
        xaxis={"title": "Time", "range": [-0.5, DAY_SEC / time_window_aggregation - 0.5]},
        yaxis={"title": "Day/Product"},
        clickmode="event+select",
        hovermode="x unified",
        plot_bgcolor="#f9f9f9",
    )

    # HTML Layout
    app.layout = html.Div([
        # Price Graph at the top
        html.Div([
            dcc.Loading(
                dcc.Graph(
                    id="price_graph",
                    figure=placeholder_fig,  # When I had a real figure here, the reset axes button was buggy...
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

        # Heatmap + Settings
        html.Div([
            # Settings (on the left)
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

                html.P("Select a time window for the heatmap (in seconds) (60 - 43200):", style={"fontWeight": "bold", "marginBottom": "0.5rem"}),
                dcc.Input(
                    id="time_window_input",
                    type="number",
                    value=time_window_aggregation,
                    min=60,
                    max=43200,
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

            # Heatmap (on the right)
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
                "minWidth": "0"
            })
        ], style={
            "display": "flex",
            "flexWrap": "nowrap",
            "alignItems": "flex-start",
            "gap": "1rem",
        }),
    ], style={"padding": "0.5rem", "fontFamily": "Arial, sans-serif", "backgroundColor": "#f0f2f5"})


    @app.callback(
        Output("metric_description", "children"),
        Input("metric_dropdown", "value")
    )
    def update_description(selected_metric):
        """
        Updates the description of the selected metric in the heatmap settings
        :param selected_metric: The user selected metric from the dropdown
        :return: The description of the selected metric
        """
        return metric_descriptions_map[selected_metric] if selected_metric in metric_descriptions_map else "No description available."


    @app.callback(
        Output("price_graph", "figure"),
        Input("heatmap_graph", "clickData"),
        Input("heatmap_graph", "hoverData"),
        State("time_window_input", "value"),
        State("price_graph", "figure"),
    )
    def update_price_graph(heatmap_click, heatmap_hover, selected_time_window, price_fig):
        """
        Updates the price graph based on several inputs:
            1) Click in the heatmap -- changes the data in the price graph
            2) Hover in the heatmap -- highlights the price graph (on the x axis)
        :param heatmap_click: Click event in the heatmap
        :param heatmap_hover: Hover event in the heatmap
        :param selected_time_window: Selected time window for the heatmap (State only)
        :param price_fig: Price graph figure (Current State -- will be changed during the callback)
        :return: Updated price graph figure
        """
        global timestamps, ask_prices, bid_prices, last_heatmap_click_count, last_hover_label

        # Boolean to check if the graph was actually updated
        updated = False

        # Transform the price graph to a FigureResampler object if it is not already
        if isinstance(price_fig, dict):
            price_fig = FigureResampler(price_fig, default_downsampler=plotly_resampler.MinMaxLTTB(parallel=True))

        # Change the price graph based on the heatmap click
        if heatmap_click and last_heatmap_click_count != heatmap_click:
            updated = True
            last_heatmap_click_count = heatmap_click
            last_hover_label = None  # Click resets the hover as well

            # Get the clicked point
            clicked_point = heatmap_click["points"][0]
            # Get the index of the clicked point
            clicked_index = names.index(clicked_point["y"])

            # Get the data for the clicked point
            clicked_data = all_data[clicked_index]

            timestamps = clicked_data["Time"].values  # Updates the global timestamps for others to use
            ask_prices = [clicked_data[f"Ask Price {i}"].values for i in range(1, level_depth + 1)]  # Same here
            bid_prices = [clicked_data[f"Bid Price {i}"].values for i in range(1, level_depth + 1)]  # And here
            imbalance_indices = clicked_data["Imbalance Index"].values
            freqs = clicked_data["Frequency of Incoming Messages"].values
            cancels = clicked_data["Cancellations Rate"].values

            # Price graph update
            price_fig = create_price_graph(timestamps, ask_prices, bid_prices, imbalance_indices, freqs, cancels, names[clicked_index])
            price_fig.register_update_graph_callback(app=app, graph_id="price_graph")  # Resampler callback

        # Update highlight if hovering on the heatmap
        if heatmap_hover and last_hover_label != heatmap_hover:
            updated = True

            # Get the hovered label
            hovered_label = heatmap_hover["points"][0]["x"]
            last_hover_label = hovered_label

            # Convert hovered label to seconds
            h, m = map(int, hovered_label.split(":"))
            hovered_sec = h * HOUR_SEC + m * MINUTE_SEC

            # Calculate start and end of the range
            highlight_start = hovered_sec - HOUR_SEC
            highlight_end = hovered_sec + selected_time_window - HOUR_SEC


            def find_index_for_sec(sec):
                """
                Find the index of the first timestamp that is greater than or equal to sec
                :param sec: The second to find the index for
                :return: The index of the first timestamp that is greater than or equal to sec
                """
                global timestamps
                timestamps_series = pd.Series(pd.to_datetime(timestamps, unit="ns"))
                seconds_since_midnight = (timestamps_series - timestamps_series.dt.normalize()).dt.total_seconds()
                for i, ts in enumerate(seconds_since_midnight):
                    if ts >= sec:
                        return i
                return len(seconds_since_midnight) - 1


            # Find the indices for the start and end of the highlight
            x0 = find_index_for_sec(highlight_start)
            x1 = find_index_for_sec(highlight_end)

            # Find the min and max y values for the highlight
            prices = np.array(bid_prices + ask_prices)
            y_min = np.nanmin(prices)
            y_max = np.nanmax(prices)

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

            # Update the highlight trace but keep the current x-range zoom
            current_range = price_fig.layout["xaxis"]["range"] if "xaxis" in price_fig.layout and "range" in price_fig.layout["xaxis"] else None
            price_fig.update_traces(
                selector=dict(name="Highlight"),
                x=highlight_x,
                y=highlight_y,
            )
            if current_range:  # Restore the zoom if it existed
                price_fig.update_layout(
                    xaxis={"range": current_range},
                )

        # If the user stopped hovering, it will be None, remove the highlight
        if heatmap_hover is None and last_hover_label is not None:
            updated = True
            last_hover_label = None

            # Remove the highlight
            price_fig.update_traces(
                selector=dict(name="Highlight"),
                x=[],
                y=[],
            )

        # Update the figure only if it was actually updated, else return no_update
        return price_fig if updated else dash.no_update


    @app.callback(
        Output("heatmap_graph", "figure"),
        Input("update_heatmap_button", "n_clicks"),
        Input("price_graph", "relayoutData"),
        State("metric_dropdown", "value"),
        State("aggregation_dropdown", "value"),
        State("time_window_input", "value"),
        State("heatmap_graph", "figure"),
    )
    def update_heatmap(update_heatmap_button, price_relayout, selected_metric, selected_aggregation, selected_time_window, heatmap_fig):
        """
        Updates the heatmap based on several inputs:
            1) Click on the "Apply" button -- changes everything about the Heatmap (metric, aggregation, time window)
            2) Zoom in the price graph -- changes the x-axis of the heatmap
        :param update_heatmap_button: "Apply" button click event
        :param price_relayout: Price graph zoom event
        :param selected_metric: User selected metric from the dropdown (State only)
        :param selected_aggregation: User selected aggregation function from the dropdown (State only)
        :param selected_time_window: User selected time window for the heatmap (State only)
        :param heatmap_fig: Heatmap figure (Current State -- will be changed during the callback)
        :return: Updated heatmap figure
        """
        global last_update_heatmap_click_count

        # Boolean to check if the graph was actually updated
        updated = False

        # Transform the heatmap graph to a Figure object if it is not already
        if isinstance(heatmap_fig, dict):
            heatmap_fig = go.Figure(heatmap_fig)

        # Update the whole heatmap, if the user clicked the button to apply the changes
        if update_heatmap_button and last_update_heatmap_click_count != update_heatmap_button:
            last_update_heatmap_click_count = update_heatmap_button

            # Calculate the new X and Z data for the heatmap
            new_x = [f"{int(i // HOUR_SEC):02d}:{int(i % HOUR_SEC // MINUTE_SEC):02d}" for i in range(0, DAY_SEC, selected_time_window)]
            new_z = aggregate_data(all_data, metric=selected_metric, aggregation=aggregation_functions_map[selected_aggregation], time_window=selected_time_window)

            # Check if the user didn't just click "Apply" button without choosing any changes
            if new_x != heatmap_fig["data"][0]["x"] or not np.array_equal(new_z, heatmap_fig["data"][0]["z"]):
                updated = True

                # Update the heatmap data
                heatmap_fig.update_traces(
                    selector=dict(type="heatmap"),
                    z=new_z,
                    x=new_x,
                    zmin=np.min(new_z),
                    zmax=np.max(new_z),
                )
                heatmap_fig.update_layout(
                    title=f"{selected_aggregation} of {selected_metric} Heatmap (Normalized Values)",
                    xaxis={"title": "Time", "range": [-0.5, len(new_x) - 0.5]},
                )

        # Update the x-axis of the heatmap based on the price graph zoom
        if price_relayout and timestamps_graph_labels:
            updated = True

            # Get the current x-range from the price graph zoom
            x0 = price_relayout.get("xaxis.range[0]", 0)
            x1 = price_relayout.get("xaxis.range[1]", len(timestamps_graph_labels) - 1)

            # Clamp x0 and x1 to the range of the heatmap
            x0 = max(0, min(x0, len(timestamps_graph_labels) - 1))
            x1 = max(0, min(x1, len(timestamps_graph_labels) - 1))

            # Convert x-range to corresponding timestamps
            t0 = timestamps_graph_labels[int(x0)]
            t1 = timestamps_graph_labels[int(x1)]

            # Split time into hours, minutes, and seconds and convert to seconds
            t0_parts = t0.split(":")
            t1_parts = t1.split(":")
            hour0, minute0, second0 = int(t0_parts[0]), int(t0_parts[1]), int(t0_parts[2].split(".")[0])
            hour1, minute1, second1 = int(t1_parts[0]), int(t1_parts[1]), int(t1_parts[2].split(".")[0])
            t0_sec = hour0 * HOUR_SEC + minute0 * MINUTE_SEC + second0
            t1_sec = hour1 * HOUR_SEC + minute1 * MINUTE_SEC + second1

            # Get heatmap x-axis data
            heatmap_x = heatmap_fig.data[0]["x"]
            heatmap_sec = [int(x.split(":")[0]) * HOUR_SEC + int(x.split(":")[1]) * MINUTE_SEC for x in heatmap_x]

            # Find closest times in heatmap
            t0_index = max(i for i, sec in enumerate(heatmap_sec) if sec <= t0_sec)
            t1_index = min(i for i, sec in enumerate(heatmap_sec) if sec >= t1_sec)
            t0_index = round(max(0, min(t0_index, len(heatmap_sec) - 1)) - 0.5, 3)
            t1_index = round(max(0, min(t1_index, len(heatmap_sec) - 1)) - 0.5, 3)

            # Update the heatmap x-axis range
            heatmap_fig.update_layout(
                xaxis={"range": [t0_index, t1_index]},
            )

        # Update the figure only if it was actually updated, else return no_update
        return heatmap_fig if updated else dash.no_update


    # Run the Dash app
    app.run(host=HOST_ADDRESS, port=PORT, debug=False)


if __name__ == "__main__":
    main()
