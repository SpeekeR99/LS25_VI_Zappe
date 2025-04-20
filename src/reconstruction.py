# ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
# | This code is NOT created by me                                                                                     |
# | I have been using this code, I have enhanced it a little bit, but it is not a main part of my thesis work          |
# | This code was created as a part of another thesis work                                                             |
# | https://portal.zcu.cz/StagPortletsJSR168/CleanUrl?urlid=prohlizeni-prace-detail&praceIdno=97020                    |
# |                                                                                                    - Dominik Zappe |
# └────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

import pandas as pd
import numpy as np
import datetime

import plotly_resampler
from dash import Dash, dcc, html, Input, Output, State, dash_table, ctx
import dash_bootstrap_components as dbc
from flask import Flask, request, jsonify
import plotly.graph_objects as go
from plotly_resampler import FigureResampler
import plotly.express as px

import io
import PIL


old_click_time = 0


class Config:
    """
    Configuration settings and useful functions
    """
    # Server config
    addr = "127.0.0.1"
    port = 1234

    # Data config
    path = "data/"
    df_cols = ["Price", "DisplayQty", "Q", "od", "do", "Trans", "Prio"]
    df_cols_type = {"Price": np.float64, "DisplayQty": np.int64, "Q": np.int64, "od": np.int64, "do": np.int64, "Trans": np.int64, "Prio": str}
    delim = ","

    @staticmethod
    def calc_nansec_from_time(time: str) -> int:
        """
        :param time: time in format hh:mm:ss.nnnnnnnnn or hh:mm:ss
        Returns number of nanoseconds from time in format hh:mm:ss.nnnnnnnnn or hh:mm:ss
        """
        time = time.split(":")
        nansec = time[-1].split(".")
        return int(int(time[0]) * 36e11 + int(time[1]) * 6e10 + int(nansec[0]) * 1e9 + (int(nansec[1].ljust(9, "0")) if len(nansec) == 2 else 0))

    @staticmethod
    def calc_time_from_nansec(nansecs: int) -> str:
        """
        :param nansecs: number of nanoseconds
        Returns time in format hh:mm:ss.nnnnnnnnn from given nanoseconds
        """
        s = nansecs // 1e9
        delta = datetime.timedelta(seconds=s)
        ns = str(int(nansecs % 1e9)).zfill(9)
        datetime_obj = (datetime.datetime.min + delta).time()
        time_formatted = datetime_obj.strftime('%H:%M:%S')
        time = time_formatted + "." + ns
        return time


class OB:
    """
    Order book
    """
    def __init__(self, instrument, security, date) -> None:
        """
        Constructor
        :param instrument: Instrument
        :param security: Security ID
        :param date: Date
        """
        self.__instrument = instrument
        self.__security = security
        self.__date = date
        self.__data = pd.DataFrame()
        self.__min_timestamp = 0  # First timestamp of day
        self.__timestamp = 0
        self.__bookA = pd.DataFrame()
        self.__bookB = pd.DataFrame()
        self.__executes = {}
        self.__changed = False  # True if new data source valid, False otherwise (used in API func)
        self.change_data_df(self.__instrument, self.__security, self.__date)
    
    def get_instrument(self):
        return self.__instrument

    def get_security(self):
        return self.__security

    def get_date(self):
        return self.__date

    def get_timestamp(self):
        return self.__timestamp

    def get_bookA(self):
        return self.__bookA

    def get_bookB(self):
        return self.__bookB

    def get_executes(self):
        return self.__executes

    def get_changed(self):
        return self.__changed
    
    def set_timestamp(self, timestamp):
        self.__timestamp = timestamp
        
    def change_data_df(self, instrument, security, date) -> None:
        """
        Load new dataframe for given instrument, securityID, date and calculate initial OB
        Keeps previous state if new file doesnt exist
        :param instrument: Instrument
        :param security: Security ID
        :param date: Date
        """
        try:
            temp = pd.read_csv(f"{Config.path}{date}-{instrument}-{security}-ob.csv", delimiter=Config.delim, usecols=Config.df_cols, dtype=Config.df_cols_type)
            if not temp.empty:
                self.__data = temp
                self.__instrument = instrument
                self.__security = security
                self.__date = date
                self.__timestamp = int(self.__data.iloc[0]["Trans"])  # Set inital time (first time of day)
                self.__min_timestamp = self.__timestamp
                self.__changed = True
        except Exception as e:
            print(e)
            self.__changed = False
    
    def calc_order_book_state(self, seq) -> None:
        """
        Calculate order book for given time sequence
        :param seq: [id1, id2], id2 - index of row with timestamp to which the OB should be reconstructed
        """
        # Select all order executes in time sequence
        executes = self.__data.loc[seq[0]:seq[1]-2]
        executes = executes.loc[executes["Trans"] < 0, ["Price", "Q", "Prio"]]
        if not executes.empty:
            executes.rename(columns={'Q': 'Qty'}, inplace=True)
            executes['Type'] = executes['Price'].apply(lambda x: 'B' if x > 0 else 'A')
            executes['Qty'] = executes['Qty'] * -1
            executes.loc[executes['Type'] == 'A', 'Price'] *= -1

        od = seq[1]

        # Select rows valid at od
        ind = (self.__data['od'] <= od) & (self.__data['do'] > od)
        p = self.__data.loc[ind, 'Price'].values
        q = self.__data.loc[ind, 'DisplayQty'].values
        
        # Sort by price
        sorted_indices = np.argsort(p)
        p = p[sorted_indices]
        q = q[sorted_indices]
        
        # Separate bid and ask orders
        indB = (q > 0) & (p > 0)
        bookB = pd.DataFrame({'Price': p[indB], 'Qty': q[indB]})
        bookB = bookB.sort_values("Price", ascending=False)
        bookB.reset_index(drop=True, inplace=True)

        indA = (q > 0) & (p < 0)
        bookA = pd.DataFrame({'Price': -p[indA], 'Qty': q[indA]})
        bookA = bookA.sort_values("Price", ascending=True)
        bookA.reset_index(drop=True, inplace=True)

        self.__bookA = bookA
        self.__bookB = bookB
        self.__executes = executes.to_dict("records")

    def get_time_seq(self, time):
        """
        Returns time sequence and timestamp for nearest smaller timestamp to given time in dataframe.
        Sequence: [id1,id2], id2 - index of row with timestamp to which the OB should be reconstructed,
        id1 - index of row with first positive timestamp before id2. Timestamp - timestamp of row with id2
        :param time: timestamp
        """
        seq, tstamp = [], 0
        if time < self.__min_timestamp:
            time = self.__min_timestamp

        temp = (self.__data['Trans'] <= time) & (self.__data['Trans'] > 0)
        rows = self.__data.loc[temp]
        rows = rows[(~rows.duplicated(["Trans"], keep="last"))].tail(2)

        if len(rows) == 2:
            seq.append(rows.iloc[[0]]["od"].values[0])
            seq.append(rows.iloc[[1]]["od"].values[0])
            tstamp = rows.iloc[[1]]["Trans"].values[0]
        else:
            od = rows.iloc[[0]]["od"].values[0]
            seq = [od, od]
            tstamp = rows.iloc[[0]]["Trans"].values[0]

        return seq, tstamp

    def get_prev_time_seq(self, time):
        """
        Returns time sequence and timestamp for nearest smaller timestamp to currently selected timestamp in dataframe
        Sequence: [id1,id2], id2 - index of row with timestamp to which the OB should be reconstructed, 
        id1 - index of row with first positive timestamp before id2. Timestamp - timestamp of row with id2
        :param time: timestamp
        """
        seq, tstamp = [], 0
        temp = (self.__data['Trans'] <= time) & (self.__data['Trans'] > 0)
        rows = self.__data.loc[temp]
        rows = rows[(~rows.duplicated(["Trans"], keep="last"))]

        if len(rows) <= 2:
            od = rows.iloc[[0]]["od"].values[0]
            seq = [od, od]
            tstamp = rows.iloc[[0]]["Trans"].values[0]
        else:    
            rows = rows.iloc[-3:-1]
            seq.append(rows.iloc[[0]]["od"].values[0])
            seq.append(rows.iloc[[1]]["od"].values[0])
            tstamp = rows.iloc[[1]]["Trans"].values[0]

        return seq, tstamp

    def get_next_time_seq(self, time):
        """
        Returns time sequence and timestamp for nearest bigger timestamp to currently selected timestamp in dataframe
        Sequence: [id1,id2], id2 - index of row with timestamp to which the OB should be reconstructed, 
        id1 - index of row with first positive timestamp before id2. Timestamp - timestamp of row with id2
        :param time: timestamp
        """
        seq, tstamp = [], 0
        temp = (self.__data['Trans'] <= time) & (self.__data['Trans'] > 0)
        start = self.__data.loc[temp].tail(1)
        seq.append(start["od"].values[0])

        temp = (self.__data['Trans'] > time) & (self.__data['Trans'] > 0)
        rows = self.__data.loc[temp]
        rows = rows[(~rows.duplicated(["Trans"], keep="last"))]

        if rows.empty:
            seq.append(seq[0])
            tstamp = start["Trans"].values[0]
        else:
            stop = rows.head(1)
            seq.append(stop["od"].values[0])
            tstamp = stop["Trans"].values[0]

        return seq, tstamp


class Lobster:
    """
    Lobster data
    """
    def __init__(self, path_to_file):
        """
        Constructor
        :param path_to_file: Path to lobster csv file
        """
        self.filepath = path_to_file

    def load_data(self):
        """
        Load data from csv file
        """
        data = pd.read_csv(self.filepath)

        return data


def imbalance_index(asks, bids, alpha=0.5, level=3):
    """
    Calculate imbalance index for a given orderbook.
    :param asks: list of ask sizes (volumes)
    :param bids: list of bid sizes (volumes)
    :param alpha: parameter for imbalance index
    :param level: number of levels to consider
    :return: imbalance index
    """
    assert len(asks) >= level and len(bids) >= level, "Not enough levels in orderbook"
    assert alpha > 0, "Alpha must be positive"
    assert level > 0, "Level must be positive"

    # Calculate imbalance index
    V_bt = sum(bids[:level] * np.exp(-alpha * np.arange(0, level)))
    V_at = sum(asks[:level] * np.exp(-alpha * np.arange(0, level)))
    return (V_bt - V_at) / (V_bt + V_at)


def imbalance_index_vectorized(asks, bids, alpha=0.5, level=3):
    """
    Calculate imbalance index for a given orderbook.
    :param asks: numpy matrix of ask sizes (volumes)
    :param bids: numpy matrix of bid sizes (volumes)
    :param alpha: parameter for imbalance index
    :param level: number of levels to consider
    :return: imbalance index
    """
    assert asks.shape[1] >= level and bids.shape[1] >= level, "Not enough levels in orderbook"
    assert alpha > 0, "Alpha must be positive"
    assert level > 0, "Level must be positive"

    # Calculate imbalance index
    V_bt = np.sum(bids[:, :level] * np.exp(-alpha * np.arange(0, level)), axis=1)
    V_at = np.sum(asks[:, :level] * np.exp(-alpha * np.arange(0, level)), axis=1)
    return (V_bt - V_at) / (V_bt + V_at)


def price_graph(date, market_segment_id, security, level_depth=3):
    """
    Create price graph
    :return: figure
    """
    lobster_fp = f"{Config.path}{date}_{market_segment_id}_{security}_lobster_augmented.csv"
    # lobster_fp = "data/lobster_test.csv"
    lobster = Lobster(lobster_fp)
    data = lobster.load_data()
    # ? why is date thrown away ?
    timestamps = data["Time"].tolist()
    temp = [Config.calc_time_from_nansec(t) for t in timestamps]
    timestamps = [Config.calc_nansec_from_time(t) for t in temp]
    ask_prices = []
    bid_prices = []
    for i in range(level_depth):
        ask_prices.append(data[f"Ask Price {i+1}"])
        bid_prices.append(data[f"Bid Price {i+1}"])

    return timestamps, ask_prices, bid_prices, data


def interpolate_color(color1, color2, factor):
    return tuple(int(color1[i] + factor * (color2[i] - color1[i])) for i in range(3))


def get_frequency_of_incoming_actions(timestamps, timestamp, time=300):
    """
    Returns the frequency of incoming actions (messages) in the last *time* seconds for given timestamp
    :param orderbook: Array of timestamps (in nano seconds)
    :param timestamp: Timestamp (in nano seconds)
    :param time: Time window in seconds (default value is 300)
    :return: Frequency of incoming actions (messages) for given timestamp
    """
    # Convert time to nanoseconds
    time_ns = time * 1e9

    timestamps = np.array(timestamps)

    # Calculate frequency of incoming actions
    freq = 0
    for ts in timestamps:
        if timestamp - time_ns < ts:
            freq += 1

        if ts >= timestamp:
            break

    return freq / time


def get_frequency_of_all_incoming_actions(timestamps, time=300):
    """
    Returns the frequency of incoming actions (messages) in the last *time* seconds for all timestamps
    :param timestamps: Array of timestamps (in nano seconds)
    :param time: Time window in seconds (default value is 300)
    :return: List of frequencies of incoming actions (messages) for all timestamps
    """
    # Convert time to nanoseconds
    time_ns = time * 1e9

    # Convert timestamps to a numpy array
    timestamps = np.array(timestamps)

    # Initialize two pointers at the end of the timestamps array
    start, end = len(timestamps) - 1, len(timestamps) - 1

    # Initialize an array to hold the frequencies
    freqs = np.zeros_like(timestamps)

    # Calculate the frequency for each timestamp
    for start in range(len(timestamps)-1, -1, -1):
        while end >= 0 and timestamps[start] - timestamps[end] <= time_ns:
            end -= 1
        freqs[start] = start - end

    return freqs / time


def main():
    """
    Main function
    """
    # Instrument, security, date
    instrument = "FGBL"
    security = "4128839"
    market_segment_id = "688"
    date = "20191202"

    # Set default OB for dash
    dash_OB = OB(instrument, security, date)
    api_OB = None

    # Price graph
    timestamps, ask_prices, bid_prices, lobster_data = price_graph(date, market_segment_id, security, level_depth=5)

    level_depth = 30
    ask_columns = [f'Ask Volume {i}' for i in range(1, level_depth+1)]
    bid_columns = [f'Bid Volume {i}' for i in range(1, level_depth+1)]

    lobster_data_matrix = lobster_data[ask_columns + bid_columns].values
    if "Imbalance Index" in lobster_data.columns:
        imbalance_indices = lobster_data["Imbalance Index"].values
    else:
        imbalance_indices = imbalance_index_vectorized(lobster_data_matrix[:, :level_depth], lobster_data_matrix[:, level_depth:], alpha=0.5, level=level_depth)

    time_window = 300
    if "Frequency of Incoming Messages" in lobster_data.columns:
        freqs = lobster_data["Frequency of Incoming Messages"].values
    else:
        freqs = get_frequency_of_all_incoming_actions(timestamps, time=time_window)

    if "Cancellations Rate" in lobster_data.columns:
        cancels = lobster_data["Cancellations Rate"].values
    else:
        cancels = np.zeros_like(timestamps)

    # Convert timestamps to HH:MM:SS format
    timestamps_graph_labels = [Config.calc_time_from_nansec(int(ts)) for ts in timestamps]
    # Convert to 0 - n
    timestamps_graph = list(range(len(timestamps_graph_labels)))
    # Tick every 10000 timestamps
    tickvals = list(range(0, len(timestamps), 10000))
    # tickvals = list(range(0, len(timestamps), 25000))
    ticklabels = [timestamps_graph_labels[i] for i in tickvals]
    # Go from HH:MM:SS.nnnnnn to truly HH:MM:SS
    ticklabels = [ts[:8] for ts in ticklabels]

    # Init Flask server with dash app
    server = Flask(__name__)
    app = Dash(__name__, external_stylesheets=[dbc.icons.FONT_AWESOME], server=server)

    # Dash app layout
    # ------------------------------
    dash_figure = {}
    dash_book = {}
    dash_time = Config.calc_time_from_nansec(dash_OB.get_timestamp())

    start_time_gif_export = "00:00:00"
    end_time_gif_export = "00:00:00"

    # Arrow icons
    fa_ar = html.I(className="fa-solid fa-arrow-right")
    fa_al = html.I(className="fa-solid fa-arrow-left")

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
            hf_marker_color=f'rgb' + str(interpolate_color((230, 31, 7), (255, 255, 255), (i-1)/len(ask_prices)))
        )

    for i, bid_price in enumerate(bid_prices, 1):
        price_graph_fig.add_trace(
            go.Scattergl(
                name=f'Bid {i}',
                yaxis="y1"
            ),
            hf_x=timestamps_graph,
            hf_y=bid_price,
            hf_marker_color=f'rgb' + str(interpolate_color((94, 163, 54), (255, 255, 255), (i-1)/len(bid_prices)))
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

    price_graph_fig.add_trace(
        go.Scattergl(
            name='Current time',
            mode='lines',
            yaxis='y1'
        ),
        hf_x=[timestamps_graph[0], timestamps_graph[0]],
        hf_y=[min(ask_prices[0]), max(bid_prices[0])],
        hf_marker_color="rgb(0, 0, 0)", hf_marker_size=2
    )

    price_graph_fig.update_layout(
        title='Price Graph',
        xaxis={'title': 'Timestamp', "tickmode": "array", "tickvals": tickvals, "ticktext": ticklabels},
        yaxis={'title': 'Price', 'side': 'left'},
        yaxis2={'title': 'Imbalance index', 'side': 'right', 'overlaying': 'y', "anchor": "free", "autoshift": True, "range": [-1, 1]},
        yaxis3={'title': 'Incoming messages (per sec)', 'side': 'right', 'overlaying': 'y', "anchor": "free", "autoshift": True},
        yaxis4={'title': 'Cancellations rate', 'side': 'right', 'overlaying': 'y', "anchor": "free", "autoshift": True},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        # ----- For thesis images export -----
        # title={"text": '', "font": {"size": 48}},
        # xaxis={'title': {'text': 'Timestamp', 'font': {'size': 36}}, "tickmode": "array", "tickvals": tickvals, "ticktext": ticklabels, "tickfont": {"size": 32}},
        # yaxis={'title': {'text': 'Price', 'font': {'size': 36}}, 'side': 'left', "tickfont": {"size": 32}},
        # yaxis2={'title': {'text': 'Imbalance index', 'font': {'size': 36}}, 'side': 'right', 'overlaying': 'y', "anchor": "free", "autoshift": True, "range": [-1, 1], "tickfont": {"size": 32}},
        # yaxis3={'title': {'text': 'Incoming messages (per sec)', 'font': {'size': 36}}, 'side': 'right', 'overlaying': 'y', "anchor": "free", "autoshift": True, "tickfont": {"size": 32}},
        # yaxis4={'title': {'text': 'Cancellations rate', 'font': {'size': 36}}, 'side': 'right', 'overlaying': 'y', "anchor": "free", "autoshift": True, "tickfont": {"size": 32}},
        # legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1, "font": {"size": 32}},
        # ----- For thesis images export -----
        clickmode='event+select',
        hovermode="x unified"
    )

    price_graph_fig.register_update_graph_callback(app=app, graph_id="price_graph")

    app.layout = html.Div(children=[
        html.Div(children=[
            dcc.Input(
                id="input_instrument",
                type="text",
                value=dash_OB.get_instrument(),
                placeholder="Instrument",
                debounce=True,
            ),
            dcc.Input(
                id="input_security",
                type="text",
                value=dash_OB.get_security(),
                placeholder="SecurityID",
                debounce=True,
            ),
            dcc.DatePickerSingle(
                id="input_date",
                date=dash_OB.get_date(),
                display_format='YYYY/MM/DD'
            ),
            dbc.Button("Select", id="select_data"),
        ]),

        dbc.Button([fa_al, ""], id="prev_ordb"),
        dcc.Input(
                id="input_time",
                type="text",
                value=dash_time,
                placeholder="Select time",
                debounce=True,
        ),
        
        dbc.Button("Show", id="curr_ordb"),
        dbc.Button([fa_ar, ""], id="next_ordb"),

        html.Div(id='opt_container'),

        html.Div(children=[
            dcc.Loading(
            dcc.Graph(
                id="price_graph",
                figure=price_graph_fig,
                config={
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': 'price_graph',
                        "width": 1920,
                        "height": 1080,
                        'scale': 3
                    }
                }
            ), type="circle")
        ], style={
            "display": "block",
            'margin-top': '2vh',
            "justify-content": "center",
            "align-items": "center",
            "height": "50vh",
            'border': '2px solid black'
        }),
        html.Div(children=[
            dcc.Loading(
            dcc.Graph(
                id="book_graph",
                figure={
                    'data': [],
                    'layout': {
                        'title': 'Order Book',
                        "autosize": True,
                    }
                }
            ), type="circle")
        ], style={
            "display": "block",
            'margin-top': '2vh',
            "justify-content": "center",
            "align-items": "center",
            "height": "50vh",
            'border': '2px solid black'
        }),
        html.Div(children=[
            dcc.Input(
                id="start_time_gif_export_id",
                type="text",
                value=start_time_gif_export,
                placeholder="Select start time",
                debounce=True,
            ),
            dcc.Input(
                id="end_time_gif_export_id",
                type="text",
                value=end_time_gif_export,
                placeholder="Select end time",
                debounce=True,
            ),
            dbc.Button("Export to GIF", id="export_to_gif_button"),
            dcc.Loading(
                dcc.Graph(
                    id="book_graph_animated",
                    figure={
                        'data': [],
                        'layout': {
                            'title': 'Order Book Animated'
                        }
                    }
                ), type="circle")
        ], style={
            "display": "block",
            'margin-top': '2vh',
            "justify-content": "center",
            "align-items": "center",
            'border': '2px solid black'
        }),
        html.Div(children=[
            dash_table.DataTable(
                id='order_book',
                columns=[{"name": "BQty", "id": "BQty"},
                         {"name": "Bid", "id": "Bid"},
                         {"name": "Ask", "id": "Ask"},
                         {"name": "AQty", "id": "AQty"}],
                style_table={'height': '400px', 'overflowY': 'auto'},
                style_cell={'width': '100px'},
                style_data_conditional=[
                    {
                        'if': {'column_id': 'Bid'},
                        "color": "rgb(94, 163, 54)"
                    },
                    {
                        'if': {'column_id': 'Ask'},
                        "color": "rgb(230, 31, 7)"
                    }
                ]
            )
        ], style={
            'width': '65%',
            'height': '500px',
            'display': 'inline-block',
            'verticalAlign': 'top',
            'overflowX': 'hidden',
            'border': '2px solid black'
        }),
        html.Div(children=[
            html.H3('Trades'),
            dash_table.DataTable(
                id='trades_book',
                columns=[{"name": "Price", "id": "Price"},
                         {"name": "Qty", "id": "Qty"},
                         {"name": "Prio", "id": "Prio"}],
                style_table={'height': '400px', 'overflowY': 'auto'},
                style_cell={'width': '100px'},
                style_data_conditional=[
                    {
                        'if': {'column_id': 'Price', 'filter_query': '{Type} eq "B"'},
                        "color": "rgb(94, 163, 54)"
                    },
                    {
                        'if': {'column_id': 'Price', 'filter_query': '{Type} eq "A"'},
                        "color": "rgb(230, 31, 7)"
                    }
                ]
            )
        ], style={
            'width': '33%',
            'height': '500px',
            'margin-left': '2vh',
            'display': 'inline-block',
            'verticalAlign': 'top',
            'overflowX': 'hidden',
            'border': '2px solid black'
        }),
    ])
    # ------------------------------

    # Dash app func
    # ------------------------------
    @app.callback(
        Output('input_instrument', 'value'),
        Output('input_security', 'value'),
        Output('input_date', 'date'),
        Output('input_time', 'value'),
        Output('order_book', 'data'),
        Output('book_graph', 'figure'),
        Output("price_graph", "figure"),
        Output('trades_book', 'data'),
        State('input_instrument', 'value'),
        State('input_security', 'value'),
        State('input_date', 'date'),
        State('input_time', 'value'),
        State("book_graph", "figure"),
        State("price_graph", "figure"),
        Input('select_data', 'n_clicks'),
        Input('prev_ordb', 'n_clicks'),
        Input('curr_ordb', 'n_clicks'),
        Input('next_ordb', 'n_clicks'),
        Input("price_graph", "clickData"),
    )
    def update_orderbook(instrument, security, date, input_time, current_book_graph, current_price_graph, a, b, c, d, price_graph_click):
        """
        Handle dash app updates
        :param instrument: Instrument
        :param security: Security ID
        :param date: Date
        :param input_time: Time
        """
        nonlocal dash_time, dash_book, dash_figure, dash_OB
        global old_click_time

        if price_graph_click is not None and price_graph_click["points"][0]["x"] != old_click_time:
            input_time = timestamps_graph_labels[price_graph_click["points"][0]["x"]]
            old_click_time = price_graph_click["points"][0]["x"]

        nansec_time = Config.calc_nansec_from_time(input_time)

        # Triggered button
        trigg_btn = ctx.triggered_id

        # Select new data source
        if trigg_btn == "select_data":
            date = date.replace("-", "")
            # Change only if new values
            if instrument != dash_OB.get_instrument() or security != dash_OB.get_security() or date != dash_OB.get_date():
                # Downlaod f"{date}-{instrument}-{security}-ob.csv" and f"{date}-{instrument}-{security}-lobster.csv"


                nonlocal timestamps, ask_prices, bid_prices, lobster_data, imbalance_indices, freqs, time_window, cancels
                dash_OB.change_data_df(instrument, security, date)
                dash_time = Config.calc_time_from_nansec(dash_OB.get_timestamp())
                seq, tstamp = dash_OB.get_time_seq(dash_OB.get_timestamp())
                timestamps, ask_prices, bid_prices, lobster_data = price_graph(instrument, security, date, level_depth=1)
                lobster_data_matrix = lobster_data[ask_columns + bid_columns].values
                if "Imbalance Index" in lobster_data.columns:
                    imbalance_indices = lobster_data["Imbalance Index"].values
                else:
                    imbalance_indices = imbalance_index_vectorized(lobster_data_matrix[:, :level_depth], lobster_data_matrix[:, level_depth:], alpha=0.5, level=level_depth)
                if "Frequency of Incoming Messages" in lobster_data.columns:
                    freqs = lobster_data["Frequency of Incoming Messages"].values
                else:
                    freqs = get_frequency_of_all_incoming_actions(timestamps, time=time_window)
                if "Cancellations Rate" in lobster_data.columns:
                    cancels = lobster_data["Cancellations Rate"].values
                else:
                    cancels = np.zeros_like(timestamps)
            else:
                return dash_OB.get_instrument(), dash_OB.get_security(), dash_OB.get_date(), dash_time, dash_book, dash_figure, dash_OB.get_executes()

        # Show one timestamp before current timestamp
        elif trigg_btn == "prev_ordb":
            seq, tstamp = dash_OB.get_prev_time_seq(nansec_time)
        
        # Show one timestamp after current timestamp
        elif trigg_btn == "next_ordb":
            seq, tstamp = dash_OB.get_next_time_seq(nansec_time)

        # Show orderbook in given time
        else:
            seq, tstamp = dash_OB.get_time_seq(nansec_time)

        dash_OB.set_timestamp(tstamp)
        dash_time = Config.calc_time_from_nansec(tstamp)
        
        # Calculate and modify order book
        dash_OB.calc_order_book_state(seq)
        A = dash_OB.get_bookA()
        A.rename(columns={'Price': 'Ask', "Qty": "AQty"}, inplace=True)
        B = dash_OB.get_bookB()
        B.rename(columns={'Price': 'Bid', "Qty": "BQty"}, inplace=True)
        
        # Select data for graph figure
        bid_x = list(B['Bid'])
        bid_y = list(B['BQty'])
        ask_x = list(A['Ask'])
        ask_y = list(A['AQty'])

        # Cut out outliers
        ratio = 0.25
        bid_x_len_10 = int(len(bid_x) * ratio)
        ask_x_len_10 = int(len(ask_x) * (1 - ratio))
        bid_x = bid_x[:-bid_x_len_10]
        bid_y = bid_y[:-bid_x_len_10]
        ask_x = ask_x[:ask_x_len_10]
        ask_y = ask_y[:ask_x_len_10]

        # Calculate imbalance index
        imbal_index = imbalance_index(ask_y, bid_y, alpha=0.5, level=30)

        # Calculate frequency of incoming actions
        time_window = 300
        freq = get_frequency_of_incoming_actions(timestamps, tstamp, time=time_window)

        # Save current zoom in the graph (id "book_graph")
        x_range = [bid_x[-1], ask_x[-1]]
        if "xaxis" in current_book_graph["layout"]:
            x_range = current_book_graph["layout"]["xaxis"]["range"]

        # Update graph figure
        dash_figure = {
            "data": [
                {'x': bid_x, 'y': bid_y, 'type': 'bar', 'name': 'Bid', 'marker': {'color': 'rgb(94, 163, 54)'}},
                {'x': ask_x, 'y': ask_y, 'type': 'bar', 'name': 'Ask', 'marker': {'color': 'rgb(230, 31, 7)'}},
            ],
            'layout': {
                    'title': f"Order Book at {dash_time}<br>Imbalance index = {imbal_index}<br>Incoming messages (per sec) = {freq}",
                    'xaxis': {'title': 'Price', "range": x_range},
                    'yaxis': {'title': 'Quantity'},
                    'autoscale': True
            }
        }

        # Update current time line in price graph (find "data" with name "Current time")
        for i, data in enumerate(current_price_graph["data"]):
            if data["name"] == "Current time":
                x_axis_current_number = 0
                try:
                    x_axis_current_number = timestamps_graph_labels.index(dash_time)
                except Exception:
                    pass
                current_price_graph["data"][i]["x"] = [x_axis_current_number, x_axis_current_number]
                break

        dash_book = pd.concat([B, A], axis=1).to_dict("records")
        return dash_OB.get_instrument(), dash_OB.get_security(), dash_OB.get_date(), dash_time, dash_book, dash_figure, current_price_graph, dash_OB.get_executes()

    @app.callback(
        Output("book_graph_animated", "figure"),
        Input("export_to_gif_button", "n_clicks"),
        State("start_time_gif_export_id", "value"),
        State("end_time_gif_export_id", "value"),
        prevent_initial_call=True
    )
    def export_to_gif(a, start_time, end_time):
        """
        Iterate over order book from start_time until end_time
        :param a: Click on button, unused
        :param start_time: start time
        :param end_time: end time
        """
        nonlocal dash_OB

        # Get timestamps
        start_time = Config.calc_nansec_from_time(start_time)
        end_time = Config.calc_nansec_from_time(end_time)

        # Copy current OB
        temp_OB = OB(dash_OB.get_instrument(), dash_OB.get_security(), dash_OB.get_date())

        Prices = []
        Qties = []
        Category = []
        Timestamps = []

        print("Iterating over order book...")

        # Iterate over timestamps
        while start_time < end_time:
            seq, tstamp = temp_OB.get_time_seq(start_time)
            temp_OB.calc_order_book_state(seq)
            A = temp_OB.get_bookA()
            B = temp_OB.get_bookB()

            time = Config.calc_time_from_nansec(tstamp)
            # Ratio for cutting out outliers
            ratio = 0.25

            Bids = list(B['Price'])
            BQties = list(B['Qty'])

            Bids_len_10 = int(len(Bids) * ratio)
            Bids = Bids[:-Bids_len_10]
            BQties = BQties[:-Bids_len_10]

            Prices.extend(Bids)
            Qties.extend(BQties)
            for i in range(len(BQties)):
                Category.append("Bid")
                Timestamps.append(time)

            Asks = list(A['Price'])
            AQties = list(A['Qty'])

            Asks_len_10 = int(len(Asks) * (1 - ratio))
            Asks = Asks[:Asks_len_10]
            AQties = AQties[:Asks_len_10]

            Prices.extend(Asks)
            Qties.extend(AQties)
            for i in range(len(AQties)):
                Category.append("Ask")
                Timestamps.append(time)

            _, start_time = temp_OB.get_next_time_seq(start_time)

        # Create dataframe for animated order book
        animation_df = pd.DataFrame({
            "Price": Prices,
            "Quantity": Qties,
            "Category": Category,
            "Timestamp": Timestamps
        })

        # Create figure for animated order book (use df["Timestamp"] for animation)
        # Same visualization as "book_graph", but with animation slider over "Timestamp"
        fig = px.bar(animation_df, x="Price", y="Quantity", animation_frame="Timestamp", color="Category",
                     color_discrete_map={"Bid": "rgb(94, 163, 54)", "Ask": "rgb(230, 31, 7)"}, title="Order Book Animated")
        fig.update_layout(
            xaxis={'title': 'Price', "range": [min(Prices), max(Prices)]},
            yaxis={'title': 'Quantity'},
            autosize=True,
        )

        # Make figure bigger for gif
        old_width = fig.layout.width
        old_height = fig.layout.height
        fig.update_layout(
            width=1200,
            height=800,
            autosize=True
        )

        print("Figure created\nIterating over frames...")

        # Export as GIF part
        frames = []
        for s, fr in enumerate(fig.frames):
            # set main traces to appropriate traces within plotly frame
            fig.update(data=fr.data)
            # move slider to correct place
            fig.layout.sliders[0].update(active=s)
            # generate image of current state
            frames.append(PIL.Image.open(io.BytesIO(fig.to_image(format="png"))))

        gif_name = f"OB_{dash_OB.get_instrument()}_{dash_OB.get_security()}_{dash_OB.get_date()}_from_{start_time}_to_{end_time}.gif"
        fps = 20
        frames[0].save(
            gif_name,
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            duration=1000 / fps,
            loop=0,
        )

        print("GIF exported")

        # Reset figure to initial state
        fig.update(data=fig.frames[0].data)
        fig.layout.sliders[0].update(active=0)
        fig.update_layout(
            width=old_width,
            height=old_height,
            autosize=True
        )

        return fig

    # ------------------------------

    # Flask api func
    # ------------------------------
    @server.route('/api/', methods=['GET'])
    def api_get_data():
        """
        Handle API requests
        """
        nonlocal api_OB

        # Get the request params
        params = request.args
        req_inst = params.get("Instrument")
        req_sec = params.get("Security")
        req_date = params.get("Date")
        req_time = params.get("Time")
        if req_inst is None or req_sec is None or req_date is None or req_time is None:
            return {}

        # If OB not initialized at all
        if api_OB is None:
            api_OB = OB(req_inst, req_sec, req_date)
        
        # OB exists but for different params
        elif req_inst != api_OB.get_instrument() or req_sec != api_OB.get_security() or req_date != api_OB.get_date():
            api_OB.change_data_df(req_inst, req_sec, req_date)
        
        # False request params
        if not api_OB.get_changed():
            return {}
        
        # Valid OB
        nansec_time = Config.calc_nansec_from_time(req_time)
        seq, tstamp = dash_OB.get_time_seq(nansec_time)

        api_OB.set_timestamp(tstamp)
        
        # Calculate and send order book
        api_OB.calc_order_book_state(seq)
        A = api_OB.get_bookA().to_dict("records")
        B = api_OB.get_bookB().to_dict("records")
        executes = api_OB.get_executes()
        payload = {"Time": str(tstamp), "Asks": A, "Bids": B, "Executes": executes}

        return jsonify(payload)
    # ------------------------------

    app.run_server(host=Config.addr, port=Config.port, debug=True)


if __name__ == '__main__':
    main()
