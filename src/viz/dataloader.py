import os
import numpy as np
import pandas as pd


# Minute in seconds (60 seconds)
MINUTE_SEC = 60
# Hour in seconds (60 minutes)
HOUR_SEC = 60 * MINUTE_SEC
# Day in seconds (24 hours)
DAY_SEC = 24 * HOUR_SEC


def load_data(date, market_segment_id, security, level_depth=1):
    """
    Load the data from the given date, market segment ID, and security
    :param date: Date of the data to load
    :param market_segment_id: Market segment ID of the data to load
    :param security: Security ID of the data to load
    :param level_depth: Level depth of prices to load (default 1 -- Top of the Book)
    :return: DataFrame with the data
    """
    lobster_fp = f"data/{date}_{market_segment_id}_{security}_lobster_augmented.csv"
    data = pd.read_csv(lobster_fp, sep=",")

    # Throw away Ask Price i and Bid Price i if the ith level > level_depth
    i = level_depth + 1
    while True:
        if f"Ask Price {i}" not in data.columns and f"Bid Price {i}" not in data.columns:
            break
        data = data.drop(columns=[f"Ask Price {i}", f"Bid Price {i}"], errors="ignore")
        i += 1

    return data


def load_all_data(level_depth=1):
    """
    Load all data from the data folder
    :param level_depth: Level depth of prices to load (default 1 -- Top of the Book)
    :return: List of DataFrames with the data and a list of names for each DataFrame
    """
    data = []
    names = []

    for file in os.listdir("data"):
        if file.endswith("_lobster_augmented.csv"):
            date, market_segment_id, security = file.split("_")[:3]
            names.append(f"{date}_{market_segment_id}_{security}")
            data.append(load_data(date, market_segment_id, security, level_depth=level_depth))

    return data, names


def aggregate_data(all_data, metric="Ask Price 1", aggregation=np.mean, time_window=3600):
    """
    Aggregate the data in the given time window
    :param all_data: Loaded data in the form of a list of DataFrames
    :param metric: Metric to aggregate
    :param aggregation: Aggregation function to use
    :param time_window: Time window for aggregation (in seconds)
    :return: Aggregated data in the form of a list of numpy arrays
    """
    aggregated_data = []  # Result list

    for data in all_data:
        tmp_agg = []  # Temporary list to store the aggregated data for each time window

        timestamps = pd.to_datetime(data["Time"].values, unit="ns")
        timestamps_series = pd.Series(timestamps)
        seconds_since_midnight = (timestamps_series - timestamps_series.dt.normalize()).dt.total_seconds()

        for i in range(0, DAY_SEC, time_window):
            start_time = i
            end_time = i + time_window

            # Get the data in the time window
            data_in_window = data[(seconds_since_midnight >= start_time) & (seconds_since_midnight < end_time)]
            data_in_window = data_in_window[metric].dropna()

            # Aggregate the metric in the time window
            if data_in_window.empty:  # If there is no data in the time window, append NaN
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