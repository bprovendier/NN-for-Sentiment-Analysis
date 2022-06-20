"""Module for data preparation."""

import numpy as np
import yaml
import joblib
import argparse
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import ta


with open("model/params.yaml", "r") as params_file:
    params = yaml.safe_load(params_file)

data_dir = params["data_dir"]
model_dir = params["model_dir"]


def load_data(file_name):
    """
    File data loader.

    Arguments:
        file_name (str): name of the file.

    Returns:
        data (DataFrame): content of the file.
    """
    data = pd.read_csv(Path(data_dir, file_name))
    return data


def save_data(df, file_name):
    """
    Save DataFrame into a csv file.

    Arguments:
        df (dataFrame): content to save.
        file_name (str): name of the file.

    Returns:
        None
    """
    df.to_csv(Path(data_dir, file_name), index=False)
    return None


def clean_data(df):
    """
    Sort by date and drop NA values.

    Arguments:
        df (DataFrame): content to clean.

    Returns:
        df_clean (DataFrame): cleaned DataFrame.
    """
    # sort by date
    df_clean = df.sort_values(by="Time").reset_index(drop=True)
    # drop NaN
    df_clean = df_clean.dropna()

    return df_clean


def create_features(df):
    """
    Feature engineering module.

    Arguments:
        df (DataFrame).

    Returns:
        df (DataFrame): updated DataFrame with new features.
    """
    # add intraday gaps
    df["High_Low_Pct"] = (df.High - df.Low) / df.Low

    # add percentage change
    change = np.diff(df.loc[:, ["Close"]].values, axis=0)
    change = np.append(change, 0)
    df["Change"] = df.Close.pct_change()

    rsi = ta.momentum.RSIIndicator(df["Close"])
    macd = ta.trend.MACD(df["Close"])
    stoch = ta.momentum.StochasticOscillator(df["Close"], df["High"], df["Low"])
    william = ta.momentum.WilliamsRIndicator(df["High"], df["Low"], df["Close"])
    obv = ta.volume.OnBalanceVolumeIndicator(df["Close"], df["Volume"])

    df["RSI"] = rsi.rsi()
    df["MACD"] = macd.macd_diff()
    df["Stochastic_oscillator"] = stoch.stoch()
    df["Williams"] = william.williams_r()
    df["OBV"] = obv.on_balance_volume()

    # drop rows with missing values
    df = df.dropna()

    return df


def split_data(df, train_frac):
    """
    Train, test data splitter.

    Arguments:
        df (DataFrame): full DataFrame.
        train_frac (float): portion of data used for training.

    Returns:
        train_df (DataFrame).
        test_df (DataFrame).
    """
    train_size = int(len(df) * train_frac)
    train_df, test_df = df[:train_size], df[train_size:]

    return train_df, test_df


def rescale_data(df):
    """
    Rescale all features using MinMaxScaler() to same scale, between 0 and 1.

    Arguments:
        df (DataFrame).

    Returns:
        df_scaled (DataFrame).
    """
    scaler = MinMaxScaler()

    scaler = scaler.fit(df)

    df_scaled = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)

    # save trained data scaler
    joblib.dump(scaler, Path(model_dir, "scaler.gz"))

    return df_scaled


def prep_data(df, train_frac, plot_df=False):
    """
    Clean data, create features, split and scale data.

    Arguments:
        df (DataFrame): full DataFrame.
        train_frac (float): portion of data used for training.
        plot_df (bool): create DataFrame used to plot data.

    Returns:
        train_df (DataFrame): train DataFrame.
        val_df (DataFrame): validation DataFrame.
        test_df (DataFrame): test DataFrame.
    """
    print("Starting with data preparation...")
    df_clean = df.copy()
    df_clean = clean_data(df_clean)

    df_clean = df_clean.drop(["Time"], axis=1)
    df_clean = create_features(df_clean)

    # split into train/test datasets
    train_df, test_df = split_data(df_clean, train_frac)
    train_df, val_df = split_data(train_df, 0.8)

    if plot_df:
        save_data(train_df, "plot_df.csv")
    # rescale data
    train_df = rescale_data(train_df)

    scaler = joblib.load(Path(model_dir, "scaler.gz"))
    val_df = pd.DataFrame(
        scaler.transform(val_df), index=val_df.index, columns=val_df.columns
    )
    test_df = pd.DataFrame(
        scaler.transform(test_df), index=test_df.index, columns=test_df.columns
    )

    # save data
    save_data(train_df, "train.csv")
    save_data(test_df, "test.csv")
    print("Completed.")

    return train_df, val_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-name", type=str, default=params["file_name"])
    parser.add_argument("--train-frac", type=float, default=params["train_frac"])
    args = parser.parse_args()

    df = load_data(args.file_name)
    prep_data(df, args.train_frac)
