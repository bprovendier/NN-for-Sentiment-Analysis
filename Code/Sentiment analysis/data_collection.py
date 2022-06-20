"""Price data collection and data merger."""
import pandas as pd
from binance import Client
import numpy as np
from datetime import datetime

client = Client()
info = client.get_exchange_info()

symbols = ["BTCUSDT"]


def get_data_timeframe(symbol):
    """
    Get dataframe containing price of a coin.

    Arguments:
        symbol (list): symbols to extract information.

    Returns:
        frame (DataFrame): price information dataframe.
    """
    frame = pd.DataFrame(
        client.get_historical_klines(symbol, "15m", "4 months ago UTC")
    )
    if len(frame) > 0:
        frame = frame.iloc[:, :6]
        frame.columns = ["Time", "Open", "High", "Low", "Close", "Volume"]
        frame = frame.astype("float")
        temp = pd.to_datetime(frame["Time"], unit="ms")
        frame["Time"] = pd.Series(pd.to_datetime(temp.values, unit="s"), dtype="string")
        return frame


df = pd.DataFrame()
for coin in symbols:
    df = get_data_timeframe(coin)

df = df[["Time", "Open", "High", "Low", "Close", "Volume"]]
df.to_csv("data/BTC-USD.csv", index=False)

sentiment_df = pd.read_csv("data/master_tweet_sentiment_BTC.csv")
sentiment_df["Time"] = sentiment_df["Date"] + "_" + sentiment_df["Time"]
sentiment_df = sentiment_df.drop(["Date"], axis=1)
dates = []
for i in sentiment_df["Time"]:
    dates.append(datetime.strptime(i, "%Y-%m-%d_%H:%M"))
sentiment_df["Time"] = dates
sentiment_df = sentiment_df.sort_values(by="Time").reset_index(drop=True)

dates = []
for i in df["Time"]:
    dates.append(datetime.strptime(i, "%Y-%m-%d %H:%M:%S"))
df["Time"] = dates

combined = df.copy()
combined = combined.merge(sentiment_df, how="left", on="Time")
combined = combined.interpolate(method="pad")

combined.to_csv("data/price_sentiment_btc.csv", index=False)
