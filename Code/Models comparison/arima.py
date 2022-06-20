"""ARIMA model price forecasting."""
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.arima_model import ARIMA


plt.style.use("ggplot")


def arima_model():
    """
    ARIMA model price forecasting on the test data.

    Returns:
        arima (Series): price predictions  for the test period.
        real_price (Series): actual Bitcoin price during the test period.
    """
    df = pd.read_csv("old_price_sentiment_btc.csv")
    arima = pd.DataFrame()
    arima["True"] = df["Close"]

    arima["Res"] = ARIMA(arima["True"], order=(1, 0, 0)).fit().resid
    arima["Prediction"] = arima["True"] - arima["Res"]
    arima = arima[-2320:]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.5)

    arima[["True", "Prediction"]].plot(ax=axes[0])
    axes[0].set_title("ARIMA Bitcoin Price Prediction", fontsize=18, fontweight="bold")
    arima[["True", "Prediction"]][-100:].plot(ax=axes[1])
    axes[1].set_title("Zooming into Last 100 Periods", fontsize=18, fontweight="bold")
    fig.savefig("arima_model.eps", dpi=600, bbox_inches="tight")

    arima = arima["Prediction"].reset_index(drop=True)
    real_price = arima["True"].reset_index(drop=True)

    return arima, real_price
