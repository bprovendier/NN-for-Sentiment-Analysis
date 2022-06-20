"""State of the art model price prediction."""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings

warnings.filterwarnings("ignore")


def Dataset(Data, Date):

    Train_Data = Data["Adj. Close"][Data["Date"] < Date].to_numpy()
    Data_Train = []
    Data_Train_X = []
    Data_Train_Y = []
    for i in range(0, len(Train_Data), 5):
        try:
            Data_Train.append(Train_Data[i : i + 5])
        except:
            pass

    if len(Data_Train[-1]) < 5:
        Data_Train.pop(-1)

    Data_Train_X = Data_Train[0:-1]
    Data_Train_X = np.array(Data_Train_X)
    Data_Train_X = Data_Train_X.reshape((-1, 5, 1))
    Data_Train_Y = Data_Train[1 : len(Data_Train)]
    Data_Train_Y = np.array(Data_Train_Y)
    Data_Train_Y = Data_Train_Y.reshape((-1, 5, 1))

    Test_Data = Data["Adj. Close"][Data["Date"] >= Date].to_numpy()
    Data_Test = []
    Data_Test_X = []
    Data_Test_Y = []
    for i in range(0, len(Test_Data), 5):
        try:
            Data_Test.append(Test_Data[i : i + 5])
        except:
            pass

    if len(Data_Test[-1]) < 5:
        Data_Test.pop(-1)

    Data_Test_X = Data_Test[0:-1]
    Data_Test_X = np.array(Data_Test_X)
    Data_Test_X = Data_Test_X.reshape((-1, 5, 1))
    Data_Test_Y = Data_Test[1 : len(Data_Test)]
    Data_Test_Y = np.array(Data_Test_Y)
    Data_Test_Y = Data_Test_Y.reshape((-1, 5, 1))

    return Data_Train_X, Data_Train_Y, Data_Test_X, Data_Test_Y


def Model():
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.LSTM(
                200,
                input_shape=(5, 1),
                activation=tf.nn.leaky_relu,
                return_sequences=True,
            ),
            tf.keras.layers.LSTM(200, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(200, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(100, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(50, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(5, activation=tf.nn.leaky_relu),
        ]
    )
    return model


def scheduler(epoch):

    if epoch <= 150:
        lrate = (10 ** -5) * (epoch / 150)
    elif epoch <= 400:
        initial_lrate = 10 ** -5
        k = 0.01
        lrate = initial_lrate * math.exp(-k * (epoch - 150))
    else:
        lrate = 10 ** -6

    return lrate


def state_of_the_art():
    """
    State of the art model price prediction.

    Returns:
        state_of_the_art (Series): state of the art model price prediction.
    """
    BTC = pd.read_csv("price_sentiment_btc.csv")
    BTC["Adj. Close"] = BTC["Close"]

    epochs = [i for i in range(1, 1001, 1)]
    lrate = [scheduler(i) for i in range(1, 1001, 1)]
    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    BTC = BTC.drop(
        [
            "Number of tweets",
            "Additive score",
            "Mean score",
            "Standard deviation score",
            "Number of positive",
            "Percentage of positive",
        ],
        axis=1,
    )
    BTC.rename(columns={"Time": "Date"}, inplace=True)

    BTC["Date"] = pd.to_datetime(BTC["Date"])

    BTC_Date = "2022-05-13 04:00:00"
    BTC_Train_X, BTC_Train_Y, BTC_Test_X, BTC_Test_Y = Dataset(BTC, BTC_Date)

    BTC_Model = Model()
    BTC_Model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="mse",
        metrics=tf.keras.metrics.RootMeanSquaredError(),
    )
    BTC_hist = BTC_Model.fit(
        BTC_Train_X,
        BTC_Train_Y,
        epochs=1000,
        validation_data=(BTC_Test_X, BTC_Test_Y),
        callbacks=[callback],
    )
    history_dict = BTC_hist.history

    loss = history_dict["loss"]
    root_mean_squared_error = history_dict["root_mean_squared_error"]
    val_loss = history_dict["val_loss"]
    val_root_mean_squared_error = history_dict["val_root_mean_squared_error"]

    epochs = range(1, len(loss) + 1)
    BTC_prediction = BTC_Model.predict(BTC_Test_X)

    real = np.array(BTC["Adj. Close"][BTC["Date"] >= "2022-05-13 04:00:00"])
    pred = BTC_prediction.reshape(-1)
    fig = plt.figure(figsize=(16, 7))
    plt.plot(real, label="BTC price")
    plt.plot(pred, label="Predicted")

    plt.legend(["BTC price", "Predicted"])
    plt.xlabel("Time", fontsize=18, fontweight="bold")
    plt.ylabel("Price", fontsize=18, fontweight="bold")
    plt.title(
        "State of the art model Bitcoin price prediction",
        fontsize=20,
        fontweight="bold",
    )
    fig.savefig("state_of_the_art.eps", dpi=600, bbox_inches="tight")
    plt.show()

    state_of_the_art = pred

    return state_of_the_art
