"""Custom loss model price forecasting."""
import numpy as np

np.random.seed(1)  # for reproducibility
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# set parameters for model
params = {"BATCH_SIZE": 32, "EPOCHS": 300, "LR": 0.0010000, "TIME_STEPS": 24}

# path to input & output
PATH_TO_DRIVE = "custom_loss"
INPUT_PATH = PATH_TO_DRIVE + "\\inputs"
OUTPUT_PATH = PATH_TO_DRIVE + "\\outputs"
TIME_STEPS = params["TIME_STEPS"]
BATCH_SIZE = params["BATCH_SIZE"]
stime = time.time()


def print_time(text, stime):
    seconds = time.time() - stime
    print(text, seconds // 60, "minutes : ", np.round(seconds % 60), "seconds")


def trim_dataset(mat, batch_size):

    # trims dataset to a size that's divisible by BATCH_SIZE
    no_of_rows_drop = mat.shape[0] % batch_size

    if no_of_rows_drop > 0:
        return mat[:-no_of_rows_drop]
    else:
        return mat


def build_timeseries(mat, y_col_index):

    dim_0 = mat.shape[0] - TIME_STEPS
    dim_1 = mat.shape[1]

    x = np.zeros((dim_0, TIME_STEPS, dim_1))
    y = np.zeros((dim_0,))

    for i in range(dim_0):
        x[i] = mat[i : TIME_STEPS + i]
        y[i] = mat[TIME_STEPS + i, y_col_index]

    return x, y


def custom_loss(y_true, y_pred):

    y_true_next = y_true[1:]
    y_pred_next = y_pred[1:]

    y_true_tdy = y_true[:-1]
    y_pred_tdy = y_pred[:-1]

    y_true_diff = tf.subtract(y_true_next, y_true_tdy)
    y_pred_diff = tf.subtract(y_pred_next, y_pred_tdy)

    standard = tf.zeros_like(y_pred_diff)

    y_true_move = tf.greater_equal(y_true_diff, standard)
    y_pred_move = tf.greater_equal(y_pred_diff, standard)
    y_true_move = tf.reshape(y_true_move, [-1])
    y_pred_move = tf.reshape(y_pred_move, [-1])

    condition = tf.not_equal(y_true_move, y_pred_move)
    indices = tf.where(condition)

    ones = tf.ones_like(indices)

    indices = tf.add(indices, ones)

    indices = K.cast(indices, dtype="int32")

    updates = K.cast(tf.ones_like(indices), dtype="float32")
    alpha = 1000

    scatter = (
        tf.scatter_nd(tf.cast(indices, tf.int32), updates, tf.shape(y_pred)) * alpha
    )
    scatter = tf.where(tf.equal(scatter, 0), tf.ones_like(scatter), scatter)
    direction_loss = K.cast(scatter, dtype="float32")

    custom_loss = K.mean(
        tf.multiply(K.square(y_true - y_pred), direction_loss), axis=-1
    )

    return custom_loss


def create_lstm_model(x_t):

    lstm_model = Sequential()
    lstm_model.add(
        LSTM(
            64,
            batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]),
            dropout=0.05,
            recurrent_dropout=0.0,
            stateful=True,
            return_sequences=True,
            kernel_initializer="random_uniform",
        )
    )

    lstm_model.add(LSTM(32, dropout=0.05))

    lstm_model.add(Dense(1, activation="linear"))

    optimizer = tf.keras.optimizers.Adam(lr=params["LR"])
    lstm_model.compile(loss=custom_loss, optimizer=optimizer)

    return lstm_model


def custom_loss():
    """
    Custom_loss model price forecasting on the test data.

    Returns:
        custom_loss (Series): custom loss model price prediction.
    """
    data = pd.read_csv("old_price_sentiment_btc.csv")
    train_cols = ["Open", "High", "Low", "Close", "Volume"]
    df_train, df_test = train_test_split(
        data, train_size=0.5, test_size=0.5, shuffle=False
    )

    x = df_train.loc[:, train_cols].values
    min_max_scaler = MinMaxScaler()
    x_train = min_max_scaler.fit_transform(x)
    x_test = min_max_scaler.transform(df_test.loc[:, train_cols])
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    x_t, y_t = build_timeseries(x_train, 3)
    x_t = trim_dataset(x_t, BATCH_SIZE)
    y_t = trim_dataset(y_t, BATCH_SIZE)

    x_temp, y_temp = build_timeseries(x_test, 3)
    x_val, x_test_t = np.array_split(trim_dataset(x_temp, BATCH_SIZE), 2)
    y_val, y_test_t = np.array_split(trim_dataset(y_temp, BATCH_SIZE), 2)

    model = None

    is_update_model = False

    if model is None or is_update_model:

        lstm_model = create_lstm_model(x_t)

        mcp = ModelCheckpoint(
            os.path.join(OUTPUT_PATH, "best_lstm_model.h5"),
            monitor="val_loss",
            verbose=2,
            save_best_only=True,
            save_weights_only=False,
            mode="min",
            period=1,
        )

        history_lstm = lstm_model.fit(
            x_t,
            y_t,
            epochs=params["EPOCHS"],
            verbose=1,
            batch_size=BATCH_SIZE,
            shuffle=False,
            validation_data=(
                trim_dataset(x_val, BATCH_SIZE),
                trim_dataset(y_val, BATCH_SIZE),
            ),
            callbacks=[mcp],
        )

    saved_model = load_model(
        os.path.join(OUTPUT_PATH, "best_lstm_model.h5"),
        custom_objects={"custom_loss": custom_loss},
    )
    print(saved_model)

    y_pred_lstm = saved_model.predict(
        trim_dataset(x_test_t, BATCH_SIZE), batch_size=BATCH_SIZE
    )
    y_pred_lstm = y_pred_lstm.flatten()
    y_test_t = trim_dataset(y_test_t, BATCH_SIZE)

    error_lstm = mean_squared_error(y_test_t, y_pred_lstm)

    y_pred_lstm_org = (
        y_pred_lstm * min_max_scaler.data_range_[3]
    ) + min_max_scaler.data_min_[3]
    y_test_t_org = (
        y_test_t * min_max_scaler.data_range_[3]
    ) + min_max_scaler.data_min_[3]

    fig = plt.figure(figsize=(16, 7))
    plt.plot(y_pred_lstm_org)
    plt.plot(y_test_t_org)
    plt.title(
        "Custom loss model Bitcoin price prediction", fontsize=18, fontweight="bold"
    )
    plt.ylabel("Price", fontsize=16, fontweight="bold")
    plt.xlabel("Time", fontsize=16, fontweight="bold")
    plt.legend(["Prediction", "BTC price"], loc="best")
    fig.savefig("custom_loss_model.eps", dpi=600, bbox_inches="tight")

    y_test_t_final = np.zeros((len(y_test_t_org) - 1,))
    y_pred_lstm_final = np.zeros((len(y_pred_lstm) - 1,))

    # convert prediction into binary output (up or down movement)
    for i in range(len(y_pred_lstm_org) - 1):
        if y_pred_lstm_org[i + 1] >= y_pred_lstm_org[i]:
            y_pred_lstm_final[i] = 1
        else:
            y_pred_lstm_final[i] = 0

    # convert prediction into binary output (up or down movement)
    for i in range(len(y_test_t_org) - 1):
        if y_test_t_org[i + 1] >= y_test_t_org[i]:
            y_test_t_final[i] = 1
        else:
            y_test_t_final[i] = 0

    error_lstm = mean_absolute_error(y_test_t_final, y_pred_lstm_final)
    print("Error is", error_lstm, y_pred_lstm_final.shape, y_test_t_final.shape)
    print(y_pred_lstm_final[0:15])
    print(y_test_t_final[0:15])

    custom_loss = y_pred_lstm_org

    return custom_loss
