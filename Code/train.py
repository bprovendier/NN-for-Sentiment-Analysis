"""Module for model training."""

import os
import yaml
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import preprocess


with open("model/params.yaml", "r") as params_file:
    params = yaml.safe_load(params_file)

model_dir = params["model_dir"]


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_len=1):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return self.X.__len__() - self.seq_len

    def __getitem__(self, index):
        return self.X[index : index + self.seq_len], self.y[index + self.seq_len]


class TSModelGRU(nn.Module):
    def __init__(self, n_features, n_hidden=32, n_layers=2):
        super(TSModelGRU, self).__init__()

        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=n_hidden,
            batch_first=True,
            num_layers=n_layers,
            dropout=0.1,
        )
        self.linear = nn.Linear(n_hidden, 1)

    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.n_hidden).requires_grad_()
        out, (hn) = self.gru(x, (h0.detach()))
        y_pred = self.linear(out[:, -1, :])

        return y_pred


direction_loss = None


def train_model(
    train_df,
    val_df,
    label_name,
    sequence_length,
    batch_size,
    n_epochs,
    n_epochs_stop,
):
    """
    Train GRU model.

    Arguments:
        train_df (DataFrame): training dataframe.
        val_df (DataFrame): validation dataframe.
        label_name (str): name of the data label to predict.
        sequence_length (int): lenght of sequence to predict.
        batch_size (int): size of batch.
        n_epochs (int): number of training epochs.
        n_epochs_stop (int): number of epochs before early stopping.

    Returns:
        hist (DataFrame): training and validation losses.
    """
    print("Starting with model training...")

    # create dataloaders
    train_dataset = TimeSeriesDataset(
        np.array(train_df), np.array(train_df[label_name]), seq_len=sequence_length
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    val_dataset = TimeSeriesDataset(
        np.array(val_df), np.array(val_df[label_name]), seq_len=sequence_length
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # set up training
    n_features = train_df.shape[1]
    model = TSModelGRU(n_features)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_hist = []
    val_hist = []

    # start training
    best_loss = np.inf
    epochs_no_improve = 0
    for epoch in range(1, n_epochs + 1):
        running_loss = 0
        model.train()
        model = model.float()

        for batch_idx, (data, target) in enumerate(train_loader, 1):
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            data = torch.tensor(np.array(data))
            output = model(data.float())
            loss = criterion(output.flatten(), target.type_as(output))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        running_loss /= len(train_loader)
        train_hist.append(running_loss)

        # test loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data = torch.Tensor(np.array(data))
                output = model(data)
                loss = criterion(output.flatten(), target.type_as(output))
                val_loss += loss.item()
            val_loss /= len(val_loader)
            val_hist.append(val_loss)

            # early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                os.remove(Path(model_dir, "model.pt"))
                torch.save(model.state_dict(), Path(model_dir, "model.pt"))
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve == n_epochs_stop:
                print("Early stopping.")
                break

        print(
            f"Epoch {epoch} train loss: {round(running_loss,5)} validation loss: {round(val_loss,5)}"
        )

        hist = pd.DataFrame()
        hist["training_loss"] = train_hist
        hist["validation_loss"] = val_hist

    print("Completed.")

    return hist


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence-length", type=int, default=params["sequence_length"]
    )
    parser.add_argument("--batch-size", type=int, default=params["batch_size"])
    parser.add_argument("--n-epochs", type=int, default=params["n_epochs"])
    parser.add_argument("--n-epochs-stop", type=int, default=params["n_epochs_stop"])
    args = parser.parse_args()

    train_df = preprocess.load_data("train.csv")
    val_df = preprocess.load_data("val.csv")
    label_name = params["label_name"]

    train_model(
        train_df,
        val_df,
        label_name,
        args.sequence_length,
        args.batch_size,
        args.n_epochs,
        args.n_epochs_stop,
    )
