"""
- defines LSTM model and its training/forecasting pipeline
- WindowDataset builds input-output pairs and standardizes
- ApplyPublicationLag zeroes the most recent observations in each window
- train runs the optimization loop with early stopping
- forecast produces recursive multi-step predictions for the target variable
- lstm trains n_models with different seeds and returns the seed-averaged forecast
"""

# imports, config & utilities
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, Subset

from config import DEVICE, TARGET, DATA_PATH, METADATA_FEATURES_PATH, SEED
from helpers import set_seed

# parameters
NH = 4
MAX_EPOCHS = 1000
LR = 0.001
NUM_LAYERS = 1
WINDOW_SIZE = 4
BATCH_SIZE = 16
VAL_QUARTERS = 6
PATIENCE = 100
N_MODELS = 5


class WindowDataset(Dataset):
    '''
    - loads dataset csv and restricts it to rows with t <= origin
    - standardizes columns using mean/std of the restricted frame
    - returns a (window_size, F) input window and the (F,) next-step target per index
    '''
    def __init__(self, data_path, metadata_path, window_size, origin=None, transform=None):
        self.metadata = pd.read_csv(metadata_path)
        self.window_size = window_size
        self.transform = transform

        df = pd.read_csv(data_path)
        df['t'] = pd.to_datetime(df['t'])
        self.cols = [c for c in df.columns if c != 't']
        self.target_idx = self.cols.index(TARGET)

        if origin is not None:
            df = df[df['t'] <= origin]

        arr = df[self.cols].values.astype('float32')
        self.mus = np.nanmean(arr, axis=0)
        self.sigs = np.nanstd(arr, axis=0)
        self.arr = torch.as_tensor((arr - self.mus) / self.sigs)

    def __len__(self):
        return len(self.arr) - self.window_size

    def __getitem__(self, idx):
        w = self.window_size
        x = self.arr[idx : idx + w]
        y = self.arr[idx + w]
        if self.transform:
            x = self.transform(x)
        return x, y


class ApplyPublicationLag:
    '''
    - zeroes the last k rows of each lagged column in a standardized window
    - k per column is read from the 'Applied NaNs' metadata field
    - zero corresponds to the column mean under standardization
    '''
    def __init__(self, metadata, cols):
        self.lag_info = [
            (cols.index(row['Name']), int(row['Applied NaNs']))
            for _, row in metadata.iterrows()
            if int(row['Applied NaNs']) > 0 and row['Name'] in cols
        ]

    def __call__(self, x):
        x = x.clone()
        for col_idx, lag in self.lag_info:
            x[-lag:, col_idx] = 0.0
        return x


class LSTMForecaster(nn.Module):
    '''
    - LSTM with a linear head mapping the last hidden state to all features
    - many-to-one in time, multi-output across features
    '''
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x, hc=None):
        out, hc = self.rnn(x, hc)
        return self.fc(out[:, -1, :]), hc


def train(model, train_loader, val_loader, *, lr, max_epochs, patience, early_stopping):
    '''
    - trains model with MSE loss and Adam optimizer
    - tracks train and val losses per epoch
    - if early_stopping is True: restores the best-val state and breaks after `patience`
        epochs without improvement
    - returns train losses, val losses, and the epoch achieving best val loss
    '''
    opt = Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_losses, val_losses = [], []
    best_val, best_state, since_best = float('inf'), None, 0

    for epoch in range(max_epochs):
        model.train()
        batch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb)[0], yb)
            loss.backward()
            opt.step()
            batch_losses.append(loss.item())
        train_losses.append(np.mean(batch_losses))

        model.eval()
        with torch.no_grad():
            val_batch_losses = [
                loss_fn(model(xb.to(DEVICE))[0], yb.to(DEVICE)).item()
                for xb, yb in val_loader
            ]
        val_losses.append(np.mean(val_batch_losses))

        if early_stopping:
            if val_losses[-1] < best_val:
                best_val = val_losses[-1]
                best_state = copy.deepcopy(model.state_dict())
                since_best = 0
            else:
                since_best += 1
                if since_best >= patience:
                    break

    if early_stopping and best_state is not None:
        model.load_state_dict(best_state)
        best_epoch = len(train_losses) - since_best
    else:
        best_epoch = len(train_losses)

    return {'train_losses': train_losses, 'val_losses': val_losses, 'best_epoch': best_epoch}


def forecast(model, arr, target_idx, mus, sigs, horizon, window_size):
    '''
    - recursive multi-step forecast for the target column
    - seeds the model with the last `window_size` rows of arr, then feeds each prediction
        back as the next single-step input while carrying the hidden state forward
    - returns `horizon` predictions in original (denormalized) units
    '''
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(arr[-window_size:]).unsqueeze(0).to(DEVICE)
        preds = np.zeros(horizon, dtype=np.float32)
        hc = None
        for t in range(horizon):
            pred, hc = model(x, hc)
            preds[t] = pred[0, target_idx].item()
            x = pred.unsqueeze(1)
    return (preds * sigs[target_idx] + mus[target_idx]).tolist()


def lstm(horizon, origin, *, seed=SEED, hidden_size=NH, num_layers=NUM_LAYERS,
         lr=LR, max_epochs=MAX_EPOCHS, window_size=WINDOW_SIZE, batch_size=BATCH_SIZE,
         val_quarters=VAL_QUARTERS, patience=PATIENCE,
         n_models=N_MODELS, early_stopping=True):
    '''
    - runs the full LSTM pipeline for one origin: builds the dataset, trains n_models with
        different seeds, and forecasts `horizon` steps with each
    - val split is the last `val_quarters` windows of the dataset
    - returns the seed-averaged forecast, the per-seed forecast array, and diagnostics
    '''
    all_preds = []
    all_diagnostics = []

    for i in range(n_models):
        set_seed(seed + i)

        dataset = WindowDataset(DATA_PATH, METADATA_FEATURES_PATH, window_size, origin=origin)
        dataset.transform = ApplyPublicationLag(dataset.metadata, dataset.cols)

        n = len(dataset)
        train_loader = DataLoader(Subset(dataset, range(n - val_quarters)), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(Subset(dataset, range(n - val_quarters, n)), batch_size=batch_size, shuffle=False)

        model = LSTMForecaster(len(dataset.cols), hidden_size, num_layers).to(DEVICE)
        diag = train(model, train_loader, val_loader, lr=lr, max_epochs=max_epochs,
                     patience=patience, early_stopping=early_stopping)
        preds = forecast(model, dataset.arr, dataset.target_idx,
                         dataset.mus, dataset.sigs, horizon, window_size)

        all_preds.append(preds)
        all_diagnostics.append(diag)

    return np.mean(all_preds, axis=0).tolist(), np.array(all_preds), all_diagnostics