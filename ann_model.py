"""
ANN model for FeCAP polarization prediction.

Data requirements:
    An Excel file (.xlsx) with a sheet named 'alldata' containing columns:
        - Voltage         : Applied voltage (V)
        - Polarization    : Measured polarization (uC/cm^2) [target]
        - Cycle number    : Wake-up cycle count
        - FE              : Ferroelectric layer thickness (nm)
        - Direction       : Sweep direction (0 or 1), used for computing P_init
        - Device          : Device identifier, used for group-based splitting
        - Number          : Measurement sweep index within a cycle

    The script automatically computes Initial_Polarization (P_init) as
    the previous polarization value within each (Device, Direction,
    Cycle number, Number) group.

Usage:
    python ann_model.py --data your_data.xlsx
"""
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import DataLoader, TensorDataset

FEATURES = ['Voltage', 'Cycle number', 'Initial_Polarization', 'FE']
TARGET = 'Polarization'
SEED = 42
BATCH_SIZE = 32
LR = 0.001
MAX_EPOCHS = 1000
PATIENCE = 50
N_INPUTS = 4


def _load_data(data_path):
    df = pd.read_excel(data_path, sheet_name='alldata')
    df = df[df['FE'].isin([15.36, 19.68])].copy()
    df = df.sort_values(['Device', 'Direction', 'Cycle number', 'Number', 'Voltage'])
    df['Initial_Polarization'] = df.groupby(['Device', 'Direction', 'Cycle number', 'Number'])[TARGET].shift(1)
    df = df.dropna(subset=['Initial_Polarization'])

    # Step 1: split off Test set (30%) by Device
    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
    trainval_idx, test_idx = next(gss_test.split(df, groups=df['Device']))
    df_trainval = df.iloc[trainval_idx]
    df_test = df.iloc[test_idx]

    # Step 2: split Train/Val from the remaining 70%
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(gss_val.split(df_trainval, groups=df_trainval['Device']))
    df_train = df_trainval.iloc[train_idx]
    df_val = df_trainval.iloc[val_idx]

    # Scaler fit ONLY on train
    scaler = QuantileTransformer(n_quantiles=100, output_distribution='uniform', random_state=SEED)
    X_train_scaled = scaler.fit_transform(df_train[FEATURES])
    X_val_scaled = scaler.transform(df_val[FEATURES])
    X_test_scaled = scaler.transform(df_test[FEATURES])

    return {
        'X_train_scaled': X_train_scaled, 'y_train': df_train[TARGET],
        'X_val_scaled': X_val_scaled,     'y_val': df_val[TARGET],
        'X_test_scaled': X_test_scaled,   'y_test': df_test[TARGET],
    }


def _compute_metrics(y_true, y_pred, n_features=N_INPUTS):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred
    mse = float(np.mean(residuals ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residuals)))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    n = len(y_true)
    p = n_features
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else float('nan')
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'Adj_R2': adj_r2}


def build_model():
    return nn.Sequential(
        nn.Linear(N_INPUTS, 36), nn.LeakyReLU(), nn.Dropout(0.05),
        nn.Linear(36, 180),      nn.LeakyReLU(), nn.Dropout(0.05),
        nn.Linear(180, 210),     nn.LeakyReLU(), nn.Dropout(0.05),
        nn.Linear(210, 180),     nn.LeakyReLU(), nn.Dropout(0.05),
        nn.Linear(180, 1),
    )


def _set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


def _train(X_train, y_train, X_val, y_val, device):
    """Train with early stopping on validation set (NOT test set)."""
    _set_seed()
    X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)
    X_v  = torch.tensor(X_val,   dtype=torch.float32).to(device)
    y_v  = torch.tensor(y_val.values,   dtype=torch.float32).unsqueeze(1).to(device)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_v), y_v).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model


def _predict(model, X, device):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        return model(X_t).cpu().numpy().flatten()


def train_and_evaluate(data_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = _load_data(data_path)
    model = _train(
        data['X_train_scaled'], data['y_train'],
        data['X_val_scaled'],   data['y_val'],
        device,
    )
    y_pred = _predict(model, data['X_test_scaled'], device)
    y_true = data['y_test'].values
    return {'combined': _compute_metrics(y_true, y_pred, n_features=N_INPUTS)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate ANN model')
    parser.add_argument('--data', required=True, help='Path to Excel data file (.xlsx)')
    args = parser.parse_args()
    results = train_and_evaluate(args.data)
    m = results['combined']
    print(f"{'Our Work (ANN)':<20} {m['MSE']:.6f}  {m['RMSE']:.6f}  {m['MAE']:.6f}  {m['Adj_R2']:.6f}")
