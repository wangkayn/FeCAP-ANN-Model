import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import SVR

FEATURES = ['Voltage', 'Cycle number', 'Initial_Polarization', 'FE']
TARGET = 'Polarization'
SEED = 42
N_FEATURES = 4


def _load_data(data_path):
    df = pd.read_excel(data_path, sheet_name='alldata')
    df = df[df['FE'].isin([15.36, 19.68])].copy()
    df = df.sort_values(['Device', 'Direction', 'Cycle number', 'Number', 'Voltage'])
    df['Initial_Polarization'] = df.groupby(['Device', 'Direction', 'Cycle number', 'Number'])[TARGET].shift(1)
    df = df.dropna(subset=['Initial_Polarization'])
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
    train_idx, test_idx = next(gss.split(df, groups=df['Device']))
    X_train = df.iloc[train_idx][FEATURES]
    X_test = df.iloc[test_idx][FEATURES]
    y_train = df.iloc[train_idx][TARGET]
    y_test = df.iloc[test_idx][TARGET]
    scaler = QuantileTransformer(n_quantiles=100, output_distribution='uniform', random_state=SEED)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return {
        'X_train_scaled': X_train_scaled, 'X_test_scaled': X_test_scaled,
        'y_train': y_train, 'y_test': y_test,
    }


def _compute_metrics(y_true, y_pred, n_features=N_FEATURES):
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


def train_and_evaluate(data_path='../ccleaned_data.xlsx'):
    data = _load_data(data_path)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svr',    SVR(kernel='rbf', C=1.0, epsilon=0.1)),
    ])
    pipeline.fit(data['X_train_scaled'], data['y_train'])
    y_pred = pipeline.predict(data['X_test_scaled'])
    y_true = data['y_test'].values
    return {'combined': _compute_metrics(y_true, y_pred)}


if __name__ == '__main__':
    results = train_and_evaluate()
    m = results['combined']
    print(f"{'SVR':<20} {m['MSE']:.6f}  {m['RMSE']:.6f}  {m['MAE']:.6f}  {m['Adj_R2']:.6f}")
