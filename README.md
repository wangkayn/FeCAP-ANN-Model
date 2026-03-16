# FeCAP-ANN-Model

Code for **"A Data-Driven ANN-Based Model for FeCAP & FeFET: Orienting to SPICE and Circuit Design"**

## Overview

This repository provides a complete, reproducible pipeline for building a data-driven ANN model for ferroelectric capacitors (FeCAP) and ferroelectric FETs (FeFET). The trained model can be directly exported to **Verilog-A** for use in SPICE circuit simulators, replacing traditional physics-based compact models.

### Workflow

```
Measured P-V data --> Train ANN (PyTorch) --> Export weights --> Verilog-A compact model --> SPICE simulation
```

### Key Features

- **Data-driven**: No material-specific physical equations required --the ANN learns the polarization-voltage relationship directly from measurement data.
- **Single unified model**: One network handles both rising and falling hysteresis branches --the initial polarization (P_init) implicitly encodes the sweep direction.
- **SPICE-ready**: `pth2va.py` converts trained PyTorch weights into Verilog-A code with explicit algebraic expressions, compatible with standard SPICE simulators (Cadence Spectre, Synopsys HSPICE, etc.).
- **Baseline comparison**: Includes Random Forest, SVR, and LASSO baselines for benchmarking.

## Data Format

All scripts expect an Excel file (`.xlsx`) with a sheet named **`alldata`** containing the following columns:

| Column | Description |
|---|---|
| `Voltage` | Applied voltage (V) |
| `Polarization` | Measured polarization (uC/cm^2) -- prediction target |
| `Cycle number` | Wake-up cycle count (e.g., 0, 1, 10, 100, ...) |
| `FE` | Ferroelectric layer thickness in nm |
| `Direction` | Sweep direction (0 = falling, 1 = rising) |
| `Device` | Device identifier (integer), used for group-based train/test splitting |
| `Number` | Measurement sweep index within a cycle |

The `Initial_Polarization` feature (P_init) is **automatically computed** by each script as the previous polarization value within each `(Device, Direction, Cycle number, Number)` group. You do not need to include it in your data file.

To use your own data, prepare an Excel file matching this format and pass it via `--data`:

```bash
python run_all.py --data path/to/your_data.xlsx
```

## Repository Structure

| File | Description |
|---|---|
| `ann_model.py` | ANN model definition, training, and evaluation (4 hidden layers: 36->180->210->180, LeakyReLU, 5% dropout) |
| `baseline_rf.py` | Random Forest baseline |
| `baseline_svr.py` | Support Vector Regression baseline |
| `baseline_lasso.py` | LASSO regression baseline |
| `run_all.py` | Run all models and print comparison table |
| `pth2va.py` | Convert trained `.pth` weights to Verilog-A ANN forward-pass code |
| `requirements.txt` | Python dependencies |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train & Evaluate All Models

```bash
python run_all.py --data path/to/your_data.xlsx
```

Output example:

```
Method               MSE          RMSE         MAE          Adj R2
------------------------------------------------------------------------
Random Forest        0.449215     0.670235     0.162684     0.998726
SVR                  1.983797     1.408473     0.650730     0.994376
LASSO                8.591007     2.931042     2.608453     0.975645
Our Work (ANN)       0.710263     0.842771     0.518530     0.997986
```

### 2. Export Trained Weights to Verilog-A

After training, save the model weights (`.pth`) and scaler parameters (`va_scalers.json`), then run:

```bash
python pth2va.py --model best_model.pth --scalers va_scalers.json --out ann_weights.va
```

This generates a Verilog-A code snippet containing the explicit ANN forward pass (all weights unrolled as algebraic expressions), ready to be embedded in your SPICE compact model.

## Model Architecture

- **Input features (4)**: Applied voltage, Cycle number, Initial polarization (P_init), FE thickness (t_FE)
- **Network**: 4 hidden layers (36 -> 180 -> 210 -> 180), LeakyReLU activation, 5% dropout
- **Output**: Polarization (uC/cm^2)
- **Training**: Adam optimizer (lr=0.001), MSE loss, early stopping (patience=50), batch size=32
- **Data split**: Train (56%) / Validation (14%) / Test (30%), group-based by Device (GroupShuffleSplit), ensuring no data leakage
- **Normalization**: QuantileTransformer (100 quantiles, uniform distribution), fit on training set only

## Verilog-A Integration

The exported Verilog-A model uses:

- **`tanh` activation** --natively supported by Verilog-A, smooth and differentiable
- **Min-max normalization** --embedded as constants in the generated code
- **Single network** --no direction switching needed, P_init implicitly determines the hysteresis branch

Module interface (drop-in replacement for physics-based pfecap):
```verilog
module pfecap(pos, neg, qin);
    // pos, neg: capacitor terminals
    // qin: input charge/polarization (as voltage)
```

## Citation

If you use this code, please cite our paper.

## License

MIT
