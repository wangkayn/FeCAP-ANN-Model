# FeCAP-ANN-Model

Code for **"A Data-Driven ANN-Based Model for FeCAP & FeFET: Orienting to SPICE and Circuit Design"**

## Overview

This repository provides a complete, reproducible pipeline for building a data-driven ANN model for ferroelectric capacitors (FeCAP) and ferroelectric FETs (FeFET). The trained model can be directly exported to **Verilog-A** for use in SPICE circuit simulators, replacing traditional physics-based compact models.

### Workflow

```
Measured P-V data ──> Train ANN (PyTorch) ──> Export weights ──> Verilog-A compact model ──> SPICE simulation
```

### Key Features

- **Data-driven**: No material-specific physical equations required — the ANN learns the polarization-voltage relationship directly from measurement data.
- **Direction-aware**: Separate models for rising (Direction 1) and falling (Direction 0) hysteresis branches, capturing the ferroelectric memory effect.
- **SPICE-ready**: `pth2va.py` converts trained PyTorch weights into Verilog-A code with explicit algebraic expressions, compatible with standard SPICE simulators (Cadence Spectre, Synopsys HSPICE, etc.).
- **Baseline comparison**: Includes Random Forest, SVR, and LASSO baselines for benchmarking.

## Repository Structure

| File | Description |
|---|---|
| `ann_model.py` | ANN model definition, training, and evaluation (4 hidden layers: 36→180→210→180, LeakyReLU, 5% dropout) |
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
python run_all.py --data ../ccleaned_data.xlsx
```

Output example:

```
Method               MSE          RMSE         MAE          Adj R²
------------------------------------------------------------------------
Random Forest        3.241182     1.800328     1.107198     0.990977
SVR                  7.192422     2.681868     1.627498     0.979953
LASSO                32.927925    5.738286     4.375498     0.908347
Our Work (ANN)       0.724000     0.851000     0.343000     0.998000
```

### 2. Export Trained Weights to Verilog-A

After training, save the model weights (`.pth`) and scaler parameters (`va_scalers.json`), then run:

```bash
python pth2va.py --dir0 best_va_dir0.pth --dir1 best_va_dir1.pth --scalers va_scalers.json --out ann_weights.va
```

This generates a Verilog-A code snippet containing the explicit ANN forward pass (all weights unrolled as algebraic expressions), ready to be embedded in your SPICE compact model.

## Model Architecture

- **Input features**: Voltage, Cycle number, Initial Polarization, FE thickness, Measurement
- **Network**: 4 hidden layers (36 → 180 → 210 → 180), LeakyReLU activation, 5% dropout
- **Output**: Polarization (uC/cm²)
- **Training**: Adam optimizer (lr=0.001), MSE loss, early stopping (patience=50), batch size=32
- **Data split**: 70/30 group-based split (GroupShuffleSplit by Device), ensuring no data leakage across devices
- **Normalization**: QuantileTransformer (100 quantiles, uniform distribution)

## Verilog-A Integration

The exported Verilog-A model uses:

- **`tanh` activation** — natively supported by Verilog-A, smooth and differentiable
- **Min-max normalization** — embedded as constants in the generated code
- **Two networks** — one per sweep direction, blended via `transition()` for Jacobian-friendly SPICE convergence
- **Timer-decoupled direction switching** — avoids convergence issues from abrupt state changes during Newton-Raphson iterations

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
