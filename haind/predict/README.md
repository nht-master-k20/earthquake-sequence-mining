# Earthquake LSTM Prediction Pipeline

PyTorch-based LSTM model for predicting time-to-next and magnitude of next earthquake.

## 📁 Files

| File | Description |
|------|-------------|
| `config.py` | Configuration settings |
| `data_preparer.py` | Data loading and preprocessing |
| `model_pytorch.py` | PyTorch LSTM model definition |
| `train_pytorch.py` | Training script |
| `predict_pytorch.py` | Prediction script |
| `quick_test.py` | Quick test script |

## 🚀 Usage

### 1. Quick Test (5 epochs)
```bash
python quick_test.py
```

### 2. Train Model

#### Train on ALL data (default)
```bash
# Train with default parameters
python train_pytorch.py --epochs 50

# Train with custom hyperparameters
python train_pytorch.py --epochs 100 --batch-size 128 --lr 0.0001
python train_pytorch.py --epochs 50 --hidden 256 128 64 --dropout 0.2

# Test mode (5 epochs)
python train_pytorch.py --test
```

#### Train on SINGLE region
```bash
python train_pytorch.py --region R257_114 --epochs 50
```

### 3. Make Predictions
```bash
# Using latest model
python predict_pytorch.py --region R257_114

# Manual input
python predict_pytorch.py --lat 15.5 --lon 108.5 --mag 5.2 --depth 10
```

## 🎛️ Model Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 50 | Number of epochs |
| `--batch-size` | 64 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--hidden` | [128, 64] | LSTM hidden units |
| `--dropout` | 0.3 | Dropout rate |
| `--patience` | 15 | Early stopping patience |

## Optional Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--region` | None | Train on specific region only |
| `--min-events` | 100 | Min events/region to include |
| `--device` | cpu | cpu or cuda |
| `--test` | False | Test mode (5 epochs) |

## 📊 Model Architecture

```
Input (batch_size, 5, 26)
    ↓
LSTM(128) → Dropout → BatchNorm
    ↓
LSTM(64) → Dropout → BatchNorm
    ↓
Dense(64) → ReLU → BatchNorm → Dropout
    ↓
    ├─→ Dense(1, ReLU) → time_to_next
    ├─→ Dense(1) → next_mag
    └─→ Dense(1, Sigmoid) → next_mag_binary (M5+)
```

## 📈 Features

- **26 input features**: Original (9) + Core (5) + Sequence (6) + LSTM (5) - region (1)
- **3 outputs**: time_to_next (seconds), next_mag (magnitude), next_mag_binary (probability)
- **Sequence length**: 5 (last 5 events)

## 📂 Output

Trained models saved to `models/`:
- `model_<region>_<timestamp>.pt` - Model checkpoint
- `scaler_<region>_<timestamp>.pkl` - Feature scaler
- `history_<region>_<timestamp>.json` - Training history
- `metadata_<region>_<timestamp>.json` - Model metadata

## 🔧 Requirements

```
torch
numpy
pandas
scikit-learn
```

## 📝 Author

haind - Earthquake Sequence Mining Project
Date: 2025-03-25
