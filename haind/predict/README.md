# Earthquake Prediction System

Complete pipeline for training LSTM model and predicting next earthquake time & magnitude.

## 📁 Directory Structure

```
haind/predict/
├── config.py                 # Configuration
├── data_preparer.py          # Data preparation
├── model_builder.py          # LSTM model builder
├── train.py                  # Training script
├── predict.py                # Prediction script
├── README.md                 # This file
├── models/                   # Trained models (created automatically)
└── logs/                     # Training logs (created automatically)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install tensorflow pandas numpy scikit-learn
```

### 2. Generate Features (if not already done)

```bash
cd haind
python add_advanced_features_mp.py
```

This will create `features_lstm.csv` with 29 columns.

### 3. Train Model

#### Option A: Train for specific region
```bash
cd predict
python train.py --region R221_570 --epochs 50
```

#### Option B: Train on all regions
```bash
cd predict
python train.py --epochs 50
```

#### Options:
- `--min-events`: Minimum events per region (default: 100)
- `--batch-size`: Batch size (default: 32)
- `--test`: Run in test mode with small dataset

### 4. Make Prediction

#### Option A: Predict from historical data
```bash
python predict.py --region R221_570
```

#### Option B: Predict from user input
```bash
python predict.py --input events.json
```

## 📝 Input Format

### events.json format:
```json
[
  {
    "time": "2025-03-25 14:00:00",
    "latitude": 20.5,
    "longitude": 105.0,
    "depth": 10.5,
    "mag": 4.5,
    "sig": 250,
    "mmi": 4.2,
    "cdi": 3.5,
    "felt": 120
  },
  {
    "time": "2025-03-25 14:30:00",
    "latitude": 20.52,
    "longitude": 105.01,
    "depth": 11.2,
    "mag": 4.2,
    "sig": 180,
    "mmi": 3.8,
    "cdi": 3.0,
    "felt": 80
  },
  {
    "time": "2025-03-25 15:00:00",
    "latitude": 20.51,
    "longitude": 105.02,
    "depth": 10.8,
    "mag": 5.1,
    "sig": 500,
    "mmi": 5.5,
    "cdi": 4.5,
    "felt": 500
  }
]
```

## 📊 Output Format

```python
{
    'region_code': 'R221_570',
    'last_event_time': '2025-03-25 15:00:00',
    'predicted_time': '2025-03-25 17:30:00',
    'time_to_next_seconds': 9000.0,
    'time_to_next_hours': 2.5,
    'predicted_magnitude': 4.8,
    'is_m5_plus': False,
    'confidence': 0.85
}
```

## 🏗️ Model Architecture

```
Input: (batch_size, 5, 26)
  - 5 timesteps (interval_lag1-5)
  - 26 features (total - time column - 3 targets)

LSTM Layers:
  - LSTM(128) → Dropout(0.3) → BatchNorm
  - LSTM(64) → Dropout(0.3) → BatchNorm
  - LSTM(32) → Dropout(0.3) → BatchNorm

Dense: Dense(64) → Dropout(0.3)

Outputs:
  - time_to_next: Dense(1, activation='relu')
  - next_mag: Dense(1, activation='linear', clip 0-10)
  - next_mag_binary: Dense(1, activation='sigmoid')
```

## 📈 Features (26 input)

**Original (9):**
- time, latitude, longitude, depth, mag, sig, mmi, cdi, felt, region_code

**Core (5):**
- is_aftershock, mainshock_mag, seismicity_density_100km
- coulomb_stress_proxy, regional_b_value

**Sequence (6):**
- sequence_id, seq_position, is_seq_mainshock
- seq_mainshock_mag, seq_length, time_since_seq_start_sec

**LSTM Temporal (5):**
- time_since_last_event, time_since_last_M5
- interval_lag1, interval_lag2, interval_lag3, interval_lag4, interval_lag5

## 🔧 Troubleshooting

### Error: "Not enough events for region"
- Solution: Lower `--min-events` parameter
- Or use `--train --region all` to train on all regions

### Error: "Model file not found"
- Solution: Train model first using `python train.py`

### Error: "Features file not found"
- Solution: Run `add_advanced_features_mp.py` first

## 📚 Documentation

- [FEATURES_GUIDE.html](../FEATURES_GUIDE.html) - Complete features documentation
- [add_advanced_features_mp.py](../add_advanced_features_mp.py) - Feature generation script

## 📧 Contact

Author: haind
Project: Earthquake Sequence Mining
Date: 2025-03-25
