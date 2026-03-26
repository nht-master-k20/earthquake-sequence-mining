# Earthquake Prediction System - Split LSTM Models

Hệ thống dự báo động đất sử dụng 2 mô hình LSTM độc lập:
- **TimeLSTM**: Dự đoán xác suất có động đất trong 7 ngày tới (Binary Classification)
- **MagLSTM**: Dự báo độ lớn magnitude của động đất tiếp theo (Regression)

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Workflow Overview](#workflow-overview)
3. [Step 1: Add Features](#step-1-add-features)
4. [Step 2: Split Data by Model](#step-2-split-data-by-model)
5. [Step 3: Train Models](#step-3-train-models)
6. [Step 4: Test & Evaluate](#step-4-test--evaluate)
7. [Step 5: Predict](#step-5-predict)
8. [Step 6: Demo](#step-6-demo)
9. [Data Format](#data-format)
10. [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# Full pipeline (from parent directory)
cd /path/to/earthquake-sequence-mining

# Step 1: Add features (feature engineering)
python add_advanced_features_mp.py

# Step 2: Split data by model
python predict2/split_data.py

# Step 3: Train models
python predict2/train_time.py --epochs 50 --batch-size 64 --hidden 128 64
python predict2/train_mag.py --epochs 50 --batch-size 64 --hidden 128 64

# Step 4: Evaluate on test set
python predict2/evaluate.py

# Step 5: Make predictions
python predict2/predict.py

# Step 6: Launch demo
python predict2/demo.py
# Open browser: http://localhost:5000
```

---

## Workflow Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAW DATA (dongdat.csv)                      │
│                    3.1M earthquake events                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: ADD FEATURES (Feature Engineering)                     │
│  → 31 features (Original + Core + Sequence + Temporal + Targets)│
│  → Script: add_advanced_features_mp.py                           │
│  → Output: features_lstm.csv                                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: SPLIT DATA BY MODEL                                     │
│  → features_time.csv (23 features) → TimeLSTM                     │
│  → features_mag.csv (25 features) → MagLSTM                      │
│  → Script: predict2/split_data.py                                │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         ▼                               ▼
┌─────────────────────┐       ┌─────────────────────┐
│  STEP 3A: TRAIN     │       │  STEP 3B: TRAIN     │
│  TimeLSTM Model     │       │  MagLSTM Model      │
│  (Binary Class.)    │       │  (Regression)       │
│  → train_time.py    │       │  → train_mag.py     │
└─────────┬───────────┘       └─────────┬───────────┘
          │                           │
          └───────────┬───────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: TEST & EVALUATE                                         │
│  → ROC-AUC, Brier Score, Precision/Recall, MAE, RMSE            │
│  → Script: predict2/evaluate.py                                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: PREDICT                                                 │
│  → Input: input_events.json (6+ events)                          │
│  → Output: prediction_results.json                               │
│  → Script: predict2/predict.py                                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 6: DEMO                                                    │
│  → Flask web app with 5 seismic zones                           │
│  → http://localhost:5000                                         │
│  → Script: predict2/demo.py                                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Add Features

### Purpose
Extract and compute features from raw earthquake data

### Input/Output
- **Input:** `dongdat.csv` (3.1M events from USGS)
- **Output:** `features_lstm.csv` (31 features)

### Features

| Category | Count | Features |
|----------|-------|----------|
| **Original** | 10 | time, mag, depth, latitude, longitude, sig, mmi, cdi, felt, region_code |
| **Core** | 5 | is_aftershock, mainshock_mag, seismicity_density_100km, coulomb_stress_proxy, regional_b_value |
| **Sequence** | 6 | sequence_id, seq_position, is_seq_mainshock, seq_mainshock_mag, seq_length, time_since_seq_start_sec |
| **Temporal** | 5 | time_since_last_event, time_since_last_M5, interval_lag1-5 |
| **Targets** | 3 | target_time_to_next, target_next_mag, target_next_mag_binary |

### Run

```bash
# From parent directory
python add_advanced_features_mp.py
```

### Configuration

```python
# File: add_advanced_features_mp.py
START_YEAR = 2000
END_YEAR = 2026
SEQUENCE_LENGTH = 5
min_events = 500  # Minimum events per region
```

**Detailed Guide:** See [FEATURES_GUIDE.html](../FEATURES_GUIDE.html)

---

## Step 2: Split Data by Model

### Purpose
Split `features_lstm.csv` into separate datasets for TimeLSTM and MagLSTM

### Input/Output
- **Input:** `features_lstm.csv`
- **Outputs:**
  - `predict2/data_processed/features_time.csv` (23 input features + target_quake_in_7days)
  - `predict2/data_processed/features_mag.csv` (25 input features + target_next_mag)

### Feature Sets

#### TimeLSTM Features (23)
```python
TIME_FEATURES = [
    # Original (4)
    'mag', 'depth', 'sig', 'region_code',
    # Core (4)
    'is_aftershock', 'mainshock_mag', 'seismicity_density_100km', 'coulomb_stress_proxy',
    # Sequence (6)
    'sequence_id', 'seq_position', 'is_seq_mainshock', 'seq_mainshock_mag',
    'seq_length', 'time_since_seq_start_sec',
    # Temporal (5)
    'time_since_last_event', 'time_since_last_M5',
    'interval_lag1', 'interval_lag2', 'interval_lag3', 'interval_lag4', 'interval_lag5'
]
```

#### MagLSTM Features (25)
```python
MAG_FEATURES = [
    # Original (9)
    'mag', 'depth', 'latitude', 'longitude', 'sig', 'mmi', 'cdi', 'felt', 'region_code',
    # Core (5)
    'is_aftershock', 'mainshock_mag', 'seismicity_density_100km',
    'coulomb_stress_proxy', 'regional_b_value',
    # Sequence (5)
    'seq_position', 'is_seq_mainshock', 'seq_mainshock_mag',
    'seq_length', 'time_since_seq_start_sec',
    # Temporal (5)
    'time_since_last_event', 'time_since_last_M5',
    'interval_lag1', 'interval_lag2', 'interval_lag3', 'interval_lag4', 'interval_lag5'
]
```

### Run

```bash
cd predict2
python split_data.py
```

### Output

```
Time Dataset:
  Features: 23
  Target: target_quake_in_7days (binary)
  Samples: ~750,000
  Positive (quake in 7 days): ~45%

Mag Dataset:
  Features: 25
  Target: target_next_mag
  Samples: ~750,000
  Mean Magnitude: ~4.5
```

---

## Step 3: Train Models

### 3A. Train TimeLSTM (Binary Classification)

#### Purpose
Predict probability of earthquake in next 7 days

#### Run

```bash
cd predict2

# Full training
python train_time.py --epochs 50 --batch-size 64 --hidden 128 64

# Custom parameters
python train_time.py --epochs 100 --hidden 256 128 --lr 0.0001 --dropout 0.4
```

#### Architecture

```
Input: (batch, 5, 23)  # 5 timesteps, 23 features
    ↓
LSTM Layer 1: 128 units
    ↓
LSTM Layer 2: 64 units
    ↓
Dropout: 0.3
    ↓
Output: 1 unit (logits)
    ↓
Sigmoid → Probability (0-1)
```

#### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 50 | Number of training epochs |
| `--batch-size` | 64 | Batch size |
| `--hidden` | 128 64 | LSTM hidden units |
| `--lr` | 0.001 | Learning rate |
| `--dropout` | 0.3 | Dropout rate |
| `--subset-ratio` | 1.0 | Data ratio (1.0=full, 0.1=10%) |
| `--patience` | 15 | Early stopping patience |

#### Output

```
predict2/models/
├── time_model_20260326_120000.pt       # Model weights + scaler
├── time_scaler_20260326_120000.pkl     # Feature scaler
├── time_region_encoder_*.pkl           # Region encoder
├── time_history_*.json                 # Training history
└── time_metadata_*.json                # Model metadata
```

---

### 3B. Train MagLSTM (Regression)

#### Purpose
Predict magnitude of next earthquake

#### Run

```bash
cd predict2

# Full training
python train_mag.py --epochs 50 --batch-size 64 --hidden 128 64

# Custom parameters
python train_mag.py --epochs 100 --hidden 256 128 --lr 0.0001
```

#### Architecture

```
Input: (batch, 5, 25)  # 5 timesteps, 25 features
    ↓
LSTM Layer 1: 128 units
    ↓
LSTM Layer 2: 64 units
    ↓
Dropout: 0.3
    ↓
Output: 1 unit (magnitude)
```

#### Parameters

Same as TimeLSTM (see table above)

#### Output

```
predict2/models/
├── mag_model_20260326_120000.pt        # Model weights + scaler
├── mag_scaler_20260326_120000.pkl      # Feature scaler
├── mag_region_encoder_*.pkl            # Region encoder
├── mag_history_*.json                  # Training history
└── mag_metadata_*.json                 # Model metadata
```

---

## Step 4: Test & Evaluate

### Purpose
Evaluate models on test set with comprehensive metrics

### Run

```bash
cd predict2
python evaluate.py
```

### Metrics

#### TimeLSTM (Binary Classification)
- **ROC-AUC**: Area under ROC curve
- **Brier Score**: Probability prediction error (lower is better)
- **Accuracy**: Classification accuracy at threshold 0.5
- **Precision/Recall**: At different thresholds (0.3, 0.5, 0.7)
- **F1 Score**: Harmonic mean of precision and recall
- **Baseline**: Poisson process comparison

#### MagLSTM (Regression)
- **MAE**: Mean Absolute Error (magnitude units)
- **RMSE**: Root Mean Squared Error
- **MSE**: Mean Squared Error

### Output

```
predict2/dashboard/plots/
├── roc_curve_*.png                    # ROC curve (TimeLSTM)
├── precision_recall_curve_*.png       # PR curve (TimeLSTM)
└── test_metrics_table_*.png           # Combined metrics table
```

### Expected Results

| Metric | TimeLSTM | MagLSTM |
|--------|----------|---------|
| ROC-AUC | > 0.75 | - |
| Brier Score | < 0.20 | - |
| Accuracy | > 0.70 | - |
| MAE | - | < 0.4 |
| RMSE | - | < 0.6 |

---

## Step 5: Predict

### Purpose
Make predictions on new earthquake sequences

### Input Format

Create `predict2/input_events.json`:

```json
[
  {
    "time": "2011-03-09T14:30:00",
    "mag": 5.4,
    "depth": 30.0,
    "latitude": 37.4470,
    "longitude": 140.9840,
    "sig": 251189,
    "mmi": 5,
    "cdi": 5,
    "felt": 120,
    "region_code": "R37_141",
    "is_aftershock": 0,
    "mainshock_mag": 5.9,
    "seismicity_density_100km": 40.5,
    "coulomb_stress_proxy": 150000000,
    "regional_b_value": 0.85,
    "sequence_id": 12345,
    "seq_position": 1,
    "is_seq_mainshock": 1,
    "seq_mainshock_mag": 5.4,
    "seq_length": 50,
    "time_since_seq_start_sec": 2592000,
    "time_since_last_event": 3600,
    "time_since_last_M5": 7200,
    "interval_lag1": 3000,
    "interval_lag2": 3600,
    "interval_lag3": 4200,
    "interval_lag4": 4800,
    "interval_lag5": 5400,
    "target_quake_in_7days": 1,
    "target_next_mag": 5.9
  }
]
```

**Minimum:** 6 events (SEQUENCE_LENGTH + 1)

### Run

```bash
cd predict2
python predict.py
```

### Output

`predict2/prediction_results.json`:

```json
{
  "timestamp": "2026-03-26 12:00:00",
  "n_predictions": 1,
  "summary": {
    "avg_m5_probability": 0.68,
    "avg_magnitude_prediction": 5.2,
    "n_predictions": 1
  },
  "predictions": [
    {
      "sequence_number": 1,
      "region": "R37_141",
      "input_events_summary": {
        "start_idx": 0,
        "end_idx": 5,
        "n_events_used": 6,
        "last_event": {
          "magnitude": 5.4,
          "depth_km": 30.0
        }
      },
      "prediction": {
        "quake_probability_7days": 0.85,
        "predicted_class": 1,
        "next_magnitude": 5.2,
        "m5_probability_7days": 0.68,
        "m5_predicted_class": 1
      },
      "ground_truth_time": {
        "quake_in_7days": 1,
        "correct": 1
      },
      "ground_truth_mag": {
        "next_magnitude": 5.1,
        "error": 0.1
      }
    }
  ]
}
```

### Prediction Explanation

| Field | Description |
|-------|-------------|
| `quake_probability_7days` | Probability of ANY earthquake in 7 days (0-1) |
| `m5_probability_7days` | Probability of M5.0+ earthquake in 7 days (0-1) |
| `next_magnitude` | Predicted magnitude of next earthquake |
| `predicted_class` | 1 if probability ≥ 0.5, else 0 |

---

## Step 6: Demo

### Purpose
Launch Flask web app with 5 seismic zones for interactive prediction

### Seismic Zones

| Zone | Location | Mainshock | File |
|------|----------|-----------|------|
| Japan | Tohoku Region, Honshu | M5.9 | `input_events_japan.json` |
| Philippines | Mindanao Region | M5.5 | `input_events_philippines.json` |
| Chile | Central Chile | M5.3 | `input_events_chile.json` |
| Indonesia | Northern Sumatra | M5.0 | `input_events_indonesia__sumatra.json` |
| USA | Baja California | M5.3 | `input_events_usa__california.json` |

### Run

```bash
cd predict2
python demo.py
```

Open browser: **http://localhost:5000**

### Features

1. **Zone Selection** - Choose from 5 seismic zones
2. **Event Chart** - Bar chart showing magnitude history
3. **Real-time Prediction** - See prediction results immediately
4. **Simulation Mode** - Add events one by one and watch predictions update

### API Endpoints

```
GET  /                          # Demo UI
GET  /api/zones                 # List all zones
POST /api/predict               # Run prediction for zone
GET  /api/events/<zone_id>      # Get events for zone
POST /api/simulate/start        # Start simulation mode
POST /api/simulate/next         # Add next event in simulation
POST /api/simulate/reset        # Reset simulation
```

---

## Data Format

### Input Requirements

| Requirement | Value |
|-------------|-------|
| Min events | 6 (SEQUENCE_LENGTH + 1) |
| Format | JSON array |
| Features | All TIME_FEATURES or MAG_FEATURES |
| Sort order | Chronological (time ASC) |

### Complete Feature List

See [FEATURES_GUIDE.html](../FEATURES_GUIDE.html) for detailed feature documentation.

---

## Project Structure

```
haind/
├── add_advanced_features_mp.py      # Step 1: Feature engineering
├── features_lstm.csv                # Output: 31 features
├── FEATURES_GUIDE.html              # Feature documentation
│
├── predict2/
│   ├── config.py                    # Configuration
│   ├── split_data.py                # Step 2: Split by model
│   ├── data_processed/
│   │   ├── features_time.csv        # Time model dataset
│   │   └── features_mag.csv         # Mag model dataset
│   │
│   ├── train_time.py                # Step 3A: Train TimeLSTM
│   ├── train_mag.py                 # Step 3B: Train MagLSTM
│   ├── models/
│   │   ├── time_model_*.pt          # Trained TimeLSTM
│   │   ├── mag_model_*.pt           # Trained MagLSTM
│   │   ├── time_scaler_*.pkl
│   │   └── mag_scaler_*.pkl
│   │
│   ├── evaluate.py                  # Step 4: Test & evaluate
│   ├── dashboard/
│   │   └── plots/                   # Evaluation plots
│   │
│   ├── predict.py                   # Step 5: Make predictions
│   ├── input_events.json            # Prediction input
│   ├── prediction_results.json      # Prediction output
│   │
│   ├── demo.py                      # Step 6: Flask demo
│   ├── templates/
│   │   └── demo.html                # Demo UI
│   └── input_events_*.json          # 5 seismic zones
```

---

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch numpy pandas scikit-learn flask tqdm numba scipy matplotlib seaborn
```

### Requirements

- Python 3.8+
- 8GB+ RAM
- GPU (NVIDIA CUDA) recommended for training

---

## Troubleshooting

### Error: "No module named 'torch'"
```bash
source venv/bin/activate
pip install torch
```

### Error: "features_lstm.csv not found"
```bash
# Run Step 1 first
python add_advanced_features_mp.py
```

### Error: "features_time.csv not found"
```bash
# Run Step 2 first
cd predict2
python split_data.py
```

### Error: "No time model found"
```bash
# Train models first
cd predict2
python train_time.py --epochs 50 --batch-size 64 --hidden 128 64
python train_mag.py --epochs 50 --batch-size 64 --hidden 128 64
```

### Error: "Port 5000 already in use"
```bash
# Kill process on port 5000
lsof -ti:5000 | xargs kill -9
```

---

## References

- **FEATURES_GUIDE.html** - Detailed feature documentation
- **USGS Earthquake Hazards Program** - Raw data source
- **Gutenberg-Richter Law** - b-value calculation
- **Coulomb Stress Transfer** - Stress accumulation theory

---

**Last Updated:** 2026-03-26
**Author:** haind
**Version:** 3.0 (Split LSTM + M5+ Prediction)
