# Predict2 - Split LSTM Models for Earthquake Prediction

Tách multi-output LSTM model thành 2 model độc lập:
- **TimeLSTM**: Dự đoán thời gian đến trận động đất tiếp theo
- **MagLSTM**: Dự đoán độ lớn trận động đất tiếp theo

## Cấu trúc thư mục

```
predict2/
├── __init__.py
├── config.py              # Configuration (features, paths)
├── split_data.py          # Tách features_lstm.csv thành 2 bộ
├── train_time.py          # Training TimeLSTM model
├── train_mag.py           # Training MagLSTM model
├── demo.py                # Demo đơn giản
├── predict.py             # Prediction script đầy đủ
├── predict_from_json.py   # Prediction từ JSON input
├── sample_input.json      # Sample JSON input
├── models/
│   ├── __init__.py
│   ├── time_model.py      # TimeLSTM + TimeTrainer
│   └── mag_model.py       # MagLSTM + MagTrainer
├── data/
│   ├── __init__.py
│   ├── time_data.py       # TimeDataPreparer
│   └── mag_data.py        # MagDataPreparer
├── data_processed/        # Generated after split_data.py
│   ├── features_time.csv
│   └── features_mag.csv
└── models/                # Generated after training
    ├── time_model_*.pt
    ├── mag_model_*.pt
    ├── time_scaler_*.pkl
    ├── mag_scaler_*.pkl
    └── ...
```

## Luồng chạy hoàn chỉnh

### 1. Split Data
```bash
python predict2/split_data.py
```
Tạo `features_time.csv` và `features_mag.csv` từ `features_lstm.csv`

### 2. Train Time Model
```bash
# Test nhanh (5 epochs)
python predict2/train_time.py --test

# Training đầy đủ
python predict2/train_time.py --epochs 50 --batch-size 64 --hidden 128 64
```

### 3. Train Mag Model
```bash
# Test nhanh (5 epochs)
python predict2/train_mag.py --test

# Training đầy đủ
python predict2/train_mag.py --epochs 50 --batch-size 64 --hidden 128 64
```

### 4. Prediction

#### Cách 1: Demo đơn giản
```bash
python predict2/demo.py
```

#### Cách 2: Từ JSON input
```bash
# Sử dụng sample input
python predict2/predict_from_json.py predict2/sample_input.json --pretty

# Sử dụng JSON input của bạn
python predict2/predict_from_json.py your_events.json --pretty

# Lưu kết quả ra file
python predict2/predict_from_json.py your_events.json --output result.json
```

#### Cách 3: Advanced prediction
```bash
# Dùng models mới nhất
python predict2/predict.py --demo

# Chỉ định models cụ thể
python predict2/predict.py --time-model path/to/time_model.pt --mag-model path/to/mag_model.pt

# Lọc theo region
python predict2/predict.py --demo --region R257_114

# Số lượng samples
python predict2/predict.py --demo --n-samples 10
```

## Input JSON Format

```json
[
  {
    "time": "2025-03-25T10:23:45",
    "latitude": 35.6,
    "longitude": 140.2,
    "depth": 10.5,
    "mag": 4.2,
    "sig": 250,
    "mmi": 4.5,
    "cdi": 4.0,
    "felt": 15,
    "region_code": "R257_114",
    "is_aftershock": 0,
    "mainshock_mag": 5.5,
    "seismicity_density_100km": 45.2,
    "coulomb_stress_proxy": 0.3,
    "regional_b_value": 1.2,
    "sequence_id": "seq_001",
    "seq_position": 3,
    "is_seq_mainshock": 0,
    "seq_mainshock_mag": 5.5,
    "seq_length": 12,
    "time_since_seq_start_sec": 86400,
    "time_since_last_event": 3600,
    "time_since_last_M5": 7200,
    "interval_lag1": 3000,
    "interval_lag2": 3600,
    "interval_lag3": 4200,
    "interval_lag4": 4800,
    "interval_lag5": 5400
  }
]
```

**Yêu cầu tối thiểu:**
- Ít nhất 5 events (SEQUENCE_LENGTH)
- Chứa tất cả features trong TIME_FEATURES và MAG_FEATURES

## Features Selection

### Time Model Features (23 features)
```
Original: mag, depth, sig, region_code
Core: is_aftershock, mainshock_mag, seismicity_density_100km, coulomb_stress_proxy
Sequence: sequence_id, seq_position, is_seq_mainshock, seq_mainshock_mag, seq_length, time_since_seq_start_sec
LSTM Temporal: time_since_last_event, time_since_last_M5, interval_lag1-5
```

### Mag Model Features (25 features)
```
Original: mag, depth, latitude, longitude, sig, mmi, cdi, felt, region_code
Core: is_aftershock, mainshock_mag, seismicity_density_100km, coulomb_stress_proxy, regional_b_value
Sequence: seq_position, is_seq_mainshock, seq_mainshock_mag, seq_length, time_since_seq_start_sec
LSTM Temporal: time_since_last_event, time_since_last_M5, interval_lag1-5
```

## Output Format

### Prediction Result
```json
{
  "sequence_start_idx": 0,
  "sequence_end_idx": 4,
  "prediction_for_idx": 5,
  "input_events": [...],
  "prediction": {
    "time_to_next_seconds": 9234.5,
    "time_to_next_minutes": 153.9,
    "time_to_next_hours": 2.6,
    "time_to_next_days": 0.1,
    "next_magnitude": 4.23
  },
  "actual": {
    "time_to_next_seconds": 8900.0,
    "next_magnitude": 4.1
  }
}
```

## Training Progress Display

```
================================================================================
                              TIME MODEL TRAINING
================================================================================
Device: cuda
Epochs: 50
Batch Size: 64
Train Samples: 121,903
Val Samples: 15,237
Learning Rate: 0.001
Patience: 15
================================================================================

Epoch   Train Loss    Val Loss     Train MAE    Val MAE     LR         Time
--------------------------------------------------------------------------------
1       1234567.8     1198765.4    234.5        228.7        1.00e-03   45.2s *
      Val Loss: ↓0, Val MAE: ↓0s
2       1198234.5     1156432.1    226.8        221.3        1.00e-03   44.8s *
      Val Loss: ↓42333.3, Val MAE: ↓7.4s
...
```

## Lưu ý

1. **Train models trước khi predict**: Chạy `train_time.py` và `train_mag.py` trước
2. **Scaler**: Models cần scaler tương ứng để prediction chính xác
3. **Sequence Length**: Cần ít nhất 5 events liên tiếp để predict
4. **Features**: JSON input phải chứa đầy đủ features yêu cầu
