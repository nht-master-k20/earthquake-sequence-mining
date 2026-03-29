"""
Earthquake M5+ Prediction (Within 7 Days)
Combines Time Model (quake probability) + Mag Model (magnitude prediction)

Logic: M5+ probability = time probability IF predicted magnitude >= 5.0

Cách dùng:
    python predict2/predict.py

Input:  predict2/input_events.json
Output: predict2/prediction_results.json

Author: haind
Date: 2025-03-25
Updated: 2026-03-26 (M5+ prediction)
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import SEQUENCE_LENGTH, TIME_FEATURES, MAG_FEATURES, MODEL_DIR
from models.time_model import TimeLSTM
from models.mag_model import MagLSTM


# Input/Output files (cố định)
INPUT_FILE = os.path.join(os.path.dirname(__file__), 'input_events.json')
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), 'prediction_results.json')


def get_device():
    """Get device for prediction"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_risk_level(probability):
    """Convert probability to risk level"""
    if probability >= 0.7:
        return 'HIGH', '🔴'
    elif probability >= 0.4:
        return 'MEDIUM', '🟡'
    else:
        return 'LOW', '🟢'


def get_m5_risk_level(probability):
    """Convert M5+ probability to risk level (more conservative)"""
    if probability >= 0.5:
        return 'CRITICAL', '🔴'
    elif probability >= 0.2:
        return 'HIGH', '🟠'
    elif probability >= 0.1:
        return 'MEDIUM', '🟡'
    else:
        return 'LOW', '🟢'


def find_latest_models():
    """Find latest trained models (supports both full and subset models)"""
    import glob

    # Find all time and mag models (including subset models)
    time_models = glob.glob(str(MODEL_DIR / 'time_model*.pt'))
    mag_models = glob.glob(str(MODEL_DIR / 'mag_model*.pt'))

    # Filter out checkpoint/incomplete files
    time_models = [m for m in time_models if os.path.getsize(m) > 1000]
    mag_models = [m for m in mag_models if os.path.getsize(m) > 1000]

    if not time_models:
        raise FileNotFoundError("Khong tim thay time model. Train truoc: python predict2/train_time.py")
    if not mag_models:
        raise FileNotFoundError("Khong tim thay mag model. Train truoc: python predict2/train_mag.py")

    time_models.sort(reverse=True)
    mag_models.sort(reverse=True)

    latest_time = time_models[0]
    latest_mag = mag_models[0]

    # Find corresponding scalers
    time_basename = os.path.basename(latest_time).replace('time_model_', '').replace('.pt', '')
    mag_basename = os.path.basename(latest_mag).replace('mag_model_', '').replace('.pt', '')

    time_scaler = str(MODEL_DIR / f'time_scaler_{time_basename}.pkl')
    mag_scaler = str(MODEL_DIR / f'mag_scaler_{mag_basename}.pkl')

    if not os.path.exists(time_scaler):
        time_scaler = None
    if not os.path.exists(mag_scaler):
        mag_scaler = None

    return latest_time, latest_mag, time_scaler, mag_scaler


def load_models(time_path, mag_path, time_scaler_path, mag_scaler_path, device):
    """Load models and scalers"""
    # Load time model
    time_checkpoint = torch.load(time_path, map_location=device, weights_only=False)
    time_model = TimeLSTM(
        n_features=time_checkpoint['n_features'],
        lstm_hidden=time_checkpoint['lstm_hidden']
    )
    time_model.load_state_dict(time_checkpoint['model_state_dict'])
    time_model = time_model.to(device)
    time_model.eval()

    # Load mag model
    mag_checkpoint = torch.load(mag_path, map_location=device, weights_only=False)
    mag_model = MagLSTM(
        n_features=mag_checkpoint['n_features'],
        lstm_hidden=mag_checkpoint['lstm_hidden']
    )
    mag_model.load_state_dict(mag_checkpoint['model_state_dict'])
    mag_model = mag_model.to(device)
    mag_model.eval()

    # Load scalers
    time_scaler = None
    mag_scaler = None

    if time_scaler_path and os.path.exists(time_scaler_path):
        with open(time_scaler_path, 'rb') as f:
            time_scaler = pickle.load(f)

    if mag_scaler_path and os.path.exists(mag_scaler_path):
        with open(mag_scaler_path, 'rb') as f:
            mag_scaler = pickle.load(f)

    return time_model, mag_model, time_scaler, mag_scaler


def load_input():
    """Load input_events.json"""
    if not os.path.exists(INPUT_FILE):
        print(f"LOI: Khong tim thay {INPUT_FILE}")
        return None

    with open(INPUT_FILE, 'r') as f:
        events = json.load(f)

    if len(events) < SEQUENCE_LENGTH:
        print(f"LOI: Can it nhat {SEQUENCE_LENGTH} events, chi co {len(events)}")
        return None

    return pd.DataFrame(events)


def validate_features(df):
    """Validate input has required features"""
    missing_time = [f for f in TIME_FEATURES if f not in df.columns]
    missing_mag = [f for f in MAG_FEATURES if f not in df.columns]

    if missing_time:
        print(f"LOI: Thieu Time Model features: {missing_time}")
        return False

    if missing_mag:
        print(f"LOI: Thieu Mag Model features: {missing_mag}")
        return False

    return True


def prepare_features(df, features, scaler=None):
    """Prepare features for prediction"""
    from sklearn.preprocessing import LabelEncoder

    df = df.copy()
    feature_cols = features.copy()

    # Encode region_code - add new column instead of replacing
    region_encoder = None
    if 'region_code' in feature_cols:
        region_encoder = LabelEncoder()
        # Keep original region_code, add encoded version
        df['region_encoded'] = region_encoder.fit_transform(df['region_code'])

    # Prepare sequences
    sequences = []

    for i in range(SEQUENCE_LENGTH, len(df)):
        seq_data = df.iloc[i-SEQUENCE_LENGTH:i]

        seq_features = []
        for feat in feature_cols:
            if feat == 'region_code':
                # Use encoded version
                seq_features.append(seq_data['region_encoded'].values)
            else:
                seq_features.append(seq_data[feat].values)

        seq_array = np.column_stack(seq_features)
        sequences.append(seq_array)

    X = np.array(sequences)

    # Scale
    if scaler is not None:
        original_shape = X.shape
        X_2d = X.reshape(-1, X.shape[-1])
        X_scaled = scaler.transform(X_2d)
        X = X_scaled.reshape(original_shape)

    return X


def predict_and_show(time_model, mag_model, time_scaler, mag_scaler, df, device):
    """Make predictions and display results (Binary Classification + Regression)

    CÁCH 1: Chỉ dùng 5 events gần nhất → 1 dự đoán
    """

    # Check if ground truth is available
    has_ground_truth_time = 'target_quake_in_7days' in df.columns
    has_ground_truth_mag = 'target_next_mag' in df.columns
    has_ground_truth = has_ground_truth_time and has_ground_truth_mag

    # CHỈ LẤY EVENTS GẦN NHẤT ĐỂ TẠO 1 SEQUENCE
    # Cần ít nhất SEQUENCE_LENGTH + 1 events để tạo 1 sequence
    n_events_needed = SEQUENCE_LENGTH + 1

    if len(df) >= n_events_needed:
        # Lấy n_events_needed events gần nhất
        df_last = df.tail(n_events_needed).reset_index(drop=True)
        start_idx = len(df) - n_events_needed
    else:
        print(f"LOI: Can it nhat {n_events_needed} events, chi co {len(df)}")
        return [], {}

    # Prepare features (chỉ 1 sequence)
    X_time = prepare_features(df_last, TIME_FEATURES, time_scaler)
    X_mag = prepare_features(df_last, MAG_FEATURES, mag_scaler)

    # Chỉ có 1 prediction
    n_predictions = 1

    results = []
    time_correct = 0
    mag_errors = []

    # Get input events (5 events gần nhất)
    input_events = df_last
    last_input = input_events.iloc[-1]

    # Get region (use original value)
    region = last_input['region_code'] if 'region_code' in last_input else 'Unknown'

    # Get ground truth from last event if available
    gt_time_binary = last_input['target_quake_in_7days'] if has_ground_truth_time else None
    gt_mag = last_input['target_next_mag'] if has_ground_truth_mag else None

    # Predict
    with torch.no_grad():
        X_time_tensor = torch.FloatTensor(X_time[0:1]).to(device)
        time_logits = time_model(X_time_tensor).cpu().numpy()[0]
        time_proba = 1 / (1 + np.exp(-time_logits))  # Sigmoid

        X_mag_tensor = torch.FloatTensor(X_mag[0:1]).to(device)
        mag_pred = mag_model(X_mag_tensor).cpu().numpy()[0]

    # Calculate M5+ probability (combine time and mag)
    # Sigmoid smoothing: mag càng gần 5.0 thì xác suất càng cao
    # P(M5+) = P(time) * sigmoid(mag_pred - 5.0)
    mag_proba = 1 / (1 + np.exp(-(mag_pred - 5.0)))  # Sigmoid centered at 5.0
    m5_proba = time_proba * mag_proba

    time_class = 1 if time_proba >= 0.5 else 0
    m5_class = 1 if m5_proba >= 0.5 else 0

    # Calculate errors if ground truth available
    mag_error = None
    if has_ground_truth_mag and gt_mag is not None:
        mag_error = abs(mag_pred - gt_mag)
        mag_errors.append(mag_error)

    if has_ground_truth_time and gt_time_binary is not None:
        if time_class == gt_time_binary:
            time_correct += 1

    # Build result object
    result = {
        'sequence_number': 1,
        'region': str(region),
        'input_events_summary': {
            'start_idx': int(start_idx),
            'end_idx': int(len(df) - 1),
            'n_events_used': n_events_needed,
            'last_event': {
                'magnitude': float(last_input.get('mag', 0)),
                'depth_km': float(last_input.get('depth', 0))
            }
        },
        'prediction': {
            # Time model (binary classification - any quake)
            'quake_probability_7days': float(time_proba),
            'predicted_class': int(time_class),
            # Mag model (regression)
            'next_magnitude': float(mag_pred),
            # M5+ prediction (combined)
            'm5_probability_7days': float(m5_proba),
            'm5_predicted_class': int(m5_class)
        }
    }

    # Add ground truth if available
    if has_ground_truth_time and gt_time_binary is not None:
        result['ground_truth_time'] = {
            'quake_in_7days': int(gt_time_binary),
            'correct': int(time_class == gt_time_binary)
        }

    if has_ground_truth_mag and gt_mag is not None:
        result['ground_truth_mag'] = {
            'next_magnitude': float(gt_mag),
            'error': float(mag_error) if mag_error is not None else None
        }

    results.append(result)

    # Display final table
    print(f"\n{'='*90}")
    print(" DU BAO TRAN DAO DAT M5+ TRONG 7 NGAY TOI ".center(90))
    print(f"{'='*90}\n")

    # Table header
    print(f"{'#':<4} {'Region':<15} {'Model':<20} {'Du Doan':<25}")
    print("-" * 70)

    for i, result in enumerate(results):
        region = result['region']

        # M5+ model row (main prediction)
        m5_prob_pct = result['prediction']['m5_probability_7days'] * 100
        print(f"{i+1:<4} {region:<15} {'M5+ (7 ngay)':<20} {m5_prob_pct:>5.1f}%")

        # Mag prediction row
        mag_pred = result['prediction']['next_magnitude']
        print(f"{'':<4} {'':<15} {'Mag du bao':<20} {mag_pred:>6.2f}")
        print()

    # Calculate summary metrics
    avg_m5_proba = np.mean([r['prediction']['m5_probability_7days'] for r in results])
    avg_mag_pred = np.mean([r['prediction']['next_magnitude'] for r in results])

    summary = {
        'avg_m5_probability': float(avg_m5_proba),
        'avg_magnitude_prediction': float(avg_mag_pred),
        'n_predictions': int(n_predictions)
    }

    # Print summary
    print(f"{'='*90}")
    print(f"{'TOM TAT':<20} DU BAO TRAN DAO DAT M5+ (7 NGAY)")
    print(f"{'Events su dung:':<20} {n_events_needed:>10,} (gan nhat)")
    print(f"{'Xac suat M5+:':<20} {avg_m5_proba:>10.2%}")
    print()
    print(f"{'TOM TAT':<20} MAG DU BAO")
    print(f"{'Magnitude:':<20} {avg_mag_pred:>10.2f}")
    print(f"{'='*90}\n")

    return results, summary


def save_results(results, summary):
    """Save results to prediction_results.json"""
    output = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_predictions': len(results),
        'summary': summary,
        'predictions': results
    }

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Ket qua da luu vao: {OUTPUT_FILE}")


def main():
    """Main function"""
    # Get device
    device = get_device()

    # Load input
    df = load_input()
    if df is None:
        return

    # Validate features
    if not validate_features(df):
        return

    # Load models
    time_path, mag_path, time_scaler_path, mag_scaler_path = find_latest_models()
    time_model, mag_model, time_scaler, mag_scaler = load_models(
        time_path, mag_path, time_scaler_path, mag_scaler_path, device
    )

    # Make predictions
    results, summary = predict_and_show(time_model, mag_model, time_scaler, mag_scaler, df, device)

    # Save results
    save_results(results, summary)


if __name__ == "__main__":
    main()
