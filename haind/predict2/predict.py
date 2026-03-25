"""
Earthquake Prediction - Binary Classification Version
Time Model: Predicts probability of earthquake within 7 days
Mag Model: Predicts next earthquake magnitude

Cách dùng:
    python predict2/predict.py

Input:  predict2/input_events.json
Output: predict2/prediction_results.json

Author: haind
Date: 2025-03-25
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


def find_latest_models():
    """Find latest trained models"""
    import glob

    time_models = glob.glob(str(MODEL_DIR / 'time_model_*.pt'))
    mag_models = glob.glob(str(MODEL_DIR / 'mag_model_*.pt'))

    if not time_models:
        raise FileNotFoundError("Khong tim thay time model. Train truoc: python predict2/train_time.py")
    if not mag_models:
        raise FileNotFoundError("Khong tim thay mag model. Train truoc: python predict2/train_mag.py")

    time_models.sort(reverse=True)
    mag_models.sort(reverse=True)

    latest_time = time_models[0]
    latest_mag = mag_models[0]

    # Find corresponding scalers
    time_ts = os.path.basename(latest_time).replace('time_model_', '').replace('.pt', '')
    mag_ts = os.path.basename(latest_mag).replace('mag_model_', '').replace('.pt', '')

    time_scaler = str(MODEL_DIR / f'time_scaler_{time_ts}.pkl')
    mag_scaler = str(MODEL_DIR / f'mag_scaler_{mag_ts}.pkl')

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
    """Make predictions and display results (Binary Classification + Regression)"""

    # Check if ground truth is available
    has_ground_truth_time = 'target_quake_in_7days' in df.columns
    has_ground_truth_mag = 'target_next_mag' in df.columns
    has_ground_truth = has_ground_truth_time and has_ground_truth_mag

    # Prepare features
    X_time = prepare_features(df, TIME_FEATURES, time_scaler)
    X_mag = prepare_features(df, MAG_FEATURES, mag_scaler)

    n_predictions = len(X_time)

    results = []
    time_correct = 0
    mag_errors = []

    # Store original region codes before encoding
    original_regions = df['region_code'].tolist() if 'region_code' in df.columns else ['Unknown'] * len(df)

    for i in range(n_predictions):
        event_idx = i + SEQUENCE_LENGTH

        # Get input events
        input_events = df.iloc[event_idx-SEQUENCE_LENGTH:event_idx]
        last_input = input_events.iloc[-1]

        # Get region (use original value)
        region = original_regions[event_idx] if event_idx < len(original_regions) else 'Unknown'

        # Get ground truth if available
        gt_time_binary = df.iloc[event_idx]['target_quake_in_7days'] if has_ground_truth_time else None
        gt_mag = df.iloc[event_idx]['target_next_mag'] if has_ground_truth_mag else None

        # Predict
        with torch.no_grad():
            X_time_tensor = torch.FloatTensor(X_time[i:i+1]).to(device)
            time_logits = time_model(X_time_tensor).cpu().numpy()[0]
            time_proba = 1 / (1 + np.exp(-time_logits))  # Sigmoid

            X_mag_tensor = torch.FloatTensor(X_mag[i:i+1]).to(device)
            mag_pred = mag_model(X_mag_tensor).cpu().numpy()[0]

        # Get risk level for time prediction
        risk_level, risk_emoji = get_risk_level(time_proba)
        time_class = 1 if time_proba >= 0.5 else 0

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
            'sequence_number': i + 1,
            'region': str(region),
            'input_events_summary': {
                'start_idx': int(event_idx - SEQUENCE_LENGTH),
                'end_idx': int(event_idx - 1),
                'last_event': {
                    'magnitude': float(last_input.get('mag', 0)),
                    'depth_km': float(last_input.get('depth', 0))
                }
            },
            'prediction': {
                # Time model (binary classification)
                'quake_probability_7days': float(time_proba),
                'risk_level': risk_level,
                'risk_emoji': risk_emoji,
                'predicted_class': int(time_class),
                # Mag model (regression)
                'next_magnitude': float(mag_pred)
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
    print(f"\n{'='*110}")
    print(" BANG TONG HOP KET QUA DU DOAN ".center(110))
    print(f"{'='*110}\n")

    if has_ground_truth:
        # Table header with ground truth
        print(f"{'#':<4} {'Region':<12} {'Model':<12} {'Du Doan':<25} {'Thuc Te':<15} {'Dung/Sai':<10}")
        print("-" * 110)

        for i, result in enumerate(results):
            region = result['region']

            # Time model row (binary classification)
            prob_pct = result['prediction']['quake_probability_7days'] * 100
            gt_str = 'Có' if result['ground_truth_time']['quake_in_7days'] else 'Không'
            correct_str = '✓' if result['ground_truth_time']['correct'] else '✗'

            print(f"{i+1:<4} {region:<12} {'Time':<12} "
                  f"{prob_pct:>5.1f}% ({result['prediction']['risk_level']}) {result['prediction']['risk_emoji']:<2} "
                  f"{gt_str:<15} {correct_str:<10}")

            # Mag model row (regression)
            if 'ground_truth_mag' in result:
                mag_error = result['ground_truth_mag']['error']
                print(f"{'':<4} {'':<12} {'Mag':<12} "
                      f"{result['prediction']['next_magnitude']:>6.2f} "
                      f"{result['ground_truth_mag']['next_magnitude']:>6.2f} "
                      f"{mag_error:>6.3f}")
            else:
                print(f"{'':<4} {'':<12} {'Mag':<12} "
                      f"{result['prediction']['next_magnitude']:>25.2f}")
            print()
    else:
        # Table header without ground truth
        print(f"{'#':<4} {'Region':<12} {'Model':<12} {'Du Doan':<35}")
        print("-" * 70)

        for i, result in enumerate(results):
            region = result['region']

            # Time model row
            prob_pct = result['prediction']['quake_probability_7days'] * 100
            print(f"{i+1:<4} {region:<12} {'Time':<12} "
                  f"{prob_pct:>5.1f}% ({result['prediction']['risk_level']}) {result['prediction']['risk_emoji']:<2}")

            # Mag model row
            print(f"{'':<4} {'':<12} {'Mag':<12} "
                  f"{result['prediction']['next_magnitude']:>25.2f}")
            print()

    # Calculate summary metrics
    time_accuracy = time_correct / n_predictions if n_predictions > 0 else 0

    mag_mae = np.mean(mag_errors) if len(mag_errors) > 0 else None
    mag_rmse = np.sqrt(np.mean(np.array(mag_errors) ** 2)) if len(mag_errors) > 0 else None

    summary = {
        'time_accuracy': float(time_accuracy),
        'time_correct': int(time_correct),
        'time_total': int(n_predictions),
        'mag_mae': float(mag_mae) if mag_mae is not None else None,
        'mag_rmse': float(mag_rmse) if mag_rmse is not None else None,
        'n_samples': int(n_predictions)
    }

    # Print summary
    print(f"{'='*110}")
    print(f"{'TOM TAT':<20} Time Model - Binary Classification")
    print(f"{'Accuracy:':<20} {time_accuracy:>10.2%}")
    print(f"{'Correct:':<20} {time_correct:>10,} / {n_predictions:,}")
    print()
    print(f"{'TOM TAT':<20} Mag Model - Regression")
    if mag_mae is not None:
        print(f"{'MAE:':<20} {mag_mae:>10.3f}")
        print(f"{'RMSE:':<20} {mag_rmse:>10.3f}")
    print(f"{'='*110}\n")

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
