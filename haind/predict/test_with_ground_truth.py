"""
Test với Ground Truth đúng
Dùng target values từ data (target_time_to_next, target_next_mag)
"""

import argparse
import json
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import torch
import numpy as np
from pathlib import Path

from config import MODEL_DIR, SEQUENCE_LENGTH
from data_preparer import DataPreparer
from model_pytorch import EarthquakeLSTM, load_scaler


def get_device(device_arg='auto'):
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    return torch.device('cpu')


def find_latest_model(region=None):
    models = list(MODEL_DIR.glob('model*.pt'))
    if not models:
        raise FileNotFoundError(f"Không tìm thấy model trong {MODEL_DIR}")

    if region:
        models = [m for m in models if f'_{region}_' in str(m) or f'_all_regions' in str(m)]

    latest_model = max(models, key=lambda x: x.stat().st_mtime)

    model_name = latest_model.stem
    scaler_name = model_name.replace('model_', 'scaler_') + '.pkl'
    scaler_path = MODEL_DIR / scaler_name

    if not scaler_path.exists():
        scalers = list(MODEL_DIR.glob('scaler*.pkl'))
        if scalers:
            scaler_path = max(scalers, key=lambda x: x.stat().st_mtime)

    return latest_model, scaler_path


def load_model(model_path, scaler_path, device='cpu'):
    print(f"\n{'='*60}")
    print(" LOAD MODEL")
    print(f"{'='*60}")
    print(f"Model: {model_path.name}")

    checkpoint = torch.load(model_path, map_location=device)

    n_features = checkpoint['n_features']
    lstm_hidden = checkpoint['lstm_hidden']

    model = EarthquakeLSTM(n_features=n_features, lstm_hidden=lstm_hidden)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"  Features: {n_features}")
    print(f"  LSTM Hidden: {lstm_hidden}")

    scaler = load_scaler(scaler_path)

    return model, scaler, checkpoint


def main():
    parser = argparse.ArgumentParser(description='Test với ground truth đúng')
    parser.add_argument('--region', type=str, default='R257_114')
    parser.add_argument('--n-samples', type=int, default=10,
                       help='Số mẫu test (default: 10)')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(" TEST GROUND TRUTH ĐÚNG")
    print(f"{'='*60}")

    device = get_device('auto')
    print(f"Device: {device}")

    # Load data
    print(f"\n{'='*60}")
    print(" LOAD DATA")
    print(f"{'='*60}")

    preparer = DataPreparer()
    preparer.load_data()

    region_data = preparer.filter_by_region(args.region, min_events=100)
    print(f"Region: {args.region}")
    print(f"Events: {len(region_data):,}")

    # Prepare sequences (có targets)
    X, y = preparer.prepare_sequences(region_data, for_training=True)

    print(f"Sequences: {len(X):,}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    # Load model
    model_path, scaler_path = find_latest_model(args.region)
    model, scaler, checkpoint = load_model(model_path, scaler_path, device)

    # Scale X
    X_2d = X.reshape(-1, X.shape[-1])
    X_scaled = scaler.transform(X_2d)
    X_scaled = X_scaled.reshape(X.shape)

    # Test n_samples cuối
    print(f"\n{'='*60}")
    print(f" TEST {args.n_samples} MẪU CUỐI")
    print(f"{'='*60}")

    results = []

    for i in range(len(X_scaled) - args.n_samples, len(X_scaled)):
        # Get input and ground truth
        X_input = X_scaled[i:i+1]
        y_true = y[i]  # [time_to_next, next_mag, next_mag_binary]

        # Predict
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_input).to(device)
            time_pred, mag_pred, binary_pred = model(X_tensor)

        time_pred_val = time_pred.cpu().numpy()[0]
        mag_pred_val = mag_pred.cpu().numpy()[0]

        # Ground truth values
        time_true = y_true[0]  # target_time_to_next
        mag_true = y_true[1]   # target_next_mag
        binary_true = y_true[2]  # target_next_mag_binary

        # Calculate errors
        time_error = abs(time_pred_val - time_true)
        mag_error = abs(mag_pred_val - mag_true)

        results.append({
            'sample': i,
            'time_true': float(time_true),
            'time_pred': float(time_pred_val),
            'time_error': float(time_error),
            'mag_true': float(mag_true),
            'mag_pred': float(mag_pred_val),
            'mag_error': float(mag_error),
            'binary_true': int(binary_true),
        })

        print(f"\nSample {i}:")
        print(f"  Thời gian - True: {time_true/3600:.1f}h, Pred: {time_pred_val/3600:.1f}h, Error: {time_error/3600:.1f}h")
        print(f"  Độ mạnh - True: {mag_true:.2f}M, Pred: {mag_pred_val:.2f}M, Error: {mag_error:.2f}M")

    # Summary
    print(f"\n{'='*60}")
    print(" TỔNG KẾT")
    print(f"{'='*60}")

    avg_time_error = np.mean([r['time_error'] for r in results])
    avg_mag_error = np.mean([r['mag_error'] for r in results])

    print(f"\nTrung bình sai số ({args.n_samples} mẫu):")
    print(f"  Thời gian: {avg_time_error / 3600:.1f} giờ")
    print(f"  Độ mạnh: {avg_mag_error:.2f} M")

    # Lưu kết quả
    output = {
        'region': args.region,
        'n_samples': args.n_samples,
        'avg_time_error_hours': float(avg_time_error / 3600),
        'avg_mag_error': float(avg_mag_error),
        'samples': results
    }

    output_path = MODEL_DIR / 'test_ground_truth.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nKết quả lưu tại: {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
