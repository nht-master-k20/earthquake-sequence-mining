"""
Test Prediction từ JSON file export từ features_lstm.csv

Usage:
    python test_from_json.py --input models/test_simplified.json
    python test_from_json.py --input models/test_simplified.json --seq-id 0
"""

import argparse
import json
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import torch
import numpy as np
from pathlib import Path

from config import MODEL_DIR, SEQUENCE_LENGTH, INPUT_FEATURES
from data_preparer import DataPreparer
from model_pytorch import EarthquakeLSTM, load_scaler


def get_device(device_arg='auto'):
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    return torch.device('cpu')


def find_latest_model():
    """Find latest model"""
    models = list(MODEL_DIR.glob('model*.pt'))
    if not models:
        raise FileNotFoundError(f"Không tìm thấy model trong {MODEL_DIR}")

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
    print(f"\n{'='*70}")
    print(" LOAD MODEL")
    print(f"{'='*70}")
    print(f"Model: {model_path.name}")
    print(f"Scaler: {scaler_path.name}")

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
    preparer = DataPreparer()
    preparer.load_data()  # Need this for region_encoder

    return model, scaler, preparer


def prepare_input_from_json(events_json, preparer):
    """
    Prepare input from JSON events
    Tạo input array trực tiếp từ JSON
    """
    import numpy as np

    # Fit region_encoder nếu chưa
    if not hasattr(preparer.region_encoder, 'classes_'):
        preparer.region_encoder.fit(preparer.data['region_code'])

    # Extract features theo đúng thứ tự INPUT_FEATURES
    feature_rows = []

    for evt in events_json:
        row = []
        for feat in INPUT_FEATURES:
            if feat == 'region_code':
                # Encode region
                region_code = str(evt.get('region_code', 'R000_000'))
                if region_code in preparer.region_encoder.classes_:
                    encoded = preparer.region_encoder.transform([region_code])[0]
                else:
                    encoded = 0
                row.append(encoded)
            elif feat == 'time':
                continue  # Skip time
            elif feat in evt:
                row.append(float(evt[feat]))
            else:
                row.append(0.0)
        feature_rows.append(row)

    # Convert to array và reshape
    X = np.array(feature_rows)
    X = X.reshape(1, SEQUENCE_LENGTH, -1)

    return X


def predict(model, X, scaler, device='cpu'):
    """Make prediction"""
    # Scale
    X_2d = X.reshape(-1, X.shape[-1])
    X_scaled = scaler.transform(X_2d)
    X_scaled = X_scaled.reshape(X.shape)

    # Predict
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        time_pred, mag_pred, binary_pred = model(X_tensor)

    return {
        'time_to_next_seconds': float(time_pred.cpu().numpy()[0]),
        'time_to_next_hours': float(time_pred.cpu().numpy()[0] / 3600),
        'next_magnitude': float(mag_pred.cpu().numpy()[0]),
        'prob_M5_plus': float(binary_pred.cpu().numpy()[0])
    }


def main():
    parser = argparse.ArgumentParser(description='Test prediction from JSON')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to JSON file')
    parser.add_argument('--seq-id', type=int, default=None,
                       help='Sequence ID to test (default: all)')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model file (default: latest)')
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(" TEST FROM JSON")
    print(f"{'='*70}")

    device = get_device('auto')

    # Load JSON
    json_path = Path(args.input)
    print(f"\nInput: {json_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Handle both list and dict (by_region format)
    if isinstance(data, dict):
        # Flatten by_region format
        sequences = []
        for region, seqs in data.items():
            sequences.extend(seqs)
    else:
        sequences = data

    print(f"Sequences: {len(sequences)}")

    # Load model
    if args.model:
        model_path = Path(args.model)
        scaler_name = model_path.stem.replace('model_', 'scaler_') + '.pkl'
        scaler_path = MODEL_DIR / scaler_name
    else:
        model_path, scaler_path = find_latest_model()

    model, scaler, preparer = load_model(model_path, scaler_path, device)

    # Test sequences
    if args.seq_id is not None:
        # Test specific sequence
        sequences = [sequences[args.seq_id]]

    results = []

    for seq in sequences:
        seq_id = seq.get('sequence_id', '?')
        region = seq.get('region_code', '?')
        events = seq['events']
        ground_truth = seq.get('ground_truth', seq.get('target', {}))

        print(f"\n{'='*70}")
        print(f" SEQUENCE {seq_id} - Region: {region}")
        print(f"{'='*70}")

        # Show last event
        last_event = events[-1]
        print(f"Last event: {last_event['time']}")
        print(f"  Location: ({last_event['latitude']:.2f}, {last_event['longitude']:.2f})")
        print(f"  Magnitude: {last_event['mag']:.2f} M")

        # Prepare input
        X = prepare_input_from_json(events, preparer)

        # Predict
        prediction = predict(model, X, scaler, device)

        # Print results
        print(f"\n📊 PREDICTION:")
        print(f"  Time to next: {prediction['time_to_next_hours']:.1f} hours")
        print(f"  Next magnitude: {prediction['next_magnitude']:.2f} M")
        print(f"  Prob M5+: {prediction['prob_M5_plus']:.1%}")

        # Compare with ground truth
        if ground_truth:
            print(f"\n📈 GROUND TRUTH:")
            print(f"  Time to next: {ground_truth['time_to_next_hours']:.1f} hours")
            print(f"  Next magnitude: {ground_truth['next_magnitude']:.2f} M")
            print(f"  Is M5+: {ground_truth['is_M5_plus']}")

            # Calculate errors
            time_error = abs(prediction['time_to_next_hours'] - ground_truth['time_to_next_hours'])
            mag_error = abs(prediction['next_magnitude'] - ground_truth['next_magnitude'])

            print(f"\n📉 ERRORS:")
            print(f"  Time: {time_error:.1f} hours")
            print(f"  Magnitude: {mag_error:.2f} M")

        results.append({
            'sequence_id': seq_id,
            'region': region,
            'prediction': prediction,
            'ground_truth': ground_truth
        })

    # Save results
    print(f"\n{'='*70}")
    print(f" SUMMARY")
    print(f"{'='*70}")

    if results:
        avg_time_error = np.mean([
            abs(r['prediction']['time_to_next_hours'] - r['ground_truth'].get('time_to_next_hours', 0))
            for r in results if r['ground_truth']
        ])
        avg_mag_error = np.mean([
            abs(r['prediction']['next_magnitude'] - r['ground_truth'].get('next_magnitude', 0))
            for r in results if r['ground_truth']
        ])

        print(f"\nAverage errors ({len(results)} sequences):")
        print(f"  Time: {avg_time_error:.1f} hours")
        print(f"  Magnitude: {avg_mag_error:.2f} M")

    output_path = MODEL_DIR / 'test_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {output_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
