"""
PyTorch Prediction Script
Make predictions using trained PyTorch LSTM model

Usage:
    python predict_pytorch.py --region R257_114
    python predict_pytorch.py --input recent_events.csv

Author: haind
Date: 2025-03-25
"""

import argparse
import os
import sys
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta

# Local imports
from config import MODEL_DIR, SEQUENCE_LENGTH, INPUT_FEATURES
from data_preparer import DataPreparer
from model_pytorch import EarthquakeLSTM, EarthquakeTrainer, load_scaler


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Predict earthquakes using trained PyTorch model')
    parser.add_argument('--model', type=str,
                       help='Path to model file (.pt)')
    parser.add_argument('--scaler', type=str,
                       help='Path to scaler file (.pkl)')
    parser.add_argument('--region', type=str,
                       help='Region code (e.g., R257_114)')
    parser.add_argument('--input', type=str,
                       help='CSV file with recent events (must have at least 5 events)')
    parser.add_argument('--lat', type=float,
                       help='Latitude for manual prediction')
    parser.add_argument('--lon', type=float,
                       help='Longitude for manual prediction')
    parser.add_argument('--depth', type=float, default=10.0,
                       help='Depth (km) for manual prediction')
    parser.add_argument('--mag', type=float,
                       help='Magnitude for manual prediction')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use (default: auto - detects GPU automatically)')

    return parser.parse_args()


def get_device(device_arg='auto'):
    """
    Get device for inference
    Auto-detects CUDA if available

    Args:
        device_arg: 'auto', 'cpu', or 'cuda'
    """
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"  ✓ Auto-detected GPU: {torch.cuda.get_device_name(0)}")
            return device
        else:
            print(f"  ⚠️  No GPU detected, using CPU")
            return torch.device('cpu')
    elif device_arg == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"  ✓ Using GPU: {torch.cuda.get_device_name(0)}")
            return device
        else:
            print(f"  ⚠️  CUDA requested but not available, using CPU")
            return torch.device('cpu')
    else:  # device_arg == 'cpu'
        print(f"  Using CPU")
        return torch.device('cpu')


def find_latest_model(region=None):
    """Find the latest trained model"""
    models = list(MODEL_DIR.glob('model*.pt'))

    if not models:
        raise FileNotFoundError(f"No models found in {MODEL_DIR}")

    # Filter by region if specified
    if region:
        models = [m for m in models if f'_{region}_' in str(m)]

    if not models:
        raise FileNotFoundError(f"No models found for region {region}")

    # Sort by modification time
    latest_model = max(models, key=lambda x: x.stat().st_mtime)

    # Find corresponding scaler
    model_name = latest_model.stem
    scaler_name = model_name.replace('model_', 'scaler_').replace('.pt', '.pkl')
    scaler_path = MODEL_DIR / scaler_name

    if not scaler_path.exists():
        # Try to find any scaler
        scalers = list(MODEL_DIR.glob('scaler*.pkl'))
        if scalers:
            scaler_path = max(scalers, key=lambda x: x.stat().st_mtime)
        else:
            raise FileNotFoundError(f"No scaler found for model {latest_model}")

    return latest_model, scaler_path


def load_model_and_scaler(model_path, scaler_path, device='cpu'):
    """Load trained model and scaler"""
    print(f"Loading model: {model_path}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Get model parameters
    n_features = checkpoint['n_features']
    lstm_hidden = checkpoint['lstm_hidden']

    # Create model
    model = EarthquakeLSTM(n_features=n_features, lstm_hidden=lstm_hidden)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"  Model loaded: {n_features} features, LSTM hidden: {lstm_hidden}")

    # Load scaler
    scaler = load_scaler(scaler_path)

    # Load metadata if exists
    metadata_path = model_path.parent / f"metadata_{model_path.stem.split('_', 1)[1]}.json"
    metadata = None
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"  Metadata: {metadata.get('region', 'N/A')}")

    return model, scaler, metadata


def prepare_input_from_csv(input_path, preparer, scaler):
    """
    Prepare input from CSV file

    Args:
        input_path: Path to CSV file with recent events
        preparer: DataPreparer instance
        scaler: Fitted scaler

    Returns:
        X: Ready for prediction
    """
    # Load input data
    input_data = pd.read_csv(input_path)
    input_data['time'] = pd.to_datetime(input_data['time'])

    # Ensure we have enough events
    if len(input_data) < SEQUENCE_LENGTH:
        # Load historical data to pad
        print(f"Need {SEQUENCE_LENGTH} events, got {len(input_data)}. Loading historical data...")
        preparer.load_data()

        # Get data from same region
        if 'region_code' in input_data.columns:
            region = input_data['region_code'].iloc[0]
            historical_data = preparer.data[preparer.data['region_code'] == region]
        else:
            historical_data = preparer.data

        # Get events before the input data
        last_time = input_data['time'].min()
        historical_data = historical_data[historical_data['time'] < last_time]

        # Combine
        combined = pd.concat([historical_data.tail(SEQUENCE_LENGTH - len(input_data)), input_data])
        input_data = combined.sort_values('time').reset_index(drop=True)

    # Prepare using data_preparer
    X = preparer.prepare_for_prediction(input_data)

    # Scale
    X_2d = X.reshape(-1, X.shape[-1])
    X_scaled = scaler.transform(X_2d)
    X_scaled = X_scaled.reshape(X.shape)

    return X_scaled


def prepare_input_manual(lat, lon, depth, mag, preparer, scaler):
    """
    Prepare input from manual parameters
    Uses most recent events from data

    Args:
        lat, lon, depth, mag: Earthquake parameters
        preparer: DataPreparer instance
        scaler: Fitted scaler

    Returns:
        X: Ready for prediction
    """
    # Load data
    preparer.load_data()

    # Create new event dataframe
    new_event = pd.DataFrame([{
        'time': pd.Timestamp.now(),
        'latitude': lat,
        'longitude': lon,
        'depth': depth,
        'mag': mag,
        'sig': 10**(mag),  # Approximate
        'mmi': 0,
        'cdi': 0,
        'felt': 0,
        'region_code': preparer.calculate_region_code(lat, lon) if hasattr(preparer, 'calculate_region_code') else 'R000_000'
    }])

    # Fill missing features with defaults
    for col in INPUT_FEATURES:
        if col not in new_event.columns and col != 'time':
            new_event[col] = 0

    # Get recent events from data
    recent_events = preparer.data.tail(SEQUENCE_LENGTH - 1).copy()

    # Add new event
    combined = pd.concat([recent_events, new_event], ignore_index=True)

    # Keep only SEQUENCE_LENGTH most recent
    combined = combined.tail(SEQUENCE_LENGTH).reset_index(drop=True)

    # Prepare
    X = preparer.prepare_for_prediction(combined)

    # Scale
    X_2d = X.reshape(-1, X.shape[-1])
    X_scaled = scaler.transform(X_2d)
    X_scaled = X_scaled.reshape(X.shape)

    return X_scaled


def predict(model, X, device='cpu'):
    """
    Make prediction

    Args:
        model: Trained model
        X: Input tensor (1, seq_length, n_features)
        device: 'cpu' or 'cuda'

    Returns:
        Dictionary with predictions
    """
    model.eval()

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        time_pred, mag_pred, binary_pred = model(X_tensor)

    time_to_next = time_pred.cpu().numpy()[0]
    next_mag = mag_pred.cpu().numpy()[0]
    next_mag_binary = binary_pred.cpu().numpy()[0]

    return {
        'time_to_next_seconds': time_to_next,
        'time_to_next_hours': time_to_next / 3600,
        'time_to_next_days': time_to_next / (24 * 3600),
        'next_magnitude': next_mag,
        'probability_M5_plus': next_mag_binary,
        'is_likely_M5_plus': next_mag_binary > 0.5
    }


def print_prediction(prediction):
    """Format and print prediction"""
    print(f"\n{'='*70}")
    print(" PREDICTION RESULTS")
    print(f"{'='*70}")

    print(f"\n📊 Time to next earthquake:")
    print(f"   {prediction['time_to_next_days']:.1f} days")
    print(f"   {prediction['time_to_next_hours']:.1f} hours")
    print(f"   {prediction['time_to_next_seconds']:,.0f} seconds")

    print(f"\n📈 Next earthquake magnitude:")
    print(f"   {prediction['next_magnitude']:.2f} M")

    print(f"\n⚠️  Probability of M5+ earthquake:")
    print(f"   {prediction['probability_M5_plus']:.1%}")
    if prediction['is_likely_M5_plus']:
        print(f"   ⚠️  HIGH RISK: Likely to be M5+")

    print(f"\n{'='*70}\n")


def main():
    """Main prediction function"""
    args = parse_args()

    print(f"\n{'='*70}")
    print(" PYTORCH LSTM PREDICTION")
    print(f"{'='*70}")

    # Get device with auto-detection
    device = get_device(args.device)
    print(f"Device: {device}")

    # Find model if not specified
    if args.model:
        model_path = args.model
        scaler_path = args.scaler if args.scaler else None
        if not scaler_path:
            # Try to find matching scaler
            model_name = Path(args.model).stem
            scaler_name = model_name.replace('model_', 'scaler_').replace('.pt', '.pkl')
            scaler_path = MODEL_DIR / scaler_name
    else:
        model_path, scaler_path = find_latest_model(args.region)

    # Load model and scaler
    model, scaler, metadata = load_model_and_scaler(model_path, scaler_path, device)

    # Create preparer
    preparer = DataPreparer()

    # Prepare input
    if args.input:
        print(f"\nLoading input from: {args.input}")
        X = prepare_input_from_csv(args.input, preparer, scaler)
    elif args.lat and args.lon and args.mag:
        print(f"\nUsing manual input: lat={args.lat}, lon={args.lon}, mag={args.mag}")
        X = prepare_input_manual(args.lat, args.lon, args.depth, args.mag, preparer, scaler)
    else:
        # Use most recent events from data
        print(f"\nUsing most recent events from data...")
        preparer.load_data()
        recent_data = preparer.data.tail(SEQUENCE_LENGTH).copy()
        X = preparer.prepare_for_prediction(recent_data)

        # Scale
        X_2d = X.reshape(-1, X.shape[-1])
        X_scaled = scaler.transform(X_2d)
        X_scaled = X_scaled.reshape(X.shape)
        X = X_scaled

    # Make prediction
    prediction = predict(model, X, device)

    # Print results
    print_prediction(prediction)

    # Save prediction to JSON
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    prediction_path = MODEL_DIR / f'prediction_{timestamp}.json'

    # Add metadata
    prediction['timestamp'] = timestamp
    prediction['model_used'] = str(model_path)
    prediction['predicted_time'] = (datetime.now() + timedelta(seconds=prediction['time_to_next_seconds'])).isoformat()

    with open(prediction_path, 'w') as f:
        json.dump(prediction, f, indent=2)

    print(f"Prediction saved to: {prediction_path}")


if __name__ == "__main__":
    main()
