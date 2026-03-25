"""
Prediction Script
Predict next earthquake time and magnitude

Usage:
    python predict.py --region R221_570
    python predict.py --input events.json

Author: haind
Date: 2025-03-25
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from data_preparer import DataPreparer
from model_builder import EarthquakeLSTM


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Predict next earthquake')
    parser.add_argument('--region', type=str,
                       help='Region code (e.g., R221_570)')
    parser.add_argument('--input', type=str,
                       help='JSON file with recent events')
    parser.add_argument('--model', type=str,
                       help=f'Model file path (default: models/model_<region>.keras or models/best_model.keras)')
    return parser.parse_args()


def calculate_region_code(lat, lon, grid_size=0.5):
    """
    Calculate region code from lat/lon

    Args:
        lat: Latitude
        lon: Longitude
        grid_size: Grid size in degrees (0.5 = ~55km)

    Returns:
        Region code string
    """
    lat_offset = 90
    lon_offset = 180
    lat_int = int((lat + lat_offset) / grid_size)
    lon_int = int((lon + lon_offset) / grid_size)
    return f"R{lat_int:03d}_{lon_int:03d}"


def predict_from_historical(region_code, n_recent=10, model_path=None):
    """
    Predict using historical data only

    Args:
        region_code: Region identifier
        n_recent: Number of recent events to display
        model_path: Path to trained model

    Returns:
        Prediction dictionary
    """
    print(f"\n{'='*70}")
    print(f" PREDICTION FOR REGION: {region_code}")
    print(f"{'='*70}")

    # Load data
    preparer = DataPreparer()
    preparer.load_data()

    # Filter by region
    try:
        region_data = preparer.filter_by_region(region_code, min_events=SEQUENCE_LENGTH)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return None

    # Show recent events
    print(f"\n📋 Recent {min(n_recent, len(region_data))} events:")
    print("-" * 70)
    recent = region_data.tail(n_recent)
    for i, (_, row) in enumerate(recent.itertuples()):
        print(f"  {i+1}. {row.time} | Mag {row.mag:.1f} | "
              f"Lat {row.latitude:.2f} | Lon {row.longitude:.2f} | "
              f"Depth {row.depth:.1f}km")

    # Prepare input
    X = preparer.prepare_for_prediction(region_data)

    # Load model
    if model_path is None:
        model_path = MODEL_DIR / f'model_{region_code}.keras'
        if not os.path.exists(model_path):
            # Try generic model
            model_path = MODEL_DIR / 'model_all_regions.keras'

    if not os.path.exists(model_path):
        print(f"\n❌ Error: Model file not found: {model_path}")
        print(f"Please train model first using: python train.py --region {region_code}")
        return None

    print(f"\n📦 Loading model: {model_path}")

    # Build model and load weights
    n_features = X.shape[-1]
    model_builder = EarthquakeLSTM(n_features=n_features)
    model_builder.load_model(model_path)

    # Make prediction
    print(f"\n🔮 Making prediction...")
    prediction = model_builder.predict(X)

    # Display results
    print(f"\n{'='*70}")
    print(f" PREDICTION RESULTS")
    print(f"{'='*70}")

    # Calculate next event time
    last_event_time = region_data['time'].iloc[-1]
    next_time_delta = timedelta(seconds=prediction['time_to_next'])
    next_event_time = last_event_time + next_time_delta

    print(f"\n📌 Last event:")
    print(f"   Time: {last_event_time}")
    print(f"   Mag:  {region_data['mag'].iloc[-1]:.1f}")

    print(f"\n🎯 Predicted next event:")
    print(f"   Time: {next_event_time}")
    print(f"   In:   {prediction['time_to_next']:.0f} seconds ({prediction['time_to_next']/3600:.1f} hours)")
    print(f"   Mag:  {prediction['next_mag']:.1f}")
    print(f"   M5+: {'YES' if prediction['next_mag_binary'] >= 0.5 else 'NO'}")

    # Confidence indicators
    print(f"\n📊 Confidence:")
    if prediction['next_mag_binary'] >= 0.5:
        confidence = prediction['next_mag_binary'] * 100
        print(f"   Probability of M5+: {confidence:.1f}%")
    else:
        confidence = (1 - prediction['next_mag_binary']) * 100
        print(f"   Probability of <M5: {confidence:.1f}%")

    # Risk assessment
    print(f"\n⚠️  Risk Assessment:")
    if prediction['next_mag'] >= 6.0:
        print(f"   🔴 HIGH RISK - M{prediction['next_mag']:.1f} earthquake predicted")
    elif prediction['next_mag'] >= 5.0:
        print(f"   🟠 MODERATE RISK - M{prediction['next_mag']:.1f} earthquake predicted")
    else:
        print(f"   🟢 LOW RISK - M{prediction['next_mag']:.1f} earthquake predicted")

    # Calculate time until prediction
    hours_until = prediction['time_to_next'] / 3600
    if hours_until < 1:
        urgency = "⚡ IMMEDIATE"
    elif hours_until < 24:
        urgency = "⏰ SOON"
    elif hours_until < 168:
        urgency = "📅 THIS WEEK"
    else:
        urgency = "🗓️ FUTURE"

    print(f"   Urgency: {urgency}")

    return {
        'region_code': region_code,
        'last_event_time': str(last_event_time),
        'predicted_time': str(next_event_time),
        'time_to_next_seconds': prediction['time_to_next'],
        'time_to_next_hours': prediction['time_to_next'] / 3600,
        'predicted_magnitude': prediction['next_mag'],
        'is_m5_plus': prediction['next_mag_binary'] >= 0.5,
        'confidence': confidence if prediction['next_mag_binary'] >= 0.5 else (1 - prediction['next_mag_binary'])
    }


def predict_from_user_input(input_file, model_path=None):
    """
    Predict using user input + historical data

    Args:
        input_file: JSON file with recent events from user
        model_path: Path to trained model

    Returns:
        Prediction dictionary
    """
    print(f"\n{'='*70}")
    print(f" PREDICTION FROM USER INPUT")
    print(f"{'='*70}")

    # Load user input
    with open(input_file, 'r') as f:
        user_events = json.load(f)

    print(f"\n📋 User provided {len(user_events)} events")

    # Convert to DataFrame
    df_input = pd.DataFrame(user_events)
    df_input['time'] = pd.to_datetime(df_input['time'])

    # Calculate region from first event
    region_code = calculate_region_code(
        df_input['latitude'].iloc[0],
        df_input['longitude'].iloc[0]
    )

    print(f"   Region: {region_code}")

    # Load historical data for this region
    preparer = DataPreparer()
    preparer.load_data()

    try:
        historical_data = preparer.filter_by_region(region_code, min_events=SEQUENCE_LENGTH)
    except ValueError:
        print(f"❌ Error: Not enough historical data for region {region_code}")
        return None

    # Combine historical + user input
    print(f"\n📊 Historical events: {len(historical_data)}")
    print(f"📊 User input events: {len(df_input)}")

    # Sort and combine
    all_data = pd.concat([historical_data, df_input], ignore_index=True)
    all_data = all_data.sort_values('time').reset_index(drop=True)

    # Show recent events (last event should be user's)
    print(f"\n📋 Recent events (last 5):")
    print("-" * 70)
    recent = all_data.tail(5)
    for i, (_, row) in enumerate(recent.itertuples()):
        source = "👤 USER" if i >= len(recent) - len(df_input) else "📚 HIST"
        print(f"  {i+1}. {row.time} | Mag {row.mag:.1f} | {source}")

    # Prepare input
    X = preparer.prepare_for_prediction(all_data)

    # Load model
    if model_path is None:
        model_path = MODEL_DIR / f'model_{region_code}.keras'
        if not os.path.exists(model_path):
            model_path = MODEL_DIR / 'model_all_regions.keras'

    print(f"\n📦 Loading model: {model_path}")

    n_features = X.shape[-1]
    model_builder = EarthquakeLSTM(n_features=n_features)
    model_builder.load_model(model_path)

    # Predict
    print(f"\n🔮 Making prediction...")
    prediction = model_builder.predict(X)

    # Display results (same as above)
    last_event_time = all_data['time'].iloc[-1]
    next_time_delta = timedelta(seconds=prediction['time_to_next'])
    next_event_time = last_event_time + next_time_delta

    print(f"\n{'='*70}")
    print(f" PREDICTION RESULTS")
    print(f"{'='*70}")

    print(f"\n📌 Last event (user input):")
    print(f"   Time: {last_event_time}")
    print(f"   Mag:  {all_data['mag'].iloc[-1]:.1f}")

    print(f"\n🎯 Predicted next event:")
    print(f"   Time: {next_event_time}")
    print(f"   In:   {prediction['time_to_next']:.0f} seconds ({prediction['time_to_next']/3600:.1f} hours)")
    print(f"   Mag:  {prediction['next_mag']:.1f}")
    print(f"   M5+: {'YES' if prediction['next_mag_binary'] >= 0.5 else 'NO'}")

    return {
        'region_code': region_code,
        'last_event_time': str(last_event_time),
        'predicted_time': str(next_event_time),
        'time_to_next_seconds': prediction['time_to_next'],
        'time_to_next_hours': prediction['time_to_next'] / 3600,
        'predicted_magnitude': prediction['next_mag'],
        'is_m5_plus': prediction['next_mag_binary'] >= 0.5
    }


def main():
    """Main prediction function"""
    args = parse_args()

    if args.region:
        # Predict from historical data
        predict_from_historical(args.region, model_path=args.model)
    elif args.input:
        # Predict from user input
        predict_from_user_input(args.input, model_path=args.model)
    else:
        print("❌ Error: Please specify --region or --input")
        print("\nUsage:")
        print("  python predict.py --region R221_570")
        print("  python predict.py --input events.json")


if __name__ == "__main__":
    main()
