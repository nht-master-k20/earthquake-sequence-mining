"""
Training Script
Train LSTM model on earthquake data

Usage:
    python train.py --region R221_570 --epochs 50

Author: haind
Date: 2025-03-25
"""

import argparse
import os
import sys
import json
from datetime import datetime

import tensorflow as tf
from config import *
from data_preparer import DataPreparer
from model_builder import EarthquakeLSTM


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train earthquake LSTM model')
    parser.add_argument('--region', type=str, help='Region code (e.g., R221_570). If not specified, train on all regions.')
    parser.add_argument('--min-events', type=int, default=100,
                       help='Minimum events per region (default: 100)')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                       help=f'Number of epochs (default: {EPOCHS})')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                       help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode (use small subset)')
    return parser.parse_args()


def train_single_region(region_code, min_events=100, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """
    Train model on single region

    Args:
        region_code: Region identifier
        min_events: Minimum events required
        epochs: Number of training epochs
        batch_size: Batch size
    """
    print(f"\n{'='*70}")
    print(f" TRAINING MODEL FOR REGION: {region_code}")
    print(f"{'='*70}")

    # Load and prepare data
    preparer = DataPreparer()
    preparer.load_data()

    # Filter by region
    region_data = preparer.filter_by_region(region_code, min_events=min_events)

    # Prepare sequences
    X, y = preparer.prepare_sequences(region_data, for_training=True)

    # Split data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = preparer.split_data(X, y)

    # Scale features
    scaled_data = preparer.scale_features(X_train, X_val, X_test)
    X_train_scaled = scaled_data['X_train']
    X_val_scaled = scaled_data['X_val']
    X_test_scaled = scaled_data['X_test']

    # Build model
    n_features = X_train_scaled.shape[-1]
    model_builder = EarthquakeLSTM(n_features=n_features)
    model_builder.build_model()

    print("\n" + "="*70)
    print(" MODEL ARCHITECTURE")
    print("="*70)
    model_builder.summary()

    # Train model
    print("\n" + "="*70)
    print(" TRAINING")
    print("="*70)
    history = model_builder.train(
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        epochs=epochs,
        batch_size=batch_size
    )

    # Evaluate on test set
    print("\n" + "="*70)
    print(" EVALUATION ON TEST SET")
    print("="*70)
    test_results = model_builder.model.evaluate(
        X_test_scaled,
        {
            'output_time': y_test[:, 0],
            'output_mag': y_test[:, 1],
            'output_binary': y_test[:, 2]
        },
        verbose=1
    )

    # Save model
    model_path = MODEL_DIR / f'model_{region_code}.keras'
    model_builder.save_model(model_path)

    # Save training history
    history_path = MODEL_DIR / f'history_{region_code}.json'
    with open(history_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        history_dict = {}
        for key, values in history.history.items():
            history_dict[key] = [float(v) for v in values]
        json.dump(history_dict, f, indent=2)

    print(f"\n{'='*70}")
    print(f" TRAINING COMPLETED")
    print(f"{'='*70}")
    print(f"Model saved: {model_path}")
    print(f"History saved: {history_path}")
    print(f"Test loss: {test_results[0]:.4f}")

    return model_builder, history


def train_all_regions(min_events=100, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """
    Train model on all regions with sufficient data

    Args:
        min_events: Minimum events per region
        epochs: Number of epochs
        batch_size: Batch size
    """
    print(f"\n{'='*70}")
    print(f" TRAINING MODEL ON ALL REGIONS")
    print(f"{'='*70}")

    # Load data
    preparer = DataPreparer()
    preparer.load_data()
    regions = preparer.get_regions()

    # Count events per region
    region_counts = {}
    for region in regions:
        count = len(preparer.data[preparer.data['region_code'] == region])
        region_counts[region] = count

    # Filter regions with enough events
    valid_regions = {r: c for r, c in region_counts.items() if c >= min_events}

    print(f"\nRegions with >= {min_events} events: {len(valid_regions)}")
    for region, count in sorted(valid_regions.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {region}: {count} events")

    # Combine all regions for training
    print(f"\nPreparing data for {len(valid_regions)} regions...")

    all_data = []
    for region in valid_regions.keys():
        region_data = preparer.filter_by_region(region, min_events=min_events)
        all_data.append(region_data)

    # Concatenate all regions
    combined_data = pd.concat(all_data, ignore_index=True)

    # Prepare sequences
    X, y = preparer.prepare_sequences(combined_data, for_training=True)

    # Split data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = preparer.split_data(X, y)

    # Scale features
    scaled_data = preparer.scale_features(X_train, X_val, X_test)

    # Build model
    n_features = X_train.shape[-1]
    model_builder = EarthquakeLSTM(n_features=n_features)
    model_builder.build_model()

    # Train
    history = model_builder.train(
        scaled_data['X_train'], y_train,
        scaled_data['X_val'], y_val,
        epochs=epochs,
        batch_size=batch_size
    )

    # Save
    model_builder.save_model(MODEL_DIR / 'model_all_regions.keras')

    return model_builder, history


def main():
    """Main training function"""
    args = parse_args()

    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    if args.test:
        print("\n⚠️  RUNNING IN TEST MODE (small subset)")
        args.epochs = 2
        args.min_events = 50

    if args.region:
        # Train on specific region
        train_single_region(
            region_code=args.region,
            min_events=args.min_events,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    else:
        # Train on all regions
        train_all_regions(
            min_events=args.min_events,
            epochs=args.epochs,
            batch_size=args.batch_size
        )


if __name__ == "__main__":
    main()
