"""
PyTorch Training Script
Train LSTM model on ALL earthquake data

Usage:
    python train_pytorch.py --epochs 50 --batch-size 64
    python train_pytorch.py --region R257_114 --epochs 50

Author: haind
Date: 2025-03-25
"""

import argparse
import os
import sys
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Local imports
from config import MODEL_DIR, LOG_DIR, SEQUENCE_LENGTH
from data_preparer import DataPreparer
from model_pytorch import (
    EarthquakeLSTM, EarthquakeTrainer, EarthquakeDataset,
    save_scaler, save_training_history
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train earthquake LSTM model with PyTorch',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model hyperparameters (what users care about)
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--hidden', type=int, nargs='+', default=[128, 64],
                       help='LSTM hidden units (e.g., --hidden 128 64 32)')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')

    # Optional parameters
    parser.add_argument('--region', type=str, default=None,
                       help='Train on specific region only (default: all regions)')
    parser.add_argument('--min-events', type=int, default=100,
                       help='Minimum events per region to include (default: 100)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use (default: auto - detects GPU automatically)')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode (5 epochs)')

    return parser.parse_args()


def get_device(device_arg='auto'):
    """
    Get device for training
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


def prepare_data(args):
    """
    Prepare data for training

    Args:
        args: Parsed arguments

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, scaler
    """
    print(f"\n{'='*70}")
    print(" DATA PREPARATION")
    print(f"{'='*70}")

    # Load data
    preparer = DataPreparer()
    preparer.load_data()

    # Get regions
    regions = preparer.get_regions()

    # Filter and prepare data
    if args.region:
        print(f"Mode: SINGLE REGION - {args.region}")
        region_data = preparer.filter_by_region(args.region, min_events=args.min_events)
        data_to_use = region_data
    else:
        print(f"Mode: ALL REGIONS (min {args.min_events} events/region)")
        all_data = []
        valid_regions = []
        total_events = 0

        for region in regions:
            region_data = preparer.data[preparer.data['region_code'] == region]
            if len(region_data) >= args.min_events:
                all_data.append(region_data)
                valid_regions.append(region)
                total_events += len(region_data)

        print(f"  Found {len(valid_regions)} valid regions")
        print(f"  Total events: {total_events:,}")

        # Show top regions
        region_counts = [(r, len(preparer.data[preparer.data['region_code'] == r]))
                         for r in valid_regions]
        region_counts.sort(key=lambda x: x[1], reverse=True)
        print(f"  Top 5 regions:")
        for r, c in region_counts[:5]:
            print(f"    {r}: {c:,} events")

        data_to_use = pd.concat(all_data, ignore_index=True)

    print(f"  Final dataset: {len(data_to_use):,} events")

    # Prepare sequences
    X, y = preparer.prepare_sequences(data_to_use, for_training=True)

    # Split data (temporal split - no shuffle)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = preparer.split_data(X, y)

    # Scale features
    scaled_data = preparer.scale_features(X_train, X_val, X_test)

    return (
        scaled_data['X_train'], scaled_data['X_val'], scaled_data['X_test'],
        y_train, y_val, y_test,
        preparer.scaler
    )


def train_model(args):
    """
    Main training function

    Args:
        args: Parsed command line arguments
    """
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Get device
    device = get_device(args.device)
    print(f"\nDevice: {device}")

    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(args)

    # Create datasets and dataloaders
    train_dataset = EarthquakeDataset(X_train, y_train)
    val_dataset = EarthquakeDataset(X_val, y_val)
    test_dataset = EarthquakeDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    # Build model with custom hyperparameters
    n_features = X_train.shape[-1]
    model = EarthquakeLSTM(
        n_features=n_features,
        lstm_hidden=args.hidden,
        dropout=args.dropout
    )

    print(f"\n{'='*70}")
    print(" MODEL ARCHITECTURE")
    print(f"{'='*70}")
    print(f"Input features: {n_features}")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"LSTM hidden units: {args.hidden}")
    print(f"Dropout: {args.dropout}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    trainer = EarthquakeTrainer(
        model=model,
        device=device,
        learning_rate=args.lr
    )

    # Train
    epochs = 5 if args.test else args.epochs
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        early_stopping_patience=args.patience
    )

    # Evaluate on test set
    print(f"\n{'='*70}")
    print(" TEST SET EVALUATION")
    print(f"{'='*70}")

    test_metrics = trainer.validate(test_loader)

    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Time MAE: {test_metrics['time_mae']:.1f} seconds")
    print(f"Test Mag MAE: {test_metrics['mag_mae']:.3f}")
    print(f"Test Binary Acc: {test_metrics['binary_acc']:.3f}")

    # Save model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    region_suffix = f"_{args.region}" if args.region else "_all_regions"

    model_path = MODEL_DIR / f'model{region_suffix}_{timestamp}.pt'
    trainer.save_model(model_path)

    # Save scaler
    scaler_path = MODEL_DIR / f'scaler{region_suffix}_{timestamp}.pkl'
    save_scaler(scaler, scaler_path)

    # Save history
    history_path = MODEL_DIR / f'history{region_suffix}_{timestamp}.json'
    save_training_history(history, history_path)

    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'region': args.region or 'all_regions',
        'min_events': args.min_events,
        'epochs_trained': len(history['train_loss']),
        'best_val_loss': trainer.best_val_loss,
        'test_metrics': test_metrics,
        'model_path': str(model_path),
        'scaler_path': str(scaler_path),
        'history_path': str(history_path),
        'n_features': n_features,
        'sequence_length': SEQUENCE_LENGTH,
        'hyperparameters': {
            'hidden_units': args.hidden,
            'dropout': args.dropout,
            'learning_rate': args.lr,
            'batch_size': args.batch_size
        }
    }

    metadata_path = MODEL_DIR / f'metadata{region_suffix}_{timestamp}.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*70}")
    print(" TRAINING COMPLETED")
    print(f"{'='*70}")
    print(f"Model: {model_path}")
    print(f"Scaler: {scaler_path}")
    print(f"History: {history_path}")
    print(f"Metadata: {metadata_path}")

    return model_path, scaler_path


def main():
    """Main function"""
    args = parse_args()

    print(f"\n{'='*70}")
    print(" PYTORCH LSTM TRAINING")
    print(f"{'='*70}")
    print(f"Data: {args.region or 'ALL REGIONS'}")
    print(f"Min events: {args.min_events}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Hidden units: {args.hidden}")
    print(f"Dropout: {args.dropout}")

    # Create directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Train
    train_model(args)


if __name__ == "__main__":
    main()
