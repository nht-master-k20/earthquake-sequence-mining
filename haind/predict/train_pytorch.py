"""
PyTorch Training Script
Train LSTM model on earthquake data using PyTorch

Usage:
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

# PyTorch imports
import torch
from torch.utils.data import DataLoader

# Local imports
from config import *
from data_preparer import DataPreparer
from model_pytorch import (
    EarthquakeLSTM, EarthquakeTrainer, EarthquakeDataset,
    save_scaler, save_training_history
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train earthquake LSTM model with PyTorch')
    parser.add_argument('--region', type=str, default=None,
                       help='Region code (e.g., R257_114). If not specified, train on all regions.')
    parser.add_argument('--min-events', type=int, default=1000,
                       help='Minimum events per region (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50,
                       help=f'Number of epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help=f'Batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience (default: 15)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use (default: cpu)')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode (fewer epochs)')

    return parser.parse_args()


def get_device(device_arg):
    """Get device for training"""
    if device_arg == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def prepare_data(region_code=None, min_events=1000):
    """
    Prepare data for training

    Args:
        region_code: Region code to filter by (None for all regions)
        min_events: Minimum events per region

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, scaler
    """
    print(f"\n{'='*70}")
    print(" DATA PREPARATION")
    print(f"{'='*70}")

    # Load data
    preparer = DataPreparer()
    preparer.load_data()
    regions = preparer.get_regions()

    # Filter regions
    if region_code:
        print(f"Filtering by region: {region_code}")
        region_data = preparer.filter_by_region(region_code, min_events=min_events)
        data_to_use = region_data
    else:
        print(f"Using all regions with >= {min_events} events")
        all_data = []
        valid_regions = []

        for region in regions:
            region_data = preparer.data[preparer.data['region_code'] == region]
            if len(region_data) >= min_events:
                all_data.append(region_data)
                valid_regions.append(region)

        print(f"Found {len(valid_regions)} valid regions")
        data_to_use = pd.concat(all_data, ignore_index=True)

    print(f"Total events: {len(data_to_use):,}")

    # Prepare sequences
    X, y = preparer.prepare_sequences(data_to_use, for_training=True)

    # Split data
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
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(
        region_code=args.region,
        min_events=args.min_events
    )

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

    # Build model
    n_features = X_train.shape[-1]
    model = EarthquakeLSTM(n_features=n_features)

    print(f"\n{'='*70}")
    print(" MODEL ARCHITECTURE")
    print(f"{'='*70}")
    print(f"Input features: {n_features}")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"LSTM hidden units: {model.lstm_hidden}")
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
        'sequence_length': SEQUENCE_LENGTH
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
    print(f"Region: {args.region or 'All regions'}")
    print(f"Min events: {args.min_events}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {args.device}")

    # Create directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Train
    train_model(args)


if __name__ == "__main__":
    main()
