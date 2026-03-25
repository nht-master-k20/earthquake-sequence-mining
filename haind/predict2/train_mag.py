"""
Training Script for MagLSTM Model
Train magnitude prediction model

Usage:
    # Full data
    python predict2/train_mag.py --epochs 50 --batch-size 64 --hidden 128 64

    # Subset (10% data) - for quick testing
    python predict2/train_mag.py --epochs 5 --subset-ratio 0.1

    # Subset (5% data)
    python predict2/train_mag.py --epochs 3 --subset-ratio 0.05

Author: haind
Date: 2025-03-25
Updated: 2026-03-26 (added subset-ratio support)
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
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    MODEL_DIR, LOG_DIR, SEQUENCE_LENGTH,
    FEATURES_MAG_FILE, MAG_FEATURES, MAG_TARGET,
    LEARNING_RATE, DROPOUT, EARLY_STOPPING_PATIENCE
)
from data.mag_data import MagDataPreparer, MagDataset
from models.mag_model import MagLSTM, MagTrainer, save_training_history
from dashboard_utils import TrainingDashboard


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train MagLSTM model for earthquake magnitude prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data subset
    parser.add_argument('--subset-ratio', type=float, default=1.0,
                       help='Data subset ratio (1.0=full, 0.1=10%%, 0.05=5%%)')

    # Model hyperparameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                       help='Learning rate')
    parser.add_argument('--hidden', type=int, nargs='+', default=[128, 64],
                       help='LSTM hidden units (e.g., --hidden 128 64 32)')
    parser.add_argument('--dropout', type=float, default=DROPOUT,
                       help='Dropout rate')
    parser.add_argument('--patience', type=int, default=EARLY_STOPPING_PATIENCE,
                       help='Early stopping patience')

    # Optional parameters
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use')

    return parser.parse_args()


def get_device(device_arg='auto'):
    """Get device for training"""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"  Auto-detected GPU: {torch.cuda.get_device_name(0)}")
            return device
        else:
            print(f"  No GPU detected, using CPU")
            return torch.device('cpu')
    elif device_arg == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
            return device
        else:
            print(f"  CUDA requested but not available, using CPU")
            return torch.device('cpu')
    else:
        print(f"  Using CPU")
        return torch.device('cpu')


def prepare_data(subset_ratio=1.0):
    """
    Prepare data for training

    Args:
        subset_ratio: Ratio of data to use (1.0=full, 0.1=10%%)

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, preparer
    """
    print(f"\n{'='*80}")
    print(" DATA PREPARATION - MAG MODEL".center(80))
    print(f"{'='*80}\n")

    # Check if features_mag.csv exists
    if not os.path.exists(FEATURES_MAG_FILE):
        print(f"\nError: {FEATURES_MAG_FILE} not found!")
        print("Run split_data.py first:")
        print("  python predict2/split_data.py")
        sys.exit(1)

    # Load data
    print("Loading data...")
    preparer = MagDataPreparer()
    preparer.load_data()

    data = preparer.data

    # Take subset if needed
    if subset_ratio < 1.0:
        n_total = len(data)
        n_subset = int(n_total * subset_ratio)
        subset_start = n_total - n_subset
        data = data.iloc[subset_start:].reset_index(drop=True)

        print(f"\n{'='*80}")
        print(f" SUBSET INFO".center(80))
        print(f"{'='*80}")
        print(f"  Original data: {n_total:,} events")
        print(f"  Subset ratio: {subset_ratio:.1%}")
        print(f"  Subset size: {n_subset:,} events (most recent)")
        print(f"  Using events from index {subset_start:,} to {n_total:,}")
        print(f"{'='*80}")

    # Show target statistics
    stats = preparer.get_target_stats(data[MAG_TARGET].values)
    print(f"\nTarget Statistics ({MAG_TARGET}):")
    print(f"  Mean:   {stats['mean']:>12.3f}")
    print(f"  Median: {stats['median']:>12.3f}")
    print(f"  Std:    {stats['std']:>12.3f}")
    print(f"  Min:    {stats['min']:>12.3f}")
    print(f"  Max:    {stats['max']:>12.3f}")

    # Prepare sequences
    print("\nPreparing sequences...")
    X, y = preparer.prepare_sequences(data, for_training=True)

    # Split data
    print("Splitting data...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = preparer.split_data(X, y)

    # Scale features
    print("Scaling features...")
    scaled_data = preparer.scale_features(X_train, X_val, X_test)

    return (
        scaled_data['X_train'], scaled_data['X_val'], scaled_data['X_test'],
        y_train, y_val, y_test,
        preparer
    )


def train_model(args):
    """
    Main training function with dashboard integration

    Args:
        args: Parsed command line arguments
    """
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Get device
    device = get_device(args.device)

    # Prepare data with subset
    X_train, X_val, X_test, y_train, y_val, y_test, preparer = prepare_data(args.subset_ratio)

    # Create datasets and dataloaders
    print("\nCreating data loaders...")
    train_dataset = MagDataset(X_train, y_train)
    val_dataset = MagDataset(X_val, y_val)
    test_dataset = MagDataset(X_test, y_test)

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
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # Build model
    n_features = X_train.shape[-1]
    model = MagLSTM(
        n_features=n_features,
        lstm_hidden=args.hidden,
        dropout=args.dropout
    )

    print(f"\n{'='*80}")
    print(" MODEL ARCHITECTURE - MAG MODEL".center(80))
    print(f"{'='*80}")
    print(f"{'Input Features:':<25} {n_features}")
    print(f"{'Sequence Length:':<25} {SEQUENCE_LENGTH}")
    print(f"{'LSTM Hidden Units:':<25} {args.hidden}")
    print(f"{'Dropout:':<25} {args.dropout}")
    print(f"{'Total Parameters:':<25} {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*80}")

    # Create trainer
    trainer = MagTrainer(
        model=model,
        device=device,
        learning_rate=args.lr
    )

    # Create dashboard
    dashboard = TrainingDashboard('MagLSTM', output_dir='predict2/dashboard')

    # Train
    print(f"\n{'='*80}")
    print(" TRAINING STARTED ".center(80))
    print(f"{'='*80}")

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        early_stopping_patience=args.patience
    )

    # Update dashboard with history
    for epoch in range(len(history['train_loss'])):
        lr = history['learning_rate'][epoch] if 'learning_rate' in history else args.lr
        epoch_time = history['epoch_time'][epoch] if 'epoch_time' in history else 0
        train_metrics = {
            'loss': history['train_loss'][epoch],
            'mae': history['train_mae'][epoch],
            'rmse': history['train_rmse'][epoch]
        }
        val_metrics = {
            'loss': history['val_loss'][epoch],
            'mae': history['val_mae'][epoch],
            'rmse': history['val_rmse'][epoch]
        }
        dashboard.update(epoch + 1, train_metrics, val_metrics, lr, epoch_time)

    # Update best epoch
    dashboard.best_epoch = trainer.best_epoch
    dashboard.best_val_loss = trainer.best_val_loss

    # Evaluate on test set
    print(f"\n{'='*80}")
    print(" TEST SET EVALUATION".center(80))
    print(f"{'='*80}")

    test_metrics = trainer.validate(test_loader)

    print(f"{'Test Loss:':<25} {test_metrics['loss']:>12.4f}")
    print(f"{'Test MAE:':<25} {test_metrics['mae']:>12.3f}")
    print(f"{'Test RMSE:':<25} {test_metrics['rmse']:>12.3f}")
    print(f"{'='*80}")

    # Save model
    print("\nSaving model artifacts...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Add subset prefix if not full data
    subset_suffix = "" if args.subset_ratio >= 1.0 else f"_subset{int(args.subset_ratio*100)}"

    # Calculate test indices (for reproducible evaluation)
    n_samples = len(X_train)
    test_start_idx = n_samples + len(X_val)
    test_indices = list(range(test_start_idx, test_start_idx + len(X_test)))

    model_path = MODEL_DIR / f'mag_model{subset_suffix}_{timestamp}.pt'
    trainer.save_model(model_path, scaler=preparer.scaler, test_indices=test_indices)

    # Save scaler
    import pickle
    scaler_path = MODEL_DIR / f'mag_scaler{subset_suffix}_{timestamp}.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(preparer.scaler, f)

    # Save region encoder
    encoder_path = MODEL_DIR / f'mag_region_encoder{subset_suffix}_{timestamp}.pkl'
    with open(encoder_path, 'wb') as f:
        pickle.dump(preparer.region_encoder, f)

    # Save history
    history_path = MODEL_DIR / f'mag_history{subset_suffix}_{timestamp}.json'
    save_training_history(history, history_path)

    # Create dashboard (training curves, metrics table, etc.)
    dashboard.create_summary_report(test_metrics)

    # Create test set visualizations
    print(f"\n{'='*80}")
    print(" CREATING TEST SET VISUALIZATIONS ".center(80))
    print(f"{'='*80}")

    # Get predictions on test set
    model.eval()
    y_pred_test = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch)
            y_pred_test.extend(pred.cpu().numpy())

    y_pred_test = np.array(y_pred_test)

    # Save comparison plot
    print("Saving comparison plot...")
    filepath = dashboard.save_comparison_plot(y_test, y_pred_test, 'Test')
    print(f"  Saved: {filepath}")

    # Save error distribution
    print("Saving error distribution...")
    filepath = dashboard.save_error_distribution(y_test, y_pred_test, 'Test')
    print(f"  Saved: {filepath}")

    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'model_type': 'MagLSTM',
        'target': MAG_TARGET,
        'n_features': n_features,
        'features': MAG_FEATURES,
        'sequence_length': SEQUENCE_LENGTH,
        'subset_ratio': args.subset_ratio,
        'epochs_trained': len(history['train_loss']),
        'best_val_loss': trainer.best_val_loss,
        'test_metrics': test_metrics,
        'dashboard_dir': str(dashboard.output_dir),
        'target_stats': {
            'mean_mag': float(np.mean(y_test)),
            'median_mag': float(np.median(y_test)),
            'mae_mag': test_metrics['mae']
        },
        'model_path': str(model_path),
        'scaler_path': str(scaler_path),
        'encoder_path': str(encoder_path),
        'history_path': str(history_path),
        'hyperparameters': {
            'hidden_units': args.hidden,
            'dropout': args.dropout,
            'learning_rate': args.lr,
            'batch_size': args.batch_size,
            'early_stopping_patience': args.patience
        }
    }

    metadata_path = MODEL_DIR / f'mag_metadata{subset_suffix}_{timestamp}.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*80}")
    print(" TRAINING COMPLETED".center(80))
    print(f"{'='*80}")
    print(f"{'Model:':<30} {model_path}")
    print(f"{'Scaler:':<30} {scaler_path}")
    print(f"{'Region Encoder:':<30} {encoder_path}")
    print(f"{'History:':<30} {history_path}")
    print(f"{'Metadata:':<30} {metadata_path}")
    print(f"{'Dashboard:':<30} {dashboard.output_dir}")
    print(f"{'='*80}\n")

    return model_path, scaler_path


def main():
    """Main function"""
    args = parse_args()

    print(f"\n{'='*80}")
    print(" MAGLSTM TRAINING ".center(80, "="))
    print(f"{'='*80}")
    print(f"{'Target:':<25} {MAG_TARGET}")
    print(f"{'Subset Ratio:':<25} {args.subset_ratio:.1%}")
    print(f"{'Features:':<25} {len(MAG_FEATURES)}")
    print(f"{'Epochs:':<25} {args.epochs}")
    print(f"{'Batch Size:':<25} {args.batch_size}")
    print(f"{'Learning Rate:':<25} {args.lr}")
    print(f"{'Hidden Units:':<25} {args.hidden}")
    print(f"{'Dropout:':<25} {args.dropout}")
    print(f"{'='*80}")

    # Create directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Train
    train_model(args)


if __name__ == "__main__":
    main()
