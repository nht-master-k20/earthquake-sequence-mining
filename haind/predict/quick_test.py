"""
Quick Test Script
Test PyTorch training with small subset of data

Usage:
    python quick_test.py
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import MODEL_DIR, SEQUENCE_LENGTH
from data_preparer import DataPreparer
from model_pytorch import EarthquakeLSTM, EarthquakeTrainer, EarthquakeDataset


def quick_test():
    """Quick test with region R257_114 (most events)"""
    print("\n" + "="*70)
    print(" QUICK TEST - PyTorch LSTM Training")
    print("="*70)

    # Load data
    preparer = DataPreparer()
    preparer.load_data()

    # Use region with most events
    region_code = 'R257_114'
    print(f"\nUsing region: {region_code}")

    region_data = preparer.filter_by_region(region_code, min_events=1000)
    print(f"Events in region: {len(region_data):,}")

    # Prepare sequences
    X, y = preparer.prepare_sequences(region_data, for_training=True)
    print(f"Sequences created: {len(X):,}")

    # Split data (smaller for quick test)
    n_train = int(len(X) * 0.8)
    n_val = int(len(X) * 0.1)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]

    print(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")

    # Scale
    scaled = preparer.scale_features(X_train, X_val, X_test)
    X_train_s = scaled['X_train']
    X_val_s = scaled['X_val']
    X_test_s = scaled['X_test']

    # Create dataloaders
    train_dataset = EarthquakeDataset(X_train_s, y_train)
    val_dataset = EarthquakeDataset(X_val_s, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Build model
    n_features = X_train_s.shape[-1]
    model = EarthquakeLSTM(n_features=n_features, lstm_hidden=[64, 32])

    print(f"\nModel: {n_features} features, LSTM hidden: [64, 32]")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train for a few epochs
    device = 'cpu'
    trainer = EarthquakeTrainer(model, device=device, learning_rate=0.001)

    print("\n" + "="*70)
    print(" TRAINING (5 epochs for quick test)")
    print("="*70)

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5,
        early_stopping_patience=10
    )

    # Test prediction
    print("\n" + "="*70)
    print(" TEST PREDICTION")
    print("="*70)

    X_sample = X_test_s[:1]  # Take one sample
    prediction = trainer.predict(X_sample)

    print(f"\nInput: 1 sequence of shape {X_sample.shape}")
    print(f"Predicted time to next: {prediction['time_to_next'][0]:.1f} seconds")
    print(f"Predicted next magnitude: {prediction['next_mag'][0]:.2f}")
    print(f"Predicted M5+ probability: {prediction['next_mag_binary'][0]:.3f}")

    # Actual values
    actual_time = y_test[0, 0]
    actual_mag = y_test[0, 1]
    actual_binary = y_test[0, 2]

    print(f"\nActual time to next: {actual_time:.1f} seconds")
    print(f"Actual next magnitude: {actual_mag:.2f}")
    print(f"Actual M5+: {actual_binary:.0f}")

    print(f"\nTime error: {abs(prediction['time_to_next'][0] - actual_time):.1f} seconds")
    print(f"Mag error: {abs(prediction['next_mag'][0] - actual_mag):.2f}")

    print("\n" + "="*70)
    print(" QUICK TEST COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nFor full training, run:")
    print("  python train_pytorch.py --region R257_114 --epochs 50")
    print("\nFor prediction, run:")
    print("  python predict_pytorch.py --region R257_114")

    return model, history


if __name__ == "__main__":
    quick_test()
