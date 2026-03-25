"""
Data Preparer for Time Model
Loads features_time.csv and prepares data for TimeLSTM training
Binary Classification: Earthquake within 7 days (0/1)

Author: haind
Date: 2025-03-25
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import Dataset
import torch

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    FEATURES_TIME_FILE,
    TIME_FEATURES,
    TIME_TARGET,
    SEQUENCE_LENGTH
)


class TimeDataset(Dataset):
    """PyTorch Dataset for binary time prediction"""

    def __init__(self, X, y):
        """
        Args:
            X: Input sequences (n_samples, seq_length, n_features)
            y: Targets (n_samples,) - binary: 1 if quake in 7 days, else 0
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y.astype(np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TimeDataPreparer:
    """Prepare data for TimeLSTM training and prediction (Binary Classification)"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.region_encoder = LabelEncoder()
        self.data = None
        self.region_encoder_fitted = False

    def load_data(self, filepath=None):
        """Load features_time.csv and sort by region_code > time"""
        if filepath is None:
            filepath = FEATURES_TIME_FILE

        print(f"Loading time data from: {filepath}")
        self.data = pd.read_csv(filepath)

        # Drop rows with missing target
        self.data = self.data.dropna(subset=[TIME_TARGET])

        # Sort by region_code first, then by time
        self.data = self.data.sort_values(by=['region_code', 'time'], ascending=[True, True]).reset_index(drop=True)

        # Show binary target distribution
        pos_count = self.data[TIME_TARGET].sum()
        neg_count = len(self.data) - pos_count

        print(f"  Total events: {len(self.data):,}")
        print(f"  Features: {len(TIME_FEATURES)}")
        print(f"  Target: {TIME_TARGET} (binary classification)")
        print(f"  Positive (quake in 7 days): {pos_count:,} ({pos_count/len(self.data):.2%})")
        print(f"  Negative (no quake in 7 days): {neg_count:,} ({neg_count/len(self.data):.2%})")
        print(f"  Sorted by: region_code > time")
        print(f"  Unique regions: {self.data['region_code'].nunique()}")

        return self.data

    def prepare_sequences(self, data, for_training=True):
        """
        Prepare sequences for LSTM

        Args:
            data: DataFrame with features
            for_training: If True, include targets

        Returns:
            X: Input sequences (n_samples, sequence_length, n_features)
            y: Targets (n_samples,) - binary labels, only if for_training=True
        """
        sequences = []
        targets = []

        # Feature columns
        feature_cols = TIME_FEATURES.copy()

        # Encode region_code if present
        if 'region_code' in feature_cols:
            if not self.region_encoder_fitted:
                self.region_encoder.fit(data['region_code'])
                self.region_encoder_fitted = True
            data['region_encoded'] = self.region_encoder.transform(data['region_code'])

        for i in range(SEQUENCE_LENGTH, len(data)):
            # Get sequence of SEQUENCE_LENGTH previous events
            seq_data = data.iloc[i-SEQUENCE_LENGTH:i]

            # Extract features
            seq_features = []

            for feat in feature_cols:
                if feat == 'region_code':
                    # Use encoded value
                    seq_features.append(seq_data['region_encoded'].values)
                else:
                    seq_features.append(seq_data[feat].values)

            # Stack features: (sequence_length, n_features)
            seq_array = np.column_stack(seq_features)
            sequences.append(seq_array)

            # Add target if training (binary label, no transformation needed)
            if for_training:
                target = data.iloc[i][TIME_TARGET]
                targets.append(target)

        X = np.array(sequences)
        y = np.array(targets) if for_training else None

        print(f"  Sequences created: {len(sequences)}")
        print(f"  X shape: {X.shape}")
        if y is not None:
            print(f"  y shape: {y.shape} (binary labels)")

        return X, y

    def scale_features(self, X_train, X_val=None, X_test=None):
        """
        Scale features using StandardScaler

        Args:
            X_train: Training data
            X_val: Validation data (optional)
            X_test: Test data (optional)

        Returns:
            Scaled data
        """
        # Reshape to 2D for scaling
        original_shape = X_train.shape
        X_train_2d = X_train.reshape(-1, X_train.shape[-1])

        # Fit and transform
        X_train_scaled = self.scaler.fit_transform(X_train_2d)
        X_train_scaled = X_train_scaled.reshape(original_shape)

        result = {'X_train': X_train_scaled}

        if X_val is not None:
            val_shape = X_val.shape
            X_val_2d = X_val.reshape(-1, X_val.shape[-1])
            X_val_scaled = self.scaler.transform(X_val_2d)
            X_val_scaled = X_val_scaled.reshape(val_shape)
            result['X_val'] = X_val_scaled

        if X_test is not None:
            test_shape = X_test.shape
            X_test_2d = X_test.reshape(-1, X_test.shape[-1])
            X_test_scaled = self.scaler.transform(X_test_2d)
            X_test_scaled = X_test_scaled.reshape(test_shape)
            result['X_test'] = X_test_scaled

        return result

    def split_data(self, X, y, train_ratio=0.8, val_ratio=0.1):
        """
        Split data into train/val/test sets (temporal split)

        Args:
            X: Input sequences
            y: Targets (binary labels)
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
        """
        n_samples = len(X)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)

        X_train = X[:n_train]
        y_train = y[:n_train]

        X_val = X[n_train:n_train + n_val]
        y_val = y[n_train:n_train + n_val]

        X_test = X[n_train + n_val:]
        y_test = y[n_train + n_val:]

        # Show class distribution in each split
        print(f"  Train: {len(X_train)} samples (pos: {y_train.sum():.0f}, {y_train.mean():.2%})")
        print(f"  Val:   {len(X_val)} samples (pos: {y_val.sum():.0f}, {y_val.mean():.2%})")
        print(f"  Test:  {len(X_test)} samples (pos: {y_test.sum():.0f}, {y_test.mean():.2%})")

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def get_target_stats(self, y):
        """Get target statistics for binary classification"""
        pos_count = int(y.sum())
        neg_count = len(y) - pos_count

        stats = {
            'total': len(y),
            'positive': pos_count,
            'negative': neg_count,
            'positive_ratio': float(pos_count / len(y)),
            'negative_ratio': float(neg_count / len(y))
        }

        return stats


# Test the module
if __name__ == "__main__":
    print("Testing TimeDataPreparer module (Binary Classification)...")

    preparer = TimeDataPreparer()
    preparer.load_data()

    data = preparer.data
    X, y = preparer.prepare_sequences(data, for_training=True)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = preparer.split_data(X, y)

    scaled_data = preparer.scale_features(X_train, X_val, X_test)

    stats = preparer.get_target_stats(y_train)
    print(f"\nTraining set statistics:")
    print(f"  Total: {stats['total']:,}")
    print(f"  Positive: {stats['positive']:,} ({stats['positive_ratio']:.2%})")
    print(f"  Negative: {stats['negative']:,} ({stats['negative_ratio']:.2%})")

    print("\nTimeDataPreparer module test completed")
