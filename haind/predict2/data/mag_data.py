"""
Data Preparer for Mag Model
Loads features_mag.csv and prepares data for MagLSTM training

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
    FEATURES_MAG_FILE,
    MAG_FEATURES,
    MAG_TARGET,
    SEQUENCE_LENGTH
)


class MagDataset(Dataset):
    """PyTorch Dataset for magnitude prediction"""

    def __init__(self, X, y):
        """
        Args:
            X: Input sequences (n_samples, seq_length, n_features)
            y: Targets (n_samples,) - next_mag magnitude
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y.astype(np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MagDataPreparer:
    """Prepare data for MagLSTM training and prediction"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.region_encoder = LabelEncoder()
        self.data = None
        self.region_encoder_fitted = False

    def load_data(self, filepath=None):
        """Load features_mag.csv and sort by region_code > time"""
        if filepath is None:
            filepath = FEATURES_MAG_FILE

        print(f"Loading mag data from: {filepath}")
        self.data = pd.read_csv(filepath)

        # Drop rows with missing target
        self.data = self.data.dropna(subset=[MAG_TARGET])

        # Sort by region_code first, then by time
        self.data = self.data.sort_values(by=['region_code', 'time'], ascending=[True, True]).reset_index(drop=True)

        print(f"  Total events: {len(self.data):,}")
        print(f"  Features: {len(MAG_FEATURES)}")
        print(f"  Target: {MAG_TARGET}")
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
            y: Targets (n_samples,) - only if for_training=True
        """
        sequences = []
        targets = []

        # Feature columns
        feature_cols = MAG_FEATURES.copy()

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

            # Add target if training
            if for_training:
                target = data.iloc[i][MAG_TARGET]
                targets.append(target)

        X = np.array(sequences)
        y = np.array(targets) if for_training else None

        print(f"  Sequences created: {len(sequences)}")
        print(f"  X shape: {X.shape}")
        if y is not None:
            print(f"  y shape: {y.shape}")

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
            y: Targets
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

        print(f"  Train: {len(X_train)} samples")
        print(f"  Val: {len(X_val)} samples")
        print(f"  Test: {len(X_test)} samples")

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def get_target_stats(self, y):
        """Get target statistics"""
        return {
            'mean': float(np.mean(y)),
            'median': float(np.median(y)),
            'std': float(np.std(y)),
            'min': float(np.min(y)),
            'max': float(np.max(y))
        }


# Test the module
if __name__ == "__main__":
    print("Testing MagDataPreparer module...")

    preparer = MagDataPreparer()
    preparer.load_data()

    data = preparer.data
    X, y = preparer.prepare_sequences(data, for_training=True)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = preparer.split_data(X, y)

    scaled_data = preparer.scale_features(X_train, X_val, X_test)

    stats = preparer.get_target_stats(y_train)
    print(f"\nTarget statistics (magnitude):")
    print(f"  Mean: {stats['mean']:.3f}")
    print(f"  Median: {stats['median']:.3f}")
    print(f"  Std: {stats['std']:.3f}")

    print("\nMagDataPreparer module test completed")
