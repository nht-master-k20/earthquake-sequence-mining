"""
Data Preparation Module for LSTM
Load, process, and prepare data for training/prediction

Author: haind
Date: 2025-03-25
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from config import (
    FEATURES_FILE, INPUT_FEATURES, TARGET_FEATURES,
    SEQUENCE_LENGTH, ORIGINAL_FEATURES
)


class DataPreparer:
    """Prepare data for LSTM training and prediction"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.region_encoder = LabelEncoder()
        self.data = None
        self.regions = None

    def load_data(self, filepath=None):
        """Load features data from CSV"""
        if filepath is None:
            filepath = FEATURES_FILE

        print(f"Loading data from: {filepath}")
        self.data = pd.read_csv(filepath)
        self.data['time'] = pd.to_datetime(self.data['time'])

        print(f"  Total events: {len(self.data):,}")
        print(f"  Features: {len(self.data.columns)}")

        return self.data

    def get_regions(self):
        """Get list of unique regions"""
        if self.data is None:
            self.load_data()

        self.regions = self.data['region_code'].unique()
        print(f"  Total regions: {len(self.regions):,}")

        return self.regions

    def filter_by_region(self, region_code, min_events=10):
        """
        Filter data for specific region

        Args:
            region_code: Region identifier (e.g., 'R221_570')
            min_events: Minimum number of events required
        """
        if self.data is None:
            self.load_data()

        region_data = self.data[self.data['region_code'] == region_code].copy()

        if len(region_data) < min_events:
            raise ValueError(
                f"Region {region_code} has only {len(region_data)} events. "
                f"Minimum {min_events} required."
            )

        print(f"  Region {region_code}: {len(region_data)} events")

        return region_data

    def prepare_sequences(self, data, for_training=True):
        """
        Prepare sequences for LSTM

        Args:
            data: DataFrame with features
            for_training: If True, include targets; if False, prediction only

        Returns:
            X: Input sequences (n_samples, sequence_length, n_features)
            y: Targets (n_samples, n_targets) - only if for_training=True
        """
        sequences = []
        targets = []

        # Get feature columns (exclude time and targets)
        feature_cols = [col for col in INPUT_FEATURES if col != 'time']

        # Encode region_code
        data['region_encoded'] = self.region_encoder.fit_transform(data['region_code'])

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

            # Add targets if training
            if for_training:
                target = data.iloc[i][TARGET_FEATURES].values
                targets.append(target)

        X = np.array(sequences)
        y = np.array(targets) if for_training else None

        print(f"  Sequences created: {len(sequences)}")
        print(f"  X shape: {X.shape}")
        if y is not None:
            print(f"  y shape: {y.shape}")

        return X, y

    def prepare_for_prediction(self, recent_events):
        """
        Prepare recent events for prediction

        Args:
            recent_events: DataFrame with recent events (at least SEQUENCE_LENGTH)
                            Can include user input + historical data

        Returns:
            X: Input sequence for LSTM (1, sequence_length, n_features)
        """
        if len(recent_events) < SEQUENCE_LENGTH:
            raise ValueError(
                f"Need at least {SEQUENCE_LENGTH} events for prediction. "
                f"Got {len(recent_events)}."
            )

        # Get last SEQUENCE_LENGTH events
        last_events = recent_events.tail(SEQUENCE_LENGTH).copy()

        # Encode region if needed
        if 'region_encoded' not in last_events.columns:
            # Fit encoder with all regions first
            if self.data is not None:
                self.region_encoder.fit(self.data['region_code'])
            last_events['region_encoded'] = self.region_encoder.transform(
                last_events['region_code']
            )

        # Prepare features
        feature_cols = [col for col in INPUT_FEATURES if col != 'time']
        seq_features = []

        for feat in feature_cols:
            if feat == 'region_code':
                seq_features.append(last_events['region_encoded'].values)
            else:
                seq_features.append(last_events[feat].values)

        # Stack and reshape
        seq_array = np.column_stack(seq_features)
        X = seq_array.reshape(1, SEQUENCE_LENGTH, -1)

        print(f"  Prediction input shape: {X.shape}")

        return X

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
        Split data into train/val/test sets

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


# Test the module
if __name__ == "__main__":
    print("Testing DataPreparer module...")

    preparer = DataPreparer()
    preparer.load_data()

    regions = preparer.get_regions()

    # Test with first region
    if len(regions) > 0:
        print(f"\nTesting with region: {regions[0]}")
        region_data = preparer.filter_by_region(regions[0])
        X, y = preparer.prepare_sequences(region_data, for_training=True)
        print(f"\n✓ DataPreparer module test completed")
    else:
        print("No regions found!")
