"""
Split features_lstm.csv into time and mag specific datasets

Usage:
    python predict2/split_data.py

Output:
    - data_processed/features_time.csv: 23 input features + target_time_to_next
    - data_processed/features_mag.csv: 25 input features + target_next_mag

Author: haind
Date: 2025-03-25
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predict2.config import (
    FEATURES_FILE,
    FEATURES_TIME_FILE,
    FEATURES_MAG_FILE,
    TIME_FEATURES,
    MAG_FEATURES,
    TIME_TARGET,
    MAG_TARGET,
    DATA_PROCESSED_DIR
)


def split_features():
    """Split features_lstm.csv into time and mag datasets"""

    print(f"\n{'='*80}")
    print(" SPLITTING FEATURES_LSTM.CSV ".center(80))
    print(f"{'='*80}\n")

    # Load original features
    print(f"Loading: {FEATURES_FILE}")
    df = pd.read_csv(FEATURES_FILE)
    print(f"  Original shape: {df.shape}")
    print(f"  Original columns: {len(df.columns)}")

    # Verify required columns exist
    missing_time = [f for f in TIME_FEATURES if f not in df.columns]
    missing_mag = [f for f in MAG_FEATURES if f not in df.columns]

    if missing_time:
        raise ValueError(f"Missing TIME_FEATURES columns: {missing_time}")
    if missing_mag:
        raise ValueError(f"Missing MAG_FEATURES columns: {missing_mag}")

    # Check for required columns for creating targets
    required_for_time = ['target_time_to_next']
    required_for_mag = [MAG_TARGET]

    missing_time_targets = [f for f in required_for_time if f not in df.columns]
    missing_mag_targets = [f for f in required_for_mag if f not in df.columns]

    if missing_time_targets:
        raise ValueError(f"Missing required columns for time dataset: {missing_time_targets}")
    if missing_mag_targets:
        raise ValueError(f"Missing required columns for mag dataset: {missing_mag_targets}")

    # Create Time Dataset
    print(f"\n{'='*80}")
    print(" TIME DATASET ".center(80))
    print(f"{'='*80}")

    # Include target_time_to_next to create binary target
    time_cols = ['time'] + TIME_FEATURES + ['target_time_to_next']
    df_time = df[time_cols].copy()

    # Drop rows with missing target_time_to_next
    original_count = len(df_time)
    df_time = df_time.dropna(subset=['target_time_to_next'])
    dropped_count = original_count - len(df_time)

    # Create binary target: earthquake within 7 days
    SECONDS_IN_7_DAYS = 7 * 24 * 3600
    df_time['target_quake_in_7days'] = (df_time['target_time_to_next'] <= SECONDS_IN_7_DAYS).astype(int)

    # Remove target_time_to_next (keep only binary target)
    df_time = df_time.drop(columns=['target_time_to_next'])

    print(f"Features: {len(TIME_FEATURES)}")
    print(f"Target: target_quake_in_7days (binary)")
    print(f"Samples: {len(df_time):,}")
    if dropped_count > 0:
        print(f"Dropped (missing target): {dropped_count:,}")

    # Binary target statistics
    pos_count = df_time['target_quake_in_7days'].sum()
    neg_count = len(df_time) - pos_count
    pos_ratio = pos_count / len(df_time)

    print(f"\nBinary Target Statistics (7-day window):")
    print(f"  Positive (quake in 7 days): {pos_count:,} ({pos_ratio:.2%})")
    print(f"  Negative (no quake in 7 days): {neg_count:,} ({1-pos_ratio:.2%})")
    print(f"  Class imbalance ratio: {neg_count/pos_count:.2f}:1")

    # Show reference statistics from original time_to_next
    if 'target_time_to_next' in df.columns:
        time_to_next = df['target_time_to_next'].dropna()
        print(f"\nReference: target_time_to_next statistics (seconds):")
        print(f"  Mean:   {time_to_next.mean():>12,.1f}")
        print(f"  Median: {time_to_next.median():>12,.1f}")
        print(f"  Events within 7 days: {(time_to_next <= SECONDS_IN_7_DAYS).sum():,} ({(time_to_next <= SECONDS_IN_7_DAYS).mean():.2%})")

    # Create Mag Dataset
    print(f"\n{'='*80}")
    print(" MAG DATASET ".center(80))
    print(f"{'='*80}")
    mag_cols = ['time'] + MAG_FEATURES + [MAG_TARGET]
    df_mag = df[mag_cols].copy()

    # Drop rows with missing targets
    original_count = len(df_mag)
    df_mag = df_mag.dropna(subset=[MAG_TARGET])
    dropped_count = original_count - len(df_mag)

    print(f"Features: {len(MAG_FEATURES)}")
    print(f"Target: {MAG_TARGET}")
    print(f"Samples: {len(df_mag):,}")
    if dropped_count > 0:
        print(f"Dropped (missing target): {dropped_count:,}")

    # Statistics for mag target
    mag_stats = {
        'mean': df_mag[MAG_TARGET].mean(),
        'median': df_mag[MAG_TARGET].median(),
        'std': df_mag[MAG_TARGET].std(),
        'min': df_mag[MAG_TARGET].min(),
        'max': df_mag[MAG_TARGET].max()
    }
    print(f"\nMagnitude Statistics:")
    print(f"  Mean:   {mag_stats['mean']:>12.3f}")
    print(f"  Median: {mag_stats['median']:>12.3f}")
    print(f"  Std:    {mag_stats['std']:>12.3f}")
    print(f"  Min:    {mag_stats['min']:>12.3f}")
    print(f"  Max:    {mag_stats['max']:>12.3f}")

    # Save datasets
    print(f"\n{'='*80}")
    print(" SAVING DATASETS ".center(80))
    print(f"{'='*80}")
    df_time.to_csv(FEATURES_TIME_FILE, index=False)
    print(f"Time dataset:  {FEATURES_TIME_FILE}")

    df_mag.to_csv(FEATURES_MAG_FILE, index=False)
    print(f"Mag dataset:   {FEATURES_MAG_FILE}")

    print(f"\n{'='*80}")
    print(" SPLITTING COMPLETED ".center(80, "="))
    print(f"{'='*80}\n")

    return FEATURES_TIME_FILE, FEATURES_MAG_FILE


if __name__ == "__main__":
    split_features()
