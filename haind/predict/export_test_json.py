"""
Export Test JSON từ features_lstm.csv (DB thực tế)
Data đã sort theo region và time, có đầy đủ features + targets

Usage:
    python export_test_json.py --region R257_114 --n-sequences 10
"""

import argparse
import json
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path

from config import FEATURES_FILE, MODEL_DIR, SEQUENCE_LENGTH


def load_data(region=None):
    """Load data từ CSV"""
    print(f"\n{'='*70}")
    print(" LOAD DATA FROM features_lstm.csv")
    print(f"{'='*70}")

    df = pd.read_csv(FEATURES_FILE)
    df['time'] = pd.to_datetime(df['time'])

    # Sort by region and time
    df = df.sort_values(['region_code', 'time']).reset_index(drop=True)

    if region:
        df = df[df['region_code'] == region].copy()
        print(f"  Region: {region}")

    print(f"  Total events: {len(df):,}")

    return df


def calculate_all_features(df):
    """
    Tính tất cả features còn thiếu cho mỗi row
    """
    print(f"\n{'='*70}")
    print(" CALCULATE FEATURES")
    print(f"{'='*70}")

    # Khởi tạo các cột mới
    df['sequence_id'] = 0
    df['seq_position'] = 0
    df['is_seq_mainshock'] = 0
    df['seq_mainshock_mag'] = 0.0
    df['seq_length'] = 1
    df['time_since_seq_start_sec'] = 0.0
    df['time_since_last_event'] = 0.0
    df['time_since_last_M5'] = 86400 * 365  # 1 year default
    df['interval_lag1'] = 0.0
    df['interval_lag2'] = 0.0
    df['interval_lag3'] = 0.0
    df['interval_lag4'] = 0.0
    df['interval_lag5'] = 0.0

    # Xử lý từng region
    for region in df['region_code'].unique():
        mask = df['region_code'] == region
        region_indices = df[mask].index.tolist()

        if len(region_indices) == 0:
            continue

        # --- Calculate LSTM features (time-based) ---
        last_m5_time = None
        for pos, idx in enumerate(region_indices):
            curr_time = df.loc[idx, 'time']

            # Time since last event
            if pos > 0:
                prev_idx = region_indices[pos - 1]
                prev_time = df.loc[prev_idx, 'time']
                df.loc[idx, 'time_since_last_event'] = (curr_time - prev_time).total_seconds()

                # Interval lags
                df.loc[idx, 'interval_lag1'] = df.loc[prev_idx, 'time_since_last_event']

            # Shift lags
            if pos >= 2:
                df.loc[idx, 'interval_lag2'] = df.loc[region_indices[pos-2], 'interval_lag1']
            if pos >= 3:
                df.loc[idx, 'interval_lag3'] = df.loc[region_indices[pos-3], 'interval_lag2']
            if pos >= 4:
                df.loc[idx, 'interval_lag4'] = df.loc[region_indices[pos-4], 'interval_lag3']
            if pos >= 5:
                df.loc[idx, 'interval_lag5'] = df.loc[region_indices[pos-5], 'interval_lag4']

            # Time since last M5+
            if df.loc[idx, 'mag'] >= 5.0:
                last_m5_time = curr_time
            if last_m5_time is not None:
                df.loc[idx, 'time_since_last_M5'] = (curr_time - last_m5_time).total_seconds()

        # --- Calculate sequence features (cluster events within 1 hour) ---
        seq_id = 0
        seq_start_idx = 0

        for pos, idx in enumerate(region_indices):
            curr_time = df.loc[idx, 'time']
            start_time = df.loc[region_indices[seq_start_idx], 'time']

            # Check if should start new sequence (more than 1 hour from sequence start)
            if pos > 0 and (curr_time - start_time).total_seconds() > 3600:
                # Start new sequence
                seq_id += 1
                seq_start_idx = pos
                start_time = curr_time

            df.loc[idx, 'sequence_id'] = seq_id
            df.loc[idx, 'seq_position'] = pos - seq_start_idx

        # Assign sequence-level features (mainshock, length, etc.)
        for seq in df.loc[region_indices, 'sequence_id'].unique():
            seq_mask = (df['region_code'] == region) & (df['sequence_id'] == seq)
            seq_indices = df[seq_mask].index.tolist()

            # Sequence length
            seq_len = len(seq_indices)

            # Find mainshock (largest mag)
            mag_values = [(idx, df.loc[idx, 'mag']) for idx in seq_indices]
            mainshock_idx = max(mag_values, key=lambda x: x[1])[0]
            mainshock_mag = df.loc[mainshock_idx, 'mag']

            # Seq start time
            seq_start_time = df.loc[seq_indices[0], 'time']

            # Assign to all events in sequence
            for idx in seq_indices:
                df.loc[idx, 'seq_length'] = seq_len
                df.loc[idx, 'is_seq_mainshock'] = 1 if idx == mainshock_idx else 0
                df.loc[idx, 'seq_mainshock_mag'] = mainshock_mag
                df.loc[idx, 'time_since_seq_start_sec'] = (df.loc[idx, 'time'] - seq_start_time).total_seconds()

    print(f"  Total sequences: {df['sequence_id'].nunique()}")
    print(f"  Features calculated")

    return df


def create_test_sequences(df, n_sequences=10):
    """
    Tạo test sequences với đầy đủ features
    """
    print(f"\n{'='*70}")
    print(f" CREATE {n_sequences} TEST SEQUENCES")
    print(f"{'='*70}")

    # Calculate features
    df = calculate_all_features(df)

    # Create sequences
    sequences = []

    for i in range(len(df) - SEQUENCE_LENGTH):
        # Get SEQUENCE_LENGTH events
        seq_df = df.iloc[i:i+SEQUENCE_LENGTH].copy()

        # Get next event for target
        if i + SEQUENCE_LENGTH < len(df):
            next_event = df.iloc[i+SEQUENCE_LENGTH]
            time_to_next = (next_event['time'] - seq_df.iloc[-1]['time']).total_seconds()
            next_mag = next_event['mag']
            is_M5_plus = 1 if next_mag >= 5.0 else 0

            # Build events list
            events = []
            for _, row in seq_df.iterrows():
                event = {}
                # All needed columns
                needed_cols = ['time', 'latitude', 'longitude', 'depth', 'mag', 'sig', 'mmi', 'cdi', 'felt',
                               'region_code', 'is_aftershock', 'mainshock_mag', 'seismicity_density_100km',
                               'coulomb_stress_proxy', 'regional_b_value', 'sequence_id', 'seq_position',
                               'is_seq_mainshock', 'seq_mainshock_mag', 'seq_length', 'time_since_seq_start_sec',
                               'time_since_last_event', 'time_since_last_M5', 'interval_lag1', 'interval_lag2',
                               'interval_lag3', 'interval_lag4', 'interval_lag5']

                for col in needed_cols:
                    if col in row.index:
                        val = row[col]
                        if pd.isna(val):
                            event[col] = 0.0
                        elif col == 'time':
                            event[col] = str(val)
                        elif isinstance(val, (np.integer, np.floating)):
                            event[col] = float(val)
                        else:
                            event[col] = val
                    else:
                        event[col] = 0.0

                events.append(event)

            sequences.append({
                'sequence_id': i,
                'region_code': seq_df.iloc[-1]['region_code'],
                'events': events,
                'ground_truth': {
                    'time_to_next_seconds': float(time_to_next),
                    'time_to_next_hours': float(time_to_next / 3600),
                    'next_magnitude': float(next_mag),
                    'is_M5_plus': int(is_M5_plus)
                }
            })

    # Get last n_sequences
    test_sequences = sequences[-n_sequences:]

    print(f"  Total sequences available: {len(sequences):,}")
    print(f"  Selected: {len(test_sequences)}")

    return test_sequences


def export_json(sequences, output_file='test_simplified.json'):
    """Export sequences to JSON"""
    output_path = MODEL_DIR / output_file

    with open(output_path, 'w') as f:
        json.dump(sequences, f, indent=2)

    print(f"\n  ✓ Exported to: {output_path}")

    # Show sample
    sample = sequences[0]
    print(f"\n  Sample sequence:")
    print(f"    ID: {sample['sequence_id']}")
    print(f"    Region: {sample['region_code']}")
    print(f"    Events: {len(sample['events'])}")
    print(f"    Event fields: {len(sample['events'][0])}")

    # Show ground truth
    gt = sample['ground_truth']
    print(f"    Ground truth:")
    print(f"      Time: {gt['time_to_next_hours']:.1f}h")
    print(f"      Mag: {gt['next_magnitude']:.2f}M")
    print(f"      M5+: {gt['is_M5_plus']}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Export test JSON from features_lstm.csv')
    parser.add_argument('--region', type=str, default='R257_114')
    parser.add_argument('--n-sequences', type=int, default=10)
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(" EXPORT TEST JSON FROM features_lstm.csv")
    print(f"{'='*70}")

    # Load data
    df = load_data(args.region)

    # Create test sequences
    sequences = create_test_sequences(df, n_sequences=args.n_sequences)

    # Export
    output_path = export_json(sequences, output_file=f'test_{args.region}_simplified.json')

    print(f"\n{'='*70}")
    print(f" ✓ DONE! Use: python test_from_json.py --input {output_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
