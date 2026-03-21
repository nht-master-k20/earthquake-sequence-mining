"""
=============================================================================
SEQUENCE DATA PREPARATION - LSTM EARTHQUAKE PREDICTION (PROPER FIX)
=============================================================================

Mục tiêu:
    - Create sequences for LSTM from prepared features
    - Simple sliding window approach per spatial cluster
    - Proper temporal ordering

Proper Fix - Cải tiến:
    - Simplified sequence creation logic
    - Clear temporal ordering within clusters
    - No complex spatial filtering - just cluster-based sequences
    - Memory efficient for large dataset

Input:
    - data/train_features.npz, val_features.npz, test_features.npz

Output:
    - data/train_sequences.npz: Training sequences for LSTM
    - data/val_sequences.npz: Validation sequences
    - data/test_sequences.npz: Test sequences
    - data/sequence_config.json: Configuration

Tác giả: Haind
Ngày tạo: 2025-03-20
Cập nhật: 2025-03-20 (Proper Fix)
=============================================================================
"""

import numpy as np
import json
from pathlib import Path
from tqdm import tqdm


def print_step(step, total, description):
    print(f"\n[{step}/{total}] {description}...")


def print_success(message):
    print(f"  ✓ {message}")


def print_info(message):
    print(f"  ℹ {message}")


def create_cluster_sequences(features, time_target, mag_target,
                               spatial_clusters, seq_len=10, min_cluster_size=seq_len+2):
    """
    Create sequences using simple sliding window per cluster

    Parameters:
    -----------
    features : array
        Shape (n_events, n_features)
    time_target : array
        Log time to next event
    mag_target : array
        Next event magnitude
    spatial_clusters : array
        Cluster labels for each event
    seq_len : int
        Sequence length (number of previous events)
    min_cluster_size : int
        Minimum cluster size to create sequences

    Returns:
    --------
    sequences : array
        Shape (n_sequences, seq_len, n_features)
    seq_time_targets : array
        Time targets for each sequence
    seq_mag_targets : array
        Magnitude targets for each sequence
    """
    sequences = []
    time_targets_list = []
    mag_targets_list = []

    # Get unique clusters
    unique_clusters = np.unique(spatial_clusters)
    unique_clusters = unique_clusters[unique_clusters >= 0]  # Exclude noise (-1)

    print_info(f"  Processing {len(unique_clusters)} clusters...")

    for cluster_id in tqdm(unique_clusters, desc="  Creating sequences"):
        # Get events in this cluster
        cluster_mask = spatial_clusters == cluster_id
        cluster_indices = np.where(cluster_mask)[0]

        # Skip small clusters
        if len(cluster_indices) < min_cluster_size:
            continue

        # Create sequences using sliding window
        # Each sequence uses seq_len previous events to predict the next
        for i in range(seq_len, len(cluster_indices)):
            # Sequence: events [i-seq_len : i]
            seq_indices = cluster_indices[i-seq_len:i]
            seq = features[seq_indices]

            # Target: event i
            target_idx = cluster_indices[i]

            # Skip if target is NaN
            if np.isnan(time_target[target_idx]) or np.isnan(mag_target[target_idx]):
                continue

            # Skip if sequence has NaN
            if np.isnan(seq).any():
                continue

            sequences.append(seq)
            time_targets_list.append(time_target[target_idx])
            mag_targets_list.append(mag_target[target_idx])

    return np.array(sequences), np.array(time_targets_list), np.array(mag_targets_list)


def main():
    print("=" * 70)
    print(" SEQUENCE DATA PREPARATION - PROPER FIX ".center(70))
    print("=" * 70)

    # ============================================================================
    # 1. LOAD FEATURES
    # ============================================================================
    print_step(1, 4, "Loading features")

    data_dir = Path(os.path.dirname(os.path.abspath(__file__))) / 'data'

    print_info("Loading train_features.npz...")
    train_data = np.load(data_dir / 'train_features.npz')
    train_features = train_data['features']
    train_time_target = train_data['time_target']
    train_mag_target = train_data['mag_target']
    train_clusters = train_data['spatial_cluster']

    print_info("Loading val_features.npz...")
    val_data = np.load(data_dir / 'val_features.npz')
    val_features = val_data['features']
    val_time_target = val_data['time_target']
    val_mag_target = val_data['mag_target']
    val_clusters = val_data['spatial_cluster']

    print_info("Loading test_features.npz...")
    test_data = np.load(data_dir / 'test_features.npz')
    test_features = test_data['features']
    test_time_target = test_data['time_target']
    test_mag_target = test_data['mag_target']
    test_clusters = test_data['spatial_cluster']

    print_success("Loaded features")
    print_info(f"Train: {len(train_features):,} events")
    print_info(f"Val: {len(val_features):,} events")
    print_info(f"Test: {len(test_features):,} events")
    print_info(f"Features per event: {train_features.shape[1]}")

    # Load config
    with open(data_dir / 'features.json', 'r') as f:
        feature_config = json.load(f)

    # ============================================================================
    # 2. CREATE SEQUENCES
    # ============================================================================
    print_step(2, 4, "Creating sequences")

    SEQ_LEN = 10  # Use 10 previous events

    print_info(f"Sequence length: {SEQ_LEN}")
    print_info("Creating training sequences...")

    X_train, y_train_time, y_train_mag = create_cluster_sequences(
        train_features, train_time_target, train_mag_target,
        train_clusters, seq_len=SEQ_LEN
    )

    print_info("Creating validation sequences...")
    X_val, y_val_time, y_val_mag = create_cluster_sequences(
        val_features, val_time_target, val_mag_target,
        val_clusters, seq_len=SEQ_LEN
    )

    print_info("Creating test sequences...")
    X_test, y_test_time, y_test_mag = create_cluster_sequences(
        test_features, test_time_target, test_mag_target,
        test_clusters, seq_len=SEQ_LEN
    )

    print_success("Created sequences")
    print_info(f"Train sequences: {len(X_train):,}")
    print_info(f"Val sequences: {len(X_val):,}")
    print_info(f"Test sequences: {len(X_test):,}")
    print_info(f"Sequence shape: {X_train.shape}")

    # ============================================================================
    # 3. SAVE SEQUENCES
    # ============================================================================
    print_step(3, 4, "Saving sequences")

    print_info("Saving train_sequences.npz...")
    np.savez_compressed(
        data_dir / 'train_sequences.npz',
        X=X_train.astype(np.float32),
        y_time=y_train_time.astype(np.float32),
        y_mag=y_train_mag.astype(np.float32)
    )

    print_info("Saving val_sequences.npz...")
    np.savez_compressed(
        data_dir / 'val_sequences.npz',
        X=X_val.astype(np.float32),
        y_time=y_val_time.astype(np.float32),
        y_mag=y_val_mag.astype(np.float32)
    )

    print_info("Saving test_sequences.npz...")
    np.savez_compressed(
        data_dir / 'test_sequences.npz',
        X=X_test.astype(np.float32),
        y_time=y_test_time.astype(np.float32),
        y_mag=y_test_mag.astype(np.float32)
    )

    # Save sequence config
    seq_config = {
        'seq_len': SEQ_LEN,
        'num_features': int(X_train.shape[2]),
        'feature_names': feature_config['feature_columns'],
        'num_sequences': {
            'train': int(len(X_train)),
            'val': int(len(X_val)),
            'test': int(len(X_test))
        },
        'sequence_creation': 'sliding_window_per_cluster',
        'description': f'Each sequence uses {SEQ_LEN} previous events from same cluster'
    }

    with open(data_dir / 'sequence_config.json', 'w') as f:
        json.dump(seq_config, f, indent=2)

    print_success("Saved all sequence files!")

    # ============================================================================
    # 4. DISPLAY SAMPLE SEQUENCE
    # ============================================================================
    print_step(4, 4, "Sample sequence display")

    if len(X_test) > 0:
        print("\n  Sample sequence (test set):")
        sample_idx = np.random.choice(len(X_test))
        sample_seq = X_test[sample_idx]

        print(f"  Sequence {sample_idx}:")
        for i in range(min(5, len(sample_seq))):
            lat, lon, mag = sample_seq[i, 0], sample_seq[i, 1], sample_seq[i, 3]
            print(f"    [{i}] lat={lat:.2f}, lon={lon:.2f}, mag={mag:.2f}")

        # Convert back from log scale
        actual_time = np.exp(y_test_time[sample_idx]) - 1
        actual_mag = y_test_mag[sample_idx]
        print(f"  → Next event: {actual_time:.3f} days, mag={actual_mag:.2f}")

    # ============================================================================
    # SUMMARY
    # ============================================================================
    print("\n" + "=" * 70)
    print(" SUMMARY ".center(70))
    print("=" * 70)
    print(f"\n  Sequences:")
    print(f"    Train: {len(X_train):,}")
    print(f"    Val:   {len(X_val):,}")
    print(f"    Test:  {len(X_test):,}")
    print(f"\n  Shape:")
    print(f"    Input:  (batch, seq_len={SEQ_LEN}, features={X_train.shape[2]})")
    print(f"    Output: (batch, 2) - [log_time_to_next, next_mag]")
    print(f"\n  Method:")
    print(f"    Sliding window per spatial cluster")
    print(f"    Each sequence = {SEQ_LEN} consecutive events from same cluster")
    print(f"\n  Files:")
    print(f"    - {data_dir / 'train_sequences.npz'}")
    print(f"    - {data_dir / 'val_sequences.npz'}")
    print(f"    - {data_dir / 'test_sequences.npz'}")
    print(f"    - {data_dir / 'sequence_config.json'}")
    print(f"\n  Next step:")
    print(f"    python 04_lstm_model.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
