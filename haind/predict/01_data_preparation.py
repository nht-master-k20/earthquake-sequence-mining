"""
=============================================================================
DATA PREPARATION - LSTM EARTHQUAKE PREDICTION (PROPER FIX)
=============================================================================

Mục tiêu:
    - Load full earthquake catalog (1.3M records)
    - Create spatial clusters for sequence grouping
    - Build temporal order + proper train/val/test split

Proper Fix - Cải tiến:
    - Clean spatial clustering với DBSCAN (thay vì KMeans)
    - Proper temporal ordering within clusters
    - Efficient memory usage for large dataset

Input:
    - features_advanced.csv: Full features (1.3M records)

Output:
    - data/prepared_data.npz: Prepared data with sequences ready
    - data/data_config.json: Configuration metadata

Tác giả: Haind
Ngày tạo: 2025-03-20
Cập nhật: 2025-03-20 (Proper Fix)
=============================================================================
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree


def print_step(step, total, description):
    print(f"\n[{step}/{total}] {description}...")


def print_success(message):
    print(f"  ✓ {message}")


def print_info(message):
    print(f"  ℹ {message}")


def create_spatial_clusters(coords, eps_km=50, min_samples=5):
    """
    Tạo spatial clusters dùng DBSCAN để tự động xác định số clusters

    Parameters:
    -----------
    coords : array-like
        (latitude, longitude) pairs
    eps_km : float
        Maximum distance between samples (km)
    min_samples : int
        Minimum samples in a cluster

    Returns:
    --------
    labels : array
        Cluster labels (-1 = noise)
    """
    # Convert to radians for haversine metric
    coords_rad = np.radians(coords)

    # DBSCAN với haversine metric
    # eps ở radians = eps_km / earth_radius_km
    earth_radius_km = 6371.0
    eps_rad = eps_km / earth_radius_km

    clustering = DBSCAN(
        eps=eps_rad,
        min_samples=min_samples,
        metric='haversine',
        n_jobs=-1
    )

    # Chạy trên sample nếu data quá lớn
    if len(coords) > 100000:
        print_info(f"  Running DBSCAN on sample of 100k points...")
        sample_indices = np.random.choice(len(coords), 100000, replace=False)
        sample_labels = clustering.fit_predict(coords_rad[sample_indices])

        # For remaining points, assign to nearest cluster center
        print_info(f"  Assigning remaining points to nearest cluster...")
        unique_labels = np.unique(sample_labels[sample_labels >= 0])
        cluster_centers = []

        for label in unique_labels:
            mask = sample_labels == label
            cluster_centers.append(coords_rad[sample_indices][mask].mean(axis=0))

        cluster_centers = np.array(cluster_centers)

        # Assign all points to nearest cluster center
        kdtree = cKDTree(cluster_centers)
        distances, indices = kdtree.query(coords_rad)

        labels = np.array([unique_labels[idx] if dist < eps_rad else -1
                          for idx, dist in zip(indices, distances)])
    else:
        labels = clustering.fit_predict(coords_rad)

    return labels


def calculate_time_since_last(df):
    """
    Tính time_since_last: thời gian (ngày) kể từ event GẦN NHẤT
    trong CÙNG spatial cluster

    Đây là feature quan trọng để capture seismic patterns
    """
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)

    time_since_last = []

    # Process theo cluster
    for cluster_id in df['spatial_cluster'].unique():
        if cluster_id == -1:  # Skip noise
            continue

        cluster_mask = df['spatial_cluster'] == cluster_id
        cluster_df = df[cluster_mask].sort_values('time').reset_index(drop=True)

        cluster_times = []
        for i in range(len(cluster_df)):
            if i == 0:
                cluster_times.append(0.0)
            else:
                current_time = cluster_df.iloc[i]['time']
                prev_time = cluster_df.iloc[i-1]['time']
                time_diff = (current_time - prev_time).total_seconds() / 86400  # days
                cluster_times.append(time_diff)

        # Map back to original dataframe
        for idx, t in zip(cluster_df.index, cluster_times):
            time_since_last.append((cluster_df.index[idx], t))

    # Create series
    result = pd.Series(0.0, index=df.index)
    for idx, val in time_since_last:
        result.loc[idx] = val

    return result


def main():
    print("=" * 70)
    print(" DATA PREPARATION - LSTM PROPER FIX ".center(70))
    print("=" * 70)

    # ============================================================================
    # 1. LOAD DATA
    # ============================================================================
    print_step(1, 6, "Loading earthquake data")

    input_path = Path(os.path.dirname(os.path.abspath(__file__))) / 'features_advanced.csv'

    # Optimized dtype với 17 advanced features mới
    dtype_opt = {
        'latitude': 'float32',
        'longitude': 'float32',
        'depth': 'float32',
        'mag': 'float32',
        'magType': 'category',
        'mmi': 'float32',
        'cdi': 'float32',
        'felt': 'float32',
        'sig': 'float32',
        'tsunami': 'float32',
        'gap': 'float32',
        'rms': 'float32',
        'nst': 'float32',
        'dmin': 'float32',
        'type': 'category',
        # Base Advanced Features (9)
        'is_aftershock': 'bool',
        'mainshock_mag': 'float32',
        'dist_to_5th_neighbor_km': 'float32',
        'dist_to_10th_neighbor_km': 'float32',
        'seismicity_density_100km': 'float32',
        'coulomb_stress_proxy': 'float32',
        'regional_b_value': 'float32',
        'seismic_gap_days': 'float32',
        'regional_max_mag_5yr': 'float32',
        # Stress Tensor Features (5) - NEW
        'stress_sigma_1_mpa': 'float32',
        'stress_sigma_3_mpa': 'float32',
        'stress_tau_max_mpa': 'float32',
        'stress_rate_mpa_per_year': 'float32',
        'stress_drop_recent_mpa': 'float32',
        # Fault Geometry Features (4) - NEW
        'fault_depth_km': 'float32',
        'fault_strike_deg': 'float32',
        'fault_dip_deg': 'float32',
        'fault_length_km': 'float32'
    }

    df = pd.read_csv(input_path, dtype=dtype_opt)

    print_success(f"Loaded {len(df):,} events")
    print_info(f"Columns: {len(df.columns)}")
    print_info(f"Memory: {df.memory_usage(deep=True).sum() / 1024**3:.2f} GB")

    # ============================================================================
    # 2. CREATE SPATIAL CLUSTERS
    # ============================================================================
    print_step(2, 6, "Creating spatial clusters with DBSCAN")

    coords = df[['latitude', 'longitude']].values

    # DBSCAN clustering
    eps_km = 50  # 50km radius
    df['spatial_cluster'] = create_spatial_clusters(coords, eps_km=eps_km)

    n_clusters = df['spatial_cluster'].nunique() - 1  # Exclude noise (-1)
    n_noise = (df['spatial_cluster'] == -1).sum()

    print_success(f"Created {n_clusters} spatial clusters")
    print_info(f"Noise points: {n_noise:,} ({n_noise/len(df)*100:.1f}%)")

    # Cluster statistics
    cluster_counts = df[df['spatial_cluster'] >= 0].groupby('spatial_cluster').size()
    print_info(f"Cluster sizes: min={cluster_counts.min()}, max={cluster_counts.max()}, "
              f"median={cluster_counts.median():.0f}")

    # ============================================================================
    # 3. TEMPORAL ORDERING & TARGET CREATION
    # ============================================================================
    print_step(3, 6, "Temporal ordering & target creation")

    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)

    # Create targets
    df['time_to_next'] = df['time'].shift(-1) - df['time']
    df['next_mag'] = df['mag'].shift(-1)

    # Convert to numeric
    df['time_to_next_days'] = df['time_to_next'].dt.total_seconds() / 86400

    # Remove last row (no target)
    df = df[:-1].copy()

    print_success("Created targets")
    print_info(f"time_to_next: mean={df['time_to_next_days'].mean():.3f}, "
              f"median={df['time_to_next_days'].median():.3f} days")
    print_info(f"next_mag: mean={df['next_mag'].mean():.3f}, "
              f"median={df['next_mag'].median():.3f}")

    # ============================================================================
    # 4. TIME SINCE LAST FEATURE
    # ============================================================================
    print_step(4, 6, "Calculating time_since_last")

    df['time_since_last'] = calculate_time_since_last(df)

    print_success("Calculated time_since_last")
    print_info(f"time_since_last: mean={df['time_since_last'].mean():.3f}, "
              f"median={df['time_since_last'].median():.3f} days")

    # ============================================================================
    # 5. LOG TRANSFORM TARGET
    # ============================================================================
    print_step(5, 6, "Log transform target")

    df['log_time_to_next_days'] = np.log1p(np.maximum(0, df['time_to_next_days']))

    print_info("Created log_time_to_next_days for better modeling")

    # ============================================================================
    # 6. TRAIN/VAL/TEST SPLIT & SAVE
    # ============================================================================
    print_step(6, 6, "Train/Val/Test split & save")

    # Time-based split (60-20-20)
    n = len(df)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    print_success(f"Split datasets")
    print_info(f"Train: {len(train):,} (60%)")
    print_info(f"Val: {len(val):,} (20%)")
    print_info(f"Test: {len(test):,} (20%)")

    # Save to data directory
    output_dir = Path(os.path.dirname(os.path.abspath(__file__))) / 'data'
    output_dir.mkdir(exist_ok=True)

    # Save as CSV cho inspection - với 17 advanced features
    columns_to_save = [
        # Basic
        'time', 'latitude', 'longitude', 'depth', 'mag',
        # Clustering & Time
        'spatial_cluster', 'time_since_last',
        # Targets
        'log_time_to_next_days', 'time_to_next_days', 'next_mag',
        # Base Advanced Features (9)
        'is_aftershock', 'mainshock_mag',
        'dist_to_5th_neighbor_km', 'dist_to_10th_neighbor_km',
        'seismicity_density_100km', 'coulomb_stress_proxy',
        'regional_b_value', 'seismic_gap_days', 'regional_max_mag_5yr',
        # Stress Tensor Features (5) - NEW
        'stress_sigma_1_mpa', 'stress_sigma_3_mpa', 'stress_tau_max_mpa',
        'stress_rate_mpa_per_year', 'stress_drop_recent_mpa',
        # Fault Geometry Features (4) - NEW
        'fault_depth_km', 'fault_strike_deg', 'fault_dip_deg', 'fault_length_km'
    ]

    print_info("Saving train.csv...")
    train[columns_to_save].to_csv(output_dir / 'train.csv', index=False)

    print_info("Saving val.csv...")
    val[columns_to_save].to_csv(output_dir / 'val.csv', index=False)

    print_info("Saving test.csv...")
    test[columns_to_save].to_csv(output_dir / 'test.csv', index=False)

    # Save config
    config = {
        'total_events': len(df),
        'train_events': len(train),
        'val_events': len(val),
        'test_events': len(test),
        'num_clusters': int(n_clusters),
        'cluster_eps_km': eps_km,
        'columns': columns_to_save,
        'target_time': 'log_time_to_next_days',
        'target_mag': 'next_mag'
    }

    with open(output_dir / 'data_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print_success("Saved all data files!")

    # ============================================================================
    # SUMMARY
    # ============================================================================
    print("\n" + "=" * 70)
    print(" SUMMARY ".center(70))
    print("=" * 70)
    print(f"\n  Total events: {len(df):,}")
    print(f"  Spatial clusters: {n_clusters}")
    print(f"\n  Split:")
    print(f"    Train: {len(train):,} (60%)")
    print(f"    Val:   {len(val):,} (20%)")
    print(f"    Test:  {len(test):,} (20%)")
    print(f"\n  Features created:")
    print(f"    - spatial_cluster: DBSCAN clustering ({eps_km}km radius)")
    print(f"    - time_since_last: days since last event in same cluster")
    print(f"    - log_time_to_next_days: log-transformed target")
    print(f"\n  Next step:")
    print(f"    python 02_feature_engineering.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
