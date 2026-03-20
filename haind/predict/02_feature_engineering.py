"""
=============================================================================
FEATURE ENGINEERING - LSTM EARTHQUAKE PREDICTION (PROPER FIX)
=============================================================================

Mục tiêu:
    - Tạo additional features từ base features
    - Log transforms cho skewed distributions
    - Rolling window statistics
    - Cluster-based features
    - NEW: Feature normalization/standardization

Proper Fix - Cải tiến:
    - Simpler, cleaner feature engineering
    - Features align với sequence model requirements
    - Efficient memory usage
    - NEW: Proper feature scaling cho LSTM training

Input:
    - data/train.csv, val.csv, test.csv: Base data from step 1

Output:
    - data/train_features.npz: Enhanced training features (normalized)
    - data/val_features.npz: Enhanced validation features (normalized)
    - data/test_features.npz: Enhanced test features (normalized)
    - data/features.json: Feature configuration
    - data/scaler_params.json: Normalization parameters

Tác giả: Haind
Ngày tạo: 2025-03-20
Cập nhật: 2026-03-20 (Added feature normalization)
=============================================================================
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


def print_step(step, total, description):
    print(f"\n[{step}/{total}] {description}...")


def print_success(message):
    print(f"  ✓ {message}")


def print_info(message):
    print(f"  ℹ {message}")


def add_log_transforms(df):
    """Log transform cho skewed features"""
    df = df.copy()

    # Log transforms cho positive features
    df['log_coulomb_stress'] = np.log1p(np.maximum(0, df['coulomb_stress_proxy']))
    df['log_seismicity_density'] = np.log1p(np.maximum(0, df['seismicity_density_100km']))
    df['log_dist_5th'] = np.log1p(np.maximum(0, df['dist_to_5th_neighbor_km']))
    df['log_dist_10th'] = np.log1p(np.maximum(0, df['dist_to_10th_neighbor_km']))
    df['log_time_since_last'] = np.log1p(np.maximum(0, df['time_since_last']))

    # Log transforms cho NEW stress tensor features
    df['log_stress_sigma_1'] = np.log1p(np.maximum(0, df['stress_sigma_1_mpa']))
    df['log_stress_tau_max'] = np.log1p(np.maximum(0, df['stress_tau_max_mpa']))
    df['log_stress_rate'] = np.log1p(np.maximum(0, df['stress_rate_mpa_per_year']))

    # Log transforms cho NEW fault geometry features
    df['log_fault_length'] = np.log1p(np.maximum(0, df['fault_length_km']))

    return df


def add_rolling_features(df, window=10):
    """
    Rolling window features theo temporal order
    Captures local seismic patterns
    """
    df = df.copy()
    df = df.sort_values('time').reset_index(drop=True)

    # Rolling statistics
    df['rolling_mag_mean'] = df['mag'].rolling(window=window, min_periods=1).mean()
    df['rolling_mag_std'] = df['mag'].rolling(window=window, min_periods=1).std().fillna(0)
    df['rolling_depth_mean'] = df['depth'].rolling(window=window, min_periods=1).mean()

    # Cumulative features
    df['cumsum_mag'] = df['mag'].cumsum()

    return df


def add_cluster_features(df):
    """
    Cluster-based statistics
    """
    df = df.copy()

    # Per-cluster statistics
    cluster_stats = df.groupby('spatial_cluster').agg({
        'mag': ['mean', 'std', 'count'],
        'time_since_last': ['mean', 'max']
    }).reset_index()

    cluster_stats.columns = ['spatial_cluster', 'cluster_mag_mean', 'cluster_mag_std',
                              'cluster_count', 'cluster_time_since_last_mean',
                              'cluster_time_since_last_max']

    # Merge back
    df = df.merge(cluster_stats, on='spatial_cluster', how='left')

    return df


def add_interaction_features(df):
    """Interaction features"""
    df = df.copy()

    # Key interactions
    df['mag_x_time_since_last'] = df['mag'] * np.log1p(df['time_since_last'])
    df['stress_x_density'] = df['coulomb_stress_proxy'] * df['seismicity_density_100km']
    df['mag_x_regional_max'] = df['mag'] * df['regional_max_mag_5yr']

    return df


def prepare_features(df, is_training=True):
    """
    Prepare all features for a dataframe
    """
    df = df.copy()

    # 1. Log transforms
    df = add_log_transforms(df)

    # 2. Rolling features (only if we have enough data)
    if len(df) > 10:
        df = add_rolling_features(df)

    # 3. Cluster features
    df = add_cluster_features(df)

    # 4. Interaction features
    df = add_interaction_features(df)

    # 5. Fill missing values
    df['regional_b_value'] = df['regional_b_value'].fillna(1.0)
    df['seismic_gap_days'] = df['seismic_gap_days'].fillna(3650)
    df['mainshock_mag'] = df['mainshock_mag'].fillna(df['mag'])
    df['cluster_mag_std'] = df['cluster_mag_std'].fillna(0)

    # NEW: Fill missing values cho stress tensor features
    df['stress_sigma_1_mpa'] = df['stress_sigma_1_mpa'].fillna(100.0)
    df['stress_sigma_3_mpa'] = df['stress_sigma_3_mpa'].fillna(30.0)
    df['stress_tau_max_mpa'] = df['stress_tau_max_mpa'].fillna(35.0)
    df['stress_rate_mpa_per_year'] = df['stress_rate_mpa_per_year'].fillna(0.0)
    df['stress_drop_recent_mpa'] = df['stress_drop_recent_mpa'].fillna(0.0)

    # NEW: Fill missing values cho fault geometry features
    df['fault_depth_km'] = df['fault_depth_km'].fillna(df['depth'])
    df['fault_strike_deg'] = df['fault_strike_deg'].fillna(0.0)
    df['fault_dip_deg'] = df['fault_dip_deg'].fillna(90.0)
    df['fault_length_km'] = df['fault_length_km'].fillna(10.0)

    return df


def get_numeric_features():
    """
    Returns list of numeric features that should be normalized
    (excluding boolean and categorical features)
    """
    return [
        # Basic numeric features
        'latitude', 'longitude', 'depth', 'mag',

        # Continuous advanced features
        'mainshock_mag',
        'dist_to_5th_neighbor_km', 'dist_to_10th_neighbor_km',
        'seismicity_density_100km', 'coulomb_stress_proxy',
        'regional_b_value', 'seismic_gap_days', 'regional_max_mag_5yr',
        'time_since_last',

        # Stress tensor features
        'stress_sigma_1_mpa', 'stress_sigma_3_mpa', 'stress_tau_max_mpa',
        'stress_rate_mpa_per_year', 'stress_drop_recent_mpa',

        # Fault geometry features
        'fault_depth_km', 'fault_strike_deg', 'fault_dip_deg', 'fault_length_km',

        # Log transformed features
        'log_coulomb_stress', 'log_seismicity_density',
        'log_dist_5th', 'log_dist_10th', 'log_time_since_last',
        'log_stress_sigma_1', 'log_stress_tau_max', 'log_stress_rate',
        'log_fault_length',

        # Rolling features
        'rolling_mag_mean', 'rolling_mag_std', 'rolling_depth_mean',

        # Cluster features (numeric only)
        'cluster_mag_mean', 'cluster_mag_std', 'cluster_count',

        # Interaction features
        'mag_x_time_since_last', 'stress_x_density', 'mag_x_regional_max'
    ]


def get_categorical_features():
    """
    Returns list of categorical features (should NOT be normalized)
    """
    return ['spatial_cluster', 'is_aftershock']


def normalize_features(train_df, val_df, test_df):
    """
    Normalize/standardize features using StandardScaler
    Fit on train data only, then transform val and test

    Returns:
        train_norm, val_norm, test_norm: normalized dataframes
        scaler_params: dict with mean and std for each feature
    """
    numeric_features = get_numeric_features()
    categorical_features = get_categorical_features()

    # Initialize scaler
    scaler = StandardScaler()

    # Fit on training data
    scaler.fit(train_df[numeric_features])

    # Transform all datasets
    train_norm = train_df.copy()
    val_norm = val_df.copy()
    test_norm = test_df.copy()

    train_norm[numeric_features] = scaler.transform(train_df[numeric_features])
    val_norm[numeric_features] = scaler.transform(val_df[numeric_features])
    test_norm[numeric_features] = scaler.transform(test_df[numeric_features])

    # Store scaler parameters
    scaler_params = {
        feature: {'mean': float(scaler.mean_[i]), 'std': float(scaler.scale_[i])}
        for i, feature in enumerate(numeric_features)
    }

    return train_norm, val_norm, test_norm, scaler_params


def get_feature_columns():
    """
    Define feature columns for model - CẬP NHẬT với 17 advanced features
    """
    return [
        # Basic features
        'latitude', 'longitude', 'depth', 'mag',

        # Aftershock features
        'is_aftershock', 'mainshock_mag',

        # Fault proximity
        'dist_to_5th_neighbor_km', 'dist_to_10th_neighbor_km',
        'seismicity_density_100km',

        # Coulomb stress
        'coulomb_stress_proxy',

        # Regional features
        'regional_b_value', 'seismic_gap_days', 'regional_max_mag_5yr',

        # NEW: Stress Tensor features (5)
        'stress_sigma_1_mpa', 'stress_sigma_3_mpa', 'stress_tau_max_mpa',
        'stress_rate_mpa_per_year', 'stress_drop_recent_mpa',

        # NEW: Fault Geometry features (4)
        'fault_depth_km', 'fault_strike_deg', 'fault_dip_deg', 'fault_length_km',

        # Cluster features
        'spatial_cluster', 'cluster_mag_mean', 'cluster_mag_std', 'cluster_count',

        # Time features
        'time_since_last',

        # Log transforms
        'log_coulomb_stress', 'log_seismicity_density',
        'log_dist_5th', 'log_dist_10th', 'log_time_since_last',

        # NEW: Log transforms cho stress tensor features
        'log_stress_sigma_1', 'log_stress_tau_max', 'log_stress_rate',

        # NEW: Log transforms cho fault geometry features
        'log_fault_length',

        # Rolling features
        'rolling_mag_mean', 'rolling_mag_std', 'rolling_depth_mean',

        # Interaction features
        'mag_x_time_since_last', 'stress_x_density', 'mag_x_regional_max'
    ]


    # ============================================================================
    # 5. PREPARE FEATURE MATRICES
    # ============================================================================
    print_step(2, 4, "Adding features")

    print_info("Processing train...")
    train = prepare_features(train)

    print_info("Processing val...")
    val = prepare_features(val)

    print_info("Processing test...")
    test = prepare_features(test)

    print_success("Added all features")

    # ============================================================================
    # 3. NORMALIZE FEATURES - NEW
    # ============================================================================
    print_step(3, 5, "Normalizing features (StandardScaler)")

    print_info("Fitting StandardScaler on train data...")
    train_norm, val_norm, test_norm, scaler_params = normalize_features(train, val, test)

    print_success("Normalized all features")
    print_info(f"Scaler parameters saved for {len(scaler_params)} numeric features")

    # ============================================================================
    # 4. PREPARE FEATURE MATRICES
    # ============================================================================
    print_step(4, 5, "Preparing feature matrices")

    feature_columns = get_feature_columns()

    # Convert to numpy arrays for efficiency (using NORMALIZED data)
    train_features = train_norm[feature_columns].values.astype(np.float32)
    val_features = val_norm[feature_columns].values.astype(np.float32)
    test_features = test_norm[feature_columns].values.astype(np.float32)

    # Targets
    train_time_target = train_norm['log_time_to_next_days'].values.astype(np.float32)
    val_time_target = val_norm['log_time_to_next_days'].values.astype(np.float32)
    test_time_target = test_norm['log_time_to_next_days'].values.astype(np.float32)

    train_mag_target = train_norm['next_mag'].values.astype(np.float32)
    val_mag_target = val_norm['next_mag'].values.astype(np.float32)
    test_mag_target = test_norm['next_mag'].values.astype(np.float32)

    # Additional metadata (for sequence creation) - use normalized data for spatial_cluster
    train_metadata = {
        'time': pd.to_datetime(train_norm['time']).values,
        'spatial_cluster': train_norm['spatial_cluster'].values,
        'time_to_next_days': train_norm['time_to_next_days'].values
    }

    val_metadata = {
        'time': pd.to_datetime(val_norm['time']).values,
        'spatial_cluster': val_norm['spatial_cluster'].values,
        'time_to_next_days': val_norm['time_to_next_days'].values
    }

    test_metadata = {
        'time': pd.to_datetime(test_norm['time']).values,
        'spatial_cluster': test_norm['spatial_cluster'].values,
        'time_to_next_days': test_norm['time_to_next_days'].values
    }

    print_success("Prepared feature matrices (NORMALIZED)")
    print_info(f"Features: {len(feature_columns)}")
    print_info(f"Shape: {train_features.shape}")
    print_info(f"Feature range: mean={train_features.mean():.3f}, std={train_features.std():.3f}")

    # ============================================================================
    # 5. SAVE FEATURES
    # ============================================================================
    print_step(5, 5, "Saving features")

    print_info("Saving train_features.npz...")
    np.savez_compressed(
        data_dir / 'train_features.npz',
        features=train_features,
        time_target=train_time_target,
        mag_target=train_mag_target,
        time=train_metadata['time'].astype('datetime64[s]').astype(np.int64),
        spatial_cluster=train_metadata['spatial_cluster'],
        time_to_next_days=train_metadata['time_to_next_days']
    )

    print_info("Saving val_features.npz...")
    np.savez_compressed(
        data_dir / 'val_features.npz',
        features=val_features,
        time_target=val_time_target,
        mag_target=val_mag_target,
        time=val_metadata['time'].astype('datetime64[s]').astype(np.int64),
        spatial_cluster=val_metadata['spatial_cluster'],
        time_to_next_days=val_metadata['time_to_next_days']
    )

    print_info("Saving test_features.npz...")
    np.savez_compressed(
        data_dir / 'test_features.npz',
        features=test_features,
        time_target=test_time_target,
        mag_target=test_mag_target,
        time=test_metadata['time'].astype('datetime64[s]').astype(np.int64),
        spatial_cluster=test_metadata['spatial_cluster'],
        time_to_next_days=test_metadata['time_to_next_days']
    )

    # Save feature config với categories mới và scaler parameters
    config = {
        'feature_columns': feature_columns,
        'num_features': len(feature_columns),
        'target_time': 'log_time_to_next_days',
        'target_mag': 'next_mag',
        'normalized': True,  # NEW: features đã được chuẩn hóa
        'scaler_params': scaler_params,  # NEW: để inverse transform khi cần
        'feature_categories': {
            'basic': ['latitude', 'longitude', 'depth', 'mag'],
            'aftershock': ['is_aftershock', 'mainshock_mag'],
            'fault': ['dist_to_5th_neighbor_km', 'dist_to_10th_neighbor_km', 'seismicity_density_100km'],
            'stress': ['coulomb_stress_proxy'],
            'regional': ['regional_b_value', 'seismic_gap_days', 'regional_max_mag_5yr'],
            'stress_tensor': ['stress_sigma_1_mpa', 'stress_sigma_3_mpa', 'stress_tau_max_mpa',
                           'stress_rate_mpa_per_year', 'stress_drop_recent_mpa'],
            'fault_geometry': ['fault_depth_km', 'fault_strike_deg', 'fault_dip_deg', 'fault_length_km'],
            'cluster': ['spatial_cluster', 'cluster_mag_mean', 'cluster_mag_std', 'cluster_count'],
            'time': ['time_since_last'],
            'log_transform': ['log_coulomb_stress', 'log_seismicity_density', 'log_dist_5th', 'log_dist_10th', 'log_time_since_last'],
            'log_transform_stress': ['log_stress_sigma_1', 'log_stress_tau_max', 'log_stress_rate'],
            'log_transform_fault': ['log_fault_length'],
            'rolling': ['rolling_mag_mean', 'rolling_mag_std', 'rolling_depth_mean'],
            'interaction': ['mag_x_time_since_last', 'stress_x_density', 'mag_x_regional_max']
        }
    }

    # Save scaler parameters separately for easy access
    with open(data_dir / 'scaler_params.json', 'w') as f:
        json.dump(scaler_params, f, indent=2)

    with open(data_dir / 'features.json', 'w') as f:
        json.dump(config, f, indent=2)

    print_success("Saved all feature files!")

    # ============================================================================
    # SUMMARY
    # ============================================================================
    print("\n" + "=" * 70)
    print(" SUMMARY ".center(70))
    print("=" * 70)
    print(f"\n  Features: {len(feature_columns)}")
    print(f"\n  Categories:")
    for category, features in config['feature_categories'].items():
        print(f"    {category}: {len(features)}")
    print(f"\n  NEW: Feature Normalization")
    print(f"    - Method: StandardScaler (mean=0, std=1)")
    print(f"    - Fit on: Training data only")
    print(f"    - Applied to: {len(scaler_params)} numeric features")
    print(f"    - Saved to: scaler_params.json")
    print(f"\n  Output files:")
    print(f"    - {data_dir / 'train_features.npz'} (normalized)")
    print(f"    - {data_dir / 'val_features.npz'} (normalized)")
    print(f"    - {data_dir / 'test_features.npz'} (normalized)")
    print(f"    - {data_dir / 'features.json'}")
    print(f"    - {data_dir / 'scaler_params.json'} (NEW)")
    print(f"\n  Next step:")
    print(f"    python 03_sequence_data.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
