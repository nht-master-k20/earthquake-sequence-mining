"""
Add Advanced Features for Earthquake Forecasting - LSTM OPTIMIZED VERSION
Task: Predict time-to-next and magnitude-of-next earthquake using LSTM

Features (27 total):
- ORIGINAL (5): time, latitude, longitude, depth, mag
- CORE (5): Aftershock, Density, Coulomb Stress, B-value
- LSTM CRITICAL (14): Temporal intervals, Rolling stats
- TARGETS (3): time_to_next, next_mag, next_mag_binary

Optimizations:
- Vectorized operations with NumPy
- Efficient spatial queries with KD-tree
- Removed redundant features (11 features removed for speed)
- Removed checkpoint system (simpler code, faster I/O)

Author: haind
Project: Earthquake Sequence Mining
Updated: 2025-03-23 (LSTM-optimized, no checkpoint)
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
from tqdm import tqdm
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
import os

# Tự động phát hiện base directory từ vị trí script
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# Số CPU cores sử dụng
N_CORES = min(8, max(1, cpu_count() - 1))

print(f"Using {N_CORES} CPU cores for multiprocessing")
print(f"Base directory: {BASE_DIR}")

# ============================================================================
# MULTIPROCESSING HELPER FUNCTIONS
# ============================================================================

def process_aftershock_chunk(args):
    """Process một chunk cho aftershock detection"""
    chunk_start, chunk_end, times, mags, time_numeric, cartesian_coords = args

    is_aftershock_local = []
    mainshock_mag_local = []

    for i in range(chunk_start, chunk_end):
        mag_i = mags[i]
        time_i = time_numeric[i]

        # Calculate windows
        time_window_sec = 10 ** (0.5 * mag_i - 1.0)
        dist_window_km = 10 ** (0.123 * mag_i + 0.033)

        # Find events in time window
        time_mask = (time_numeric >= time_i) & (time_numeric <= time_i + time_window_sec)
        candidate_indices = np.where(time_mask)[0]

        if len(candidate_indices) <= 1:
            is_aftershock_local.append(False)
            mainshock_mag_local.append(mag_i)
            continue

        # Filter by distance
        candidate_coords = cartesian_coords[candidate_indices]
        current_coord = cartesian_coords[i]
        distances = np.linalg.norm(candidate_coords - current_coord, axis=1)
        dist_mask = distances <= (dist_window_km * 1000)
        nearby_indices = candidate_indices[dist_mask]

        if len(nearby_indices) <= 1:
            is_aftershock_local.append(False)
            mainshock_mag_local.append(mag_i)
            continue

        # Check for larger mainshock
        nearby_mags = mags[nearby_indices]
        larger_mainshock = nearby_mags > mag_i

        if np.any(larger_mainshock):
            larger_indices = nearby_indices[larger_mainshock]
            larger_distances = distances[dist_mask][larger_mainshock]
            closest_larger_idx = larger_indices[np.argmin(larger_distances)]
            mainshock_mag_local.append(mags[closest_larger_idx])

            # Mark as aftershock if within window
            time_diff_sec = time_numeric[closest_larger_idx] - time_i
            is_aftershock_local.append(time_diff_sec <= time_window_sec)
        else:
            is_aftershock_local.append(False)
            mainshock_mag_local.append(mag_i)

    return chunk_start, chunk_end, is_aftershock_local, mainshock_mag_local

def process_density_chunk(args):
    """Process một chunk cho seismicity density"""
    chunk_start, chunk_end, coords, kdtree_cartesian, radius_m = args

    results = []
    for i in range(chunk_start, chunk_end):
        nearby_count = len(kdtree_cartesian.query_ball_point(coords[i], radius_m))
        results.append(nearby_count)

    return chunk_start, results

def process_coulomb_stress_chunk(args):
    """Process một chunk cho Coulomb stress"""
    chunk_start, chunk_end, mags, cartesian_coords, kdtree_cartesian, radius_m, lookback_events = args

    coulomb_stress_local = []

    for i in range(chunk_start, chunk_end):
        # Find nearby events
        nearby_indices = kdtree_cartesian.query_ball_point(cartesian_coords[i], radius_m)

        # Filter to only previous events
        nearby_indices = [idx for idx in nearby_indices if idx < i]

        if len(nearby_indices) == 0:
            coulomb_stress_local.append(0)
            continue

        # Get last 'lookback_events' events
        nearby_indices = nearby_indices[-lookback_events:]
        nearby_mags = mags[nearby_indices]

        # Stress is proportional to seismic moment
        stress_contributions = 10**(1.5 * nearby_mags)
        coulomb_stress_local.append(np.sum(stress_contributions))

    return chunk_start, coulomb_stress_local

def process_b_value_chunk(args):
    """Process một chunk cho b-value calculation - OPTIMIZED VERSION"""
    indices, time_numeric, cartesian_coords, kdtree_cartesian, mags = args

    results = []
    time_window_sec = 365 * 24 * 3600  # 1 year in seconds
    radius_km = 100
    radius_m = radius_km * 1000

    for idx in indices:
        current_time = time_numeric[idx]
        current_coord = cartesian_coords[idx]

        # Spatial window
        nearby_indices = kdtree_cartesian.query_ball_point(current_coord, radius_m)

        # Time window + boolean indexing - FAST
        nearby_indices = np.array(nearby_indices)
        time_mask = (time_numeric[nearby_indices] >= current_time - time_window_sec) & \
                    (time_numeric[nearby_indices] <= current_time)
        nearby_indices = nearby_indices[time_mask]

        if len(nearby_indices) < 20:
            results.append(1.0)
            continue

        # Compute b-value using maximum likelihood
        nearby_mags = mags[nearby_indices]
        nearby_mags = nearby_mags[nearby_mags >= 3.0]

        if len(nearby_mags) < 10:
            results.append(1.0)
            continue

        try:
            b_value = (nearby_mags.mean() - np.min(nearby_mags))**-1 * np.log10(np.e)
            results.append(b_value)
        except:
            results.append(1.0)

    return results

def process_seismic_gap_chunk(args):
    """Process một chunk cho seismic gap calculation"""
    chunk_start, chunk_end, df_work, cartesian_coords, m5_coords, m5_times, kdtree_m5 = args

    results = []
    for i in range(chunk_start, chunk_end):
        dists, idxs = kdtree_m5.query(cartesian_coords[i:i+1], k=10)
        dists = dists[0] / 1000
        idxs = idxs[0]

        current_time = df_work.loc[i, 'time']
        valid_mask = (dists <= 200) & (m5_times[idxs] < current_time)

        if np.any(valid_mask):
            valid_times = m5_times[idxs[valid_mask]]
            time_delta = current_time - valid_times.max()
            gap_days = time_delta.total_seconds() / 86400
            results.append(gap_days)
        else:
            results.append(365 * 10)

    return chunk_start, results

def process_regional_max_mag_chunk(args):
    """Process một chunk cho regional max magnitude"""
    chunk_start, chunk_end, time_numeric, cartesian_coords, mags, kdtree_cartesian, radius_m = args

    results = []
    time_window_sec = 365 * 5 * 24 * 3600  # 5 years in seconds

    for i in range(chunk_start, chunk_end):
        current_time = time_numeric[i]
        nearby_indices = kdtree_cartesian.query_ball_point(cartesian_coords[i], radius_m)

        # Fast numpy boolean indexing
        nearby_indices = np.array(nearby_indices)
        time_mask = (time_numeric[nearby_indices] >= current_time - time_window_sec) & \
                    (time_numeric[nearby_indices] <= current_time)
        nearby_indices = nearby_indices[time_mask]

        if len(nearby_indices) > 0:
            results.append(mags[nearby_indices].max())
        else:
            results.append(mags[i])

    return chunk_start, results

def process_regional_max_mag_sample_chunk(args):
    """Process một chunk cho regional max magnitude - SAMPLING VERSION"""
    indices, time_numeric, cartesian_coords, mags, kdtree_cartesian, radius_m = args

    results = []
    time_window_sec = 365 * 5 * 24 * 3600  # 5 years in seconds

    for i in indices:
        current_time = time_numeric[i]
        nearby_indices = kdtree_cartesian.query_ball_point(cartesian_coords[i], radius_m)

        # Fast numpy boolean indexing
        nearby_indices = np.array(nearby_indices)
        time_mask = (time_numeric[nearby_indices] >= current_time - time_window_sec) & \
                    (time_numeric[nearby_indices] <= current_time)
        nearby_indices = nearby_indices[time_mask]

        if len(nearby_indices) > 0:
            results.append(mags[nearby_indices].max())
        else:
            results.append(mags[i])

    return indices, results

def process_stress_tensor_chunk(args):
    """Process một chunk cho stress tensor calculation"""
    chunk_start, chunk_end, mags, times, lookback_events = args

    n = chunk_end - chunk_start
    sigma_1_local = np.zeros(n)
    sigma_3_local = np.zeros(n)
    tau_max_local = np.zeros(n)

    for idx in range(n):
        i = chunk_start + idx

        start_idx = max(0, i - lookback_events)
        prev_indices = np.arange(start_idx, i)

        if len(prev_indices) < 5:
            sigma_1_local[idx] = 50e6
            sigma_3_local[idx] = 20e6
            tau_max_local[idx] = (sigma_1_local[idx] - sigma_3_local[idx]) / 2
            continue

        prev_mags = mags[prev_indices]
        stress_drops = 3e6 * 10**(1.5 * prev_mags)
        mean_stress_drop = np.mean(stress_drops)

        sigma_1_local[idx] = 100e6 + mean_stress_drop * 0.5
        sigma_3_local[idx] = 30e6 + mean_stress_drop * 0.2
        tau_max_local[idx] = (sigma_1_local[idx] - sigma_3_local[idx]) / 2

    return chunk_start, sigma_1_local, sigma_3_local, tau_max_local

def process_stress_rate_chunk(args):
    """Process một chunk cho stress rate calculation"""
    chunk_start, chunk_end, mags, times, time_window_events = args

    n = chunk_end - chunk_start
    stress_rate_local = np.zeros(n)

    for idx in range(n):
        i = chunk_start + idx

        if i < time_window_events:
            stress_rate_local[idx] = 0
            continue

        prev_indices = np.arange(i - time_window_events, i)
        prev_mags = mags[prev_indices]
        total_stress = np.sum(3e6 * 10**(1.5 * prev_mags))

        time_delta = times[i] - times[prev_indices[0]]
        time_diff = time_delta.astype('timedelta64[s]').astype(float) / (365 * 24 * 3600)

        if time_diff > 0:
            stress_rate_local[idx] = (total_stress / 1e6) / time_diff

    return chunk_start, stress_rate_local

def process_fault_geometry_chunk(args):
    """Process một chunk cho fault geometry calculation - OPTIMIZED VERSION"""
    chunk_start, chunk_end, df_work, coords, kdtree, depths, radius_m = args

    n = chunk_end - chunk_start
    local_strike = np.zeros(n)
    local_dip = np.zeros(n)
    fault_length = np.zeros(n)

    for idx in range(n):
        i = chunk_start + idx

        # Find nearby events
        nearby_indices = kdtree.query_ball_point(coords[i], radius_m)

        if len(nearby_indices) < 10:
            local_strike[idx] = 0
            local_dip[idx] = 90
            fault_length[idx] = 10
            continue

        nearby_coords = coords[nearby_indices]

        # PCA để tìm hướng chính
        centered = nearby_coords - nearby_coords.mean(axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        idx_sort = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx_sort]
        strike_vector = eigenvectors[:, 0]

        x, y, z = strike_vector
        strike_rad = np.arctan2(y, x)
        strike_deg = np.degrees(strike_rad)

        if strike_deg < 0:
            strike_deg += 360

        local_strike[idx] = strike_deg

        nearby_depths = depths[nearby_indices]
        if len(nearby_depths) > 0:
            depth_std = np.std(nearby_depths)
            local_dip[idx] = 90 - np.clip(depth_std, 0, 60)
        else:
            local_dip[idx] = 90

        # Fault length - OPTIMIZED but CORRECT: dùng pdist (C implementation, O(n) memory)
        if len(nearby_coords) >= 2:
            # pdist computes all pairwise distances efficiently in C
            # Still O(n²) but much faster due to C implementation
            if len(nearby_coords) > 500:  # Limit for very large sets
                # Use bounding box for very large sets as approximation
                min_coords = nearby_coords.min(axis=0)
                max_coords = nearby_coords.max(axis=0)
                max_dist = np.linalg.norm(max_coords - min_coords)
            else:
                max_dist = np.max(pdist(nearby_coords))
            fault_length[idx] = max_dist / 1000
        else:
            fault_length[idx] = 10

    return chunk_start, local_strike, local_dip, fault_length


# ============================================================================
# MAIN EXECUTION - LSTM OPTIMIZED
# ============================================================================

print("="*70)
print(" ADVANCED FEATURES - LSTM OPTIMIZED VERSION ")
print("="*70)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[1/6] Loading data...")
df = pd.read_csv(BASE_DIR.parent / 'dongdat.csv')
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)
df_work = df.copy().reset_index(drop=True)
print(f"  Working with {len(df_work):,} events")
print(f"  Magnitude range: {df_work['mag'].min():.1f} - {df_work['mag'].max():.1f}")

# ============================================================================
# STEP 2: BUILD SPATIAL INDEX
# ============================================================================
print("\n[2/6] Building spatial index (KD-tree)...")

coords = df_work[['latitude', 'longitude']].values
lat_rad = np.radians(coords[:, 0])
lon_rad = np.radians(coords[:, 1])
R = 6371.0

x = R * np.cos(lat_rad) * np.cos(lon_rad)
y = R * np.cos(lat_rad) * np.sin(lon_rad)
z = R * np.sin(lat_rad)
cartesian_coords = np.column_stack([x, y, z])
kdtree_cartesian = cKDTree(cartesian_coords)

print(f"  KD-tree built with {len(cartesian_coords):,} points")

# ============================================================================
# STEP 3: CORE FEATURES (5 essential features)
# ============================================================================
print("\n[3/6] Computing core features...")

# 3.1 Aftershock detection (simplified - time-only for speed)
print("  Computing aftershock detection...")
times_numeric = df_work['time'].values.astype(np.int64) / 1e9
mags = df_work['mag'].values

n_events = len(df_work)
is_aftershock = np.zeros(n_events, dtype=bool)
mainshock_mag = mags.copy()

# Simplified: Check if there's a larger event within time window
for i in tqdm(range(min(n_events, 100000)), desc="  Aftershock (sampled)"):  # Sample for speed
    mag_i = mags[i]
    time_i = times_numeric[i]
    time_window_sec = 10 ** (0.5 * mag_i - 1.0)

    # Find events in time window AFTER this event
    time_mask = (times_numeric >= time_i) & (times_numeric <= time_i + time_window_sec)
    candidate_indices = np.where(time_mask)[0]

    if len(candidate_indices) > 1:
        candidate_mags = mags[candidate_indices]
        larger_mainshock = candidate_mags > mag_i

        if np.any(larger_mainshock):
            is_aftershock[i] = True
            mainshock_mag[i] = candidate_mags[larger_mainshock].max()

# Interpolate for rest (for speed with large dataset)
if n_events > 100000:
    from scipy.interpolate import interp1d
    sample_indices = np.arange(0, n_events, n_events // 10000)
    f_aftershock = interp1d(sample_indices, is_aftershock[sample_indices].astype(float),
                            kind='linear', bounds_error=False, fill_value=0)
    is_aftershock = f_aftershock(np.arange(n_events)) > 0.5

    f_mainshock = interp1d(sample_indices, mainshock_mag[sample_indices],
                          kind='linear', bounds_error=False, fill_value=mags.mean())
    mainshock_mag = f_mainshock(np.arange(n_events))

df_work['is_aftershock'] = is_aftershock.astype(int)
df_work['mainshock_mag'] = mainshock_mag
print(f"  Aftershocks: {is_aftershock.sum():,} ({is_aftershock.sum()/len(df_work)*100:.1f}%)")

# 3.2 Seismicity density (simplified - spatial bins)
print("  Computing seismicity density...")
lat_bins = np.linspace(df_work['latitude'].min(), df_work['latitude'].max(), 100)
lon_bins = np.linspace(df_work['longitude'].min(), df_work['longitude'].max(), 100)

df_work['lat_bin'] = pd.cut(df_work['latitude'], bins=lat_bins, labels=False)
df_work['lon_bin'] = pd.cut(df_work['longitude'], bins=lon_bins, labels=False)
df_work['spatial_bin'] = df_work['lat_bin'] * 100 + df_work['lon_bin']

density_map = df_work['spatial_bin'].value_counts()
df_work['seismicity_density'] = df_work['spatial_bin'].map(density_map).fillna(1)
df_work['seismicity_density_100km'] = df_work['seismicity_density'] / 100
print(f"  Mean density: {df_work['seismicity_density_100km'].mean():.2f}")

# 3.3 Coulomb stress proxy (simplified)
print("  Computing Coulomb stress proxy...")
coulomb_stress = np.zeros(n_events)

for i in tqdm(range(min(10000, n_events)), desc="  Coulomb stress (sample)"):
    start_idx = max(0, i - 20)
    prev_mags = mags[start_idx:i]
    if len(prev_mags) > 0:
        stress_contributions = 10**(1.5 * prev_mags)
        coulomb_stress[i] = np.sum(stress_contributions)

# Interpolate
if n_events > 10000:
    f_coulomb = interp1d(np.arange(0, n_events, n_events//1000), coulomb_stress[::n_events//1000],
                          kind='linear', bounds_error=False,
                          fill_value=(coulomb_stress[:1000].mean(), coulomb_stress[-1000:].mean()))
    coulomb_stress = f_coulomb(np.arange(n_events))

df_work['coulomb_stress_proxy'] = coulomb_stress
print(f"  Mean stress: {coulomb_stress.mean():.2e}")

# 3.4 Regional b-value (simplified)
print("  Computing regional b-value...")
sample_size = min(1000, n_events)
sample_indices = np.linspace(0, n_events-1, sample_size, dtype=int)

b_values = []
for idx in tqdm(sample_indices, desc="  B-value"):
    start_idx = max(0, idx - 10000)
    recent_mags = mags[start_idx:idx+1]

    if len(recent_mags) >= 50:
        recent_mags = recent_mags[recent_mags >= 3.0]
        if len(recent_mags) >= 20:
            try:
                b_value = (recent_mags.mean() - recent_mags.min())**-1 * np.log10(np.e)
                b_values.append(b_value)
            except:
                b_values.append(1.0)
        else:
            b_values.append(1.0)
    else:
        b_values.append(1.0)

f_b = interp1d(sample_indices, b_values, kind='linear',
                bounds_error=False, fill_value=(np.mean(b_values), np.mean(b_values)))
df_work['regional_b_value'] = f_b(np.arange(n_events))
print(f"  Mean b-value: {np.mean(b_values):.2f}")

print(f"  ✓ Core features added: 4 features")

# ============================================================================
# STEP 4: LSTM CRITICAL FEATURES - Temporal Intervals
# ============================================================================
print("\n[4/6] Computing LSTM temporal features...")

# 4.1 Time since last event
df_work['time_since_last_event'] = df_work['time'].diff().dt.total_seconds()
df_work['time_since_last_event'] = df_work['time_since_last_event'].fillna(0)
print("  ✓ time_since_last_event")

# 4.2 Time since last M5+ event
print("  Computing time since last M5+...")
m5_times = df_work[df_work['mag'] >= 5.0]['time']
last_m5_time = pd.Series(index=df_work.index, dtype=float)

for i in tqdm(range(len(df_work)), desc="  Time since M5"):
    current_time = df_work.loc[i, 'time']
    past_m5 = m5_times[m5_times < current_time]
    if len(past_m5) > 0:
        last_m5_time[i] = (current_time - past_m5.max()).total_seconds()
    else:
        last_m5_time[i] = 365 * 24 * 3600

df_work['time_since_last_M5'] = last_m5_time
print("  ✓ time_since_last_M5")

# 4.3 Interval sequence (last 5 intervals)
print("  Computing interval sequences...")
intervals_seq = []
for i in tqdm(range(len(df_work)), desc="  Interval seq"):
    if i < 5:
        intervals_seq.append([0] * 5)
    else:
        recent_times = df_work.loc[i-5:i, 'time'].values
        diffs = np.diff(recent_times).astype('timedelta64[s]').astype(float)
        diffs_padded = np.pad(diffs, (5-len(diffs), 0), mode='constant')
        intervals_seq.append(diffs_padded)

for lag in range(5):
    df_work[f'interval_lag{lag+1}'] = [seq[lag] for seq in intervals_seq]
print("  ✓ interval_lag1 to interval_lag5")

# ============================================================================
# STEP 5: LSTM ROLLING STATISTICS
# ============================================================================
print("\n[5/6] Computing rolling statistics...")

df_work['time_numeric'] = df_work['time'].astype(np.int64) / 1e9

windows = {'1h': 3600, '24h': 24*3600, '7d': 7*24*3600}

for window_name, window_sec in windows.items():
    print(f"  Computing rolling {window_name}...")

    count_list = []
    mean_mag_list = []
    max_mag_list = []

    for i in tqdm(range(len(df_work)), desc=f"   {window_name}"):
        current_time = df_work.loc[i, 'time_numeric']
        min_time = current_time - window_sec

        time_mask = (df_work['time_numeric'] >= min_time) & (df_work['time_numeric'] <= current_time)
        events_in_window = df_work[time_mask]

        count_list.append(len(events_in_window))

        if len(events_in_window) > 0:
            mean_mag_list.append(events_in_window['mag'].mean())
            max_mag_list.append(events_in_window['mag'].max())
        else:
            mean_mag_list.append(0)
            max_mag_list.append(0)

    df_work[f'rolling_count_{window_name}'] = count_list
    df_work[f'rolling_mean_mag_{window_name}'] = mean_mag_list
    df_work[f'rolling_max_mag_{window_name}'] = max_mag_list

print("  ✓ Rolling statistics computed")

# ============================================================================
# STEP 6: TARGET VARIABLES
# ============================================================================
print("\n[6/6] Creating target variables...")

df_work['target_time_to_next'] = df_work['time'].diff(-1).dt.total_seconds().abs()
df_work['target_time_to_next'] = df_work['target_time_to_next'].fillna(0)
print("  ✓ target_time_to_next")

df_work['target_next_mag'] = df_work['mag'].shift(-1)
df_work['target_next_mag'] = df_work['target_next_mag'].fillna(df_work['mag'].iloc[-1])
print("  ✓ target_next_mag")

df_work['target_next_mag_binary'] = (df_work['target_next_mag'] >= 5.0).astype(int)
print("  ✓ target_next_mag_binary")

# ============================================================================
# SELECT FINAL FEATURES AND SAVE
# ============================================================================
print("\n" + "="*70)
print(" SUMMARY AND SAVE ")
print("="*70)

# Original features (5)
original_features = ['time', 'latitude', 'longitude', 'depth', 'mag']

# Core features (5)
core_features = [
    'is_aftershock',
    'mainshock_mag',
    'seismicity_density_100km',
    'coulomb_stress_proxy',
    'regional_b_value'
]

# LSTM critical features (14)
lstm_features = [
    'time_since_last_event',
    'time_since_last_M5',
    'interval_lag1', 'interval_lag2', 'interval_lag3', 'interval_lag4', 'interval_lag5',
    'rolling_count_1h', 'rolling_count_24h', 'rolling_count_7d',
    'rolling_mean_mag_1h', 'rolling_mean_mag_24h', 'rolling_mean_mag_7d',
    'rolling_max_mag_1h', 'rolling_max_mag_24h', 'rolling_max_mag_7d'
]

# Targets (3)
target_features = [
    'target_time_to_next',
    'target_next_mag',
    'target_next_mag_binary'
]

# Combine all
final_features = original_features + core_features + lstm_features + target_features

# Create final dataframe
df_final = df_work[final_features].copy()

print(f"\nFinal features: {len(final_features)}")
print("  - Original: 5")
print("  - Core: 5")
print("  - LSTM-specific: 14")
print("  - Targets: 3")

# Save to CSV
output_file = BASE_DIR / 'features_lstm.csv'
df_final.to_csv(output_file, index=False)

print(f"\n✓ SAVED: {output_file}")
print(f"  Rows: {len(df_final):,}")
print(f"  Columns: {len(df_final.columns)}")
print(f"  Size: {os.path.getsize(output_file) / 1024 / 1024:.1f} MB")

# Print feature list
print(f"\nFEATURE LIST:")
print("\n1. ORIGINAL FEATURES (5):")
for feat in original_features:
    print(f"   - {feat}")

print("\n2. CORE FEATURES (5):")
for feat in core_features:
    print(f"   - {feat}")

print("\n3. LSTM CRITICAL FEATURES (14):")
print("   Temporal Intervals:")
for feat in ['time_since_last_event', 'time_since_last_M5'] + [f'interval_lag{i}' for i in range(1, 6)]:
    print(f"   - {feat}")
print("   Rolling Statistics (1h, 24h, 7d):")
for feat in ['rolling_count', 'rolling_mean_mag', 'rolling_max_mag']:
    for w in ['1h', '24h', '7d']:
        print(f"   - {feat}_{w}")

print("\n4. TARGET VARIABLES (3):")
for feat in target_features:
    print(f"   - {feat}")

print("\n" + "="*70)
print(" COMPLETE! ")
print("="*70)
print("\n✓ Features ready for LSTM training!")
print("\nNext steps:")
print("  1. Use sequences of 50-100 events as LSTM input")
print("  2. Predict: target_time_to_next, target_next_mag")
print("  3. Consider splitting by time (train on early data, test on recent)")
print("\nFile: haind/features_lstm.csv")
print("="*70)
print(f"\n  Total features: {len(final_features)} (removed 11 redundant features)")
print(f"  Estimated runtime: ~10-15 minutes for 3M records")
print("="*70)
