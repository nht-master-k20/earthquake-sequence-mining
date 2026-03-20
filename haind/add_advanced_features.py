"""
Add Advanced Features for Earthquake Forecasting - OPTIMIZED VERSION
Features: Fault Line, Aftershock Detection, Coulomb Stress, Regional Features
NEW: Stress Tensor Features, Fault Geometry Features

IMPORTANT: Process ALL magnitudes (not just M >= 3.0) to preserve foreshocks

Optimizations:
- Vectorized operations with NumPy
- Efficient spatial queries with KD-tree
- Progress bars for tracking
- Optimized Gardner-Knopoff algorithm

Author: haind
Project: Earthquake Sequence Mining
Updated: 2025-03-20 (Added Stress Tensor & Fault Geometry features)
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial import cKDTree
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print(" ADVANCED FEATURES FOR EARTHQUAKE FORECASTING (OPTIMIZED) ")
print("="*70)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1/7] Loading data...")
df = pd.read_csv('/home/haind/Desktop/earthquake-sequence-mining/dongdat.csv')
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)

# KHÔNG lọc magnitude - giữ lại toàn bộ data bao gồm cả foreshocks nhỏ
df_work = df.copy().reset_index(drop=True)
print(f"  Working with {len(df_work):,} events (ALL magnitudes)")
print(f"  Magnitude range: {df_work['mag'].min():.1f} - {df_work['mag'].max():.1f}")

# ============================================================================
# PRE-COMPUTE SPATIAL DATA WITH KD-TREE
# ============================================================================
print("\n[2/7] Building spatial index (KD-tree)...")

# Convert lat/lon to approximate Cartesian coordinates (for faster distance calc)
coords = df_work[['latitude', 'longitude']].values
lat_rad = np.radians(coords[:, 0])
lon_rad = np.radians(coords[:, 1])

# Earth radius in km
R = 6371.0

# Convert to 3D Cartesian for more accurate distance calculations
x = R * np.cos(lat_rad) * np.cos(lon_rad)
y = R * np.cos(lat_rad) * np.sin(lon_rad)
z = R * np.sin(lat_rad)
cartesian_coords = np.column_stack([x, y, z])

# Build KD-tree for efficient nearest neighbor searches
kdtree_cartesian = cKDTree(cartesian_coords)

print(f"  KD-tree built with {len(cartesian_coords):,} points")

# ============================================================================
# FEATURE 1: OPTIMIZED AFTERSHOCK DETECTION (Memory-Efficient)
# ============================================================================
print("\n[3/7] Aftershock detection (Gardner-Knopoff, memory-optimized)...")

# Get arrays
times = df_work['time'].values
mags = df_work['mag'].values

# Initialize arrays
is_aftershock = np.zeros(len(df_work), dtype=bool)
mainshock_mag = mags.copy()

# Convert times to numeric (seconds since epoch)
time_numeric = times.astype(np.int64) / 1e9

# Process events in chunks to avoid memory issues
chunk_size = 5000
n_events = len(df_work)

print("  Processing events in chunks...")
for chunk_start in tqdm(range(0, n_events, chunk_size), desc="  Declustering"):
    chunk_end = min(chunk_start + chunk_size, n_events)

    for i in range(chunk_start, chunk_end):
        if is_aftershock[i]:
            continue

        mag_i = mags[i]
        time_i = time_numeric[i]

        # Calculate windows
        time_window_sec = 10 ** (0.5 * mag_i - 1.0)
        dist_window_km = 10 ** (0.123 * mag_i + 0.034)
        dist_window_m = dist_window_km * 1000

        # Find candidates: events after i within time window
        # Use binary search for efficiency
        time_min = time_i + 0.0001  # Exclude self
        time_max = time_i + time_window_sec

        # Vectorized search using binary search
        candidates_start = np.searchsorted(time_numeric, time_min, side='left')
        candidates_end = np.searchsorted(time_numeric, time_max, side='right')

        if candidates_start >= candidates_end:
            continue

        candidates = np.arange(candidates_start, min(candidates_end, n_events))

        # Skip if already marked as aftershocks
        candidates = candidates[~is_aftershock[candidates]]

        if len(candidates) == 0:
            continue

        # Check distances using KD-tree
        mainshock_coord = cartesian_coords[i:i+1]
        candidate_coords = cartesian_coords[candidates]

        # Query distances
        dists = kdtree_cartesian.query(mainshock_coord, k=len(candidates)+1,
                                        distance_upper_bound=dist_window_m)[0][0][1:]

        # Mark aftershocks within distance window
        within_dist = dists <= dist_window_m
        valid_candidates = candidates[within_dist]
        valid_dists = dists[within_dist]

        for j, dist_m in zip(valid_candidates, valid_dists):
            if dist_m <= dist_window_m:
                is_aftershock[j] = True
                mainshock_mag[j] = mag_i

# Add to dataframe
df_work['is_aftershock'] = is_aftershock
df_work['mainshock_mag'] = mainshock_mag

n_aftershocks = is_aftershock.sum()
print(f"\n  ✓ Aftershocks detected: {n_aftershocks:,} ({n_aftershocks/len(df_work)*100:.1f}%)")

# ============================================================================
# FEATURE 2: FAULT PROXIMITY FEATURES (Vectorized)
# ============================================================================
print("\n[4/7] Fault proximity features (using seismicity density)...")

# Distance to k-th nearest neighbor (vectorized with KD-tree)
# Note: 20th neighbor removed - redundant with 5th and 10th
for k in [5, 10]:
    dists, _ = kdtree_cartesian.query(cartesian_coords, k=k+1)
    df_work[f'dist_to_{k}th_neighbor_km'] = dists[:, k] / 1000  # Convert m to km

# Seismicity density within radius (using KD-tree)
def count_within_radius_vectorized(kdtree, coords, radius_km):
    """Count events within radius for all points (vectorized)"""
    radius_m = radius_km * 1000
    counts = np.array([len(kdtree.query_ball_point(coord, radius_m)) - 1
                       for coord in tqdm(coords, desc=f"  Density ({radius_km}km)")])
    return counts

print("  Calculating seismicity density...")
density_100km = count_within_radius_vectorized(kdtree_cartesian, cartesian_coords, 100)
df_work['seismicity_density_100km'] = density_100km

print(f"  ✓ Median density: {np.median(density_100km):.0f} events/100km")

# ============================================================================
# FEATURE 3: COULOMB STRESS PROXY (Vectorized)
# ============================================================================
print("\n  Coulomb stress proxy (simplified)...")

# Vectorized calculation using sliding window
lookback = 20
mags = df_work['mag'].values

# For each event, sum of (mag^2 / distance) from previous lookback events
stress_proxy = np.zeros(len(df_work))

for i in tqdm(range(len(df_work)), desc="  Stress proxy"):
    start_idx = max(0, i - lookback)
    prev_indices = np.arange(start_idx, i)

    if len(prev_indices) == 0:
        continue

    # Get distances to previous events
    prev_coords = cartesian_coords[prev_indices]
    current_coord = cartesian_coords[i:i+1]

    dists = kdtree_cartesian.query(current_coord, k=len(prev_indices)+1)[0][0][1:]

    # Avoid division by zero
    dists = np.maximum(dists, 100)  # Minimum 100m

    # Calculate stress contribution
    prev_mags = mags[prev_indices]
    stress = np.sum((prev_mags ** 2) / dists) * 1000  # Scale factor
    stress_proxy[i] = stress

df_work['coulomb_stress_proxy'] = stress_proxy
print(f"  ✓ Range: {stress_proxy.min():.2f} - {stress_proxy.max():.2f}")

# ============================================================================
# FEATURE 4: REGIONAL FEATURES (Optimized)
# ============================================================================
print("\n[5/7] Regional seismicity features...")

# 4.1 Regional b-value (sampled for efficiency)
print("  Regional b-value (sampled)...")

sample_size = min(1000, len(df_work))
sample_indices = np.random.choice(len(df_work), sample_size, replace=False)
b_values = []

radius_km = 200
time_window_days = 365 * 5
time_window = pd.Timedelta(days=time_window_days)

for idx in tqdm(sample_indices, desc="  B-values"):
    lat = df_work.loc[idx, 'latitude']
    lon = df_work.loc[idx, 'longitude']
    time = df_work.loc[idx, 'time']

    # Find local events
    lat_mask = np.abs(df_work['latitude'] - lat) <= radius_km / 111
    lon_mask = np.abs(df_work['longitude'] - lon) <= radius_km / 111
    time_mask = (df_work['time'] >= time - time_window) & (df_work['time'] <= time)

    local_mags = df_work.loc[lat_mask & lon_mask & time_mask, 'mag'].values

    if len(local_mags) < 30:
        b_values.append(1.0)  # Default global b-value
        continue

    # Calculate b-value
    try:
        bins = np.arange(local_mags.min(), local_mags.max() + 0.1, 0.1)
        hist, _ = np.histogram(local_mags, bins=bins)
        valid = hist > 0

        if np.sum(valid) >= 3:
            bin_centers = (bins[:-1] + bins[1:]) / 2
            log_count = np.log10(hist[valid] + 1)
            slope, _, _, _, _ = stats.linregress(bin_centers[valid], log_count)
            b_values.append(-slope)
        else:
            b_values.append(1.0)
    except:
        b_values.append(1.0)

# Interpolate b-values for all events
from scipy.interpolate import interp1d
f_b = interp1d(sample_indices, b_values, kind='linear',
                bounds_error=False, fill_value=(np.mean(b_values), np.mean(b_values)))
df_work['regional_b_value'] = f_b(np.arange(len(df_work)))

print(f"  ✓ Mean b-value: {np.mean(b_values):.2f}")

# 4.2 Seismic gap (time since last M5+ in region)
print("  Seismic gap calculation...")

# Use KD-tree to find recent M5+ events
m5_indices = df_work[df_work['mag'] >= 5.0].index.tolist()
m5_coords = cartesian_coords[m5_indices]
m5_times = df_work.loc[m5_indices, 'time'].values

if len(m5_coords) > 0:
    kdtree_m5 = cKDTree(m5_coords)

    seismic_gaps = []
    for i in tqdm(range(len(df_work)), desc="  Seismic gaps"):
        dists, idxs = kdtree_m5.query(cartesian_coords[i:i+1], k=10)
        dists = dists[0] / 1000  # Convert to km
        idxs = idxs[0]

        # Find closest M5+ within 200km that occurred before this event
        current_time = df_work.loc[i, 'time']
        valid_mask = (dists <= 200) & (m5_times[idxs] < current_time)

        if np.any(valid_mask):
            valid_times = m5_times[idxs[valid_mask]]
            gap_days = (current_time - valid_times.max()).total_seconds() / 86400
            seismic_gaps.append(gap_days)
        else:
            seismic_gaps.append(365 * 10)  # 10 years default

    df_work['seismic_gap_days'] = seismic_gaps
else:
    df_work['seismic_gap_days'] = 365 * 10

print(f"  ✓ Median gap: {np.median(seismic_gaps):.0f} days")

# 4.3 Regional max magnitude (last 5 years)
print("  Regional max magnitude...")

regional_max_mags = []
time_window_5yr = pd.Timedelta(days=365 * 5)

for i in tqdm(range(len(df_work)), desc="  Max mag"):
    lat = df_work.loc[i, 'latitude']
    lon = df_work.loc[i, 'longitude']
    time = df_work.loc[i, 'time']

    lat_mask = np.abs(df_work['latitude'] - lat) <= 200 / 111
    lon_mask = np.abs(df_work['longitude'] - lon) <= 200 / 111
    time_mask = (df_work['time'] >= time - time_window_5yr) & (df_work['time'] < time)

    local_max = df_work.loc[lat_mask & lon_mask & time_mask, 'mag'].max()

    if pd.isna(local_max):
        regional_max_mags.append(df_work['mag'].min())
    else:
        regional_max_mags.append(local_max)

df_work['regional_max_mag_5yr'] = regional_max_mags
print(f"  ✓ Mean: {np.mean(regional_max_mags):.2f}")

# ============================================================================
# FEATURE 5: STRESS TENSOR FEATURES (NEW)
# ============================================================================
print("\n[6/7] Stress tensor features (NEW)...")

def estimate_stress_tensor_from_seismicity(df, coords, kdtree, lookback_events=50):
    """
    Ước lượng stress tensor từ dữ liệu động đất
    Dựa trên giả định: stress được giải phóng khi động đất xảy ra
    """
    n = len(df)
    stress_features = {}

    # Arrays for stress components
    # Note: stress_mean and stress_deviatoric removed due to multicollinearity
    sigma_1 = np.zeros(n)      # Principal stress lớn nhất
    sigma_3 = np.zeros(n)      # Principal stress nhỏ nhất
    tau_max = np.zeros(n)      # Maximum shear stress

    mags = df['mag'].values
    times = df['time'].values

    print("  Computing stress tensor components...")
    for i in tqdm(range(n), desc="  Stress tensor"):
        # Lấy lookback events gần đây
        start_idx = max(0, i - lookback_events)
        prev_indices = np.arange(start_idx, i)

        if len(prev_indices) < 5:
            # Default values nếu không đủ data
            sigma_1[i] = 50e6  # 50 MPa
            sigma_3[i] = 20e6  # 20 MPa
            tau_max[i] = (sigma_1[i] - sigma_3[i]) / 2
            continue

        # Ước lượng stress từ magnitudes
        # Stress drop ≈ 3e6 * 10^(1.5*M) Pascal
        prev_mags = mags[prev_indices]
        stress_drops = 3e6 * 10**(1.5 * prev_mags)

        # Mean stress drop
        mean_stress_drop = np.mean(stress_drops)

        # Principal stresses từ tectonic setting
        # Giả sử: sigma_1 > sigma_2 > sigma_3
        # và stress ratio R = (sigma_2 - sigma_1) / (sigma_3 - sigma_1)

        # Ước lượng sigma_1 từ stress drop lớn nhất
        sigma_1[i] = 100e6 + mean_stress_drop * 0.5  # Base 100 MPa + contribution
        sigma_3[i] = 30e6 + mean_stress_drop * 0.2   # Base 30 MPa + contribution

        # Maximum shear stress (Tresca criterion)
        tau_max[i] = (sigma_1[i] - sigma_3[i]) / 2

        # Note: mean_stress and deviatoric_stress removed
        # - mean_stress = (sigma_1 + sigma_3) / 2 (exact function, redundant)
        # - deviatoric_stress ≈ 1.732 × tau_max (r ≈ 0.99, highly correlated)

    stress_features['stress_sigma_1_mpa'] = sigma_1 / 1e6  # Convert to MPa
    stress_features['stress_sigma_3_mpa'] = sigma_3 / 1e6
    stress_features['stress_tau_max_mpa'] = tau_max / 1e6

    # Stress rate (MPa/year) - tốc độ tích tụ stress
    print("  Computing stress rate...")
    stress_rate = np.zeros(n)
    time_window_events = 20

    for i in tqdm(range(n), desc="  Stress rate"):
        if i < time_window_events:
            stress_rate[i] = 0
            continue

        # Tính stress accumulation rate
        prev_indices = np.arange(i - time_window_events, i)
        prev_mags = mags[prev_indices]

        # Tổng stress released
        total_stress = np.sum(3e6 * 10**(1.5 * prev_mags))

        # Time window
        time_diff = (times[i] - times[prev_indices[0]]).total_seconds() / (365 * 24 * 3600)

        if time_diff > 0:
            stress_rate[i] = (total_stress / 1e6) / time_diff  # MPa/year

    stress_features['stress_rate_mpa_per_year'] = stress_rate

    # Stress drop từ động đất lớn gần đây
    print("  Computing recent stress drop...")
    stress_drop_recent = np.zeros(n)

    for i in tqdm(range(n), desc="  Stress drop"):
        if i < 10:
            stress_drop_recent[i] = 0
            continue

        # Lấy event lớn nhất trong 10 events trước
        prev_indices = np.arange(max(0, i - 10), i)
        prev_mags = mags[prev_indices]

        if len(prev_mags) > 0:
            max_mag = np.max(prev_mags)
            stress_drop_recent[i] = 3 * 10**(1.5 * max_mag)  # MPa

    stress_features['stress_drop_recent_mpa'] = stress_drop_recent

    return stress_features

# Tính stress tensor features
stress_features = estimate_stress_tensor_from_seismicity(
    df_work, cartesian_coords, kdtree_cartesian
)

# Thêm vào dataframe
for key, values in stress_features.items():
    df_work[key] = values

print(f"  ✓ Stress tensor features added: {len(stress_features)} features")

# ============================================================================
# FEATURE 6: FAULT GEOMETRY FEATURES (NEW)
# ============================================================================
print("\n[7/7] Fault geometry features (NEW)...")

def estimate_fault_geometry_from_seismicity(df, coords, kdtree, n_clusters=50):
    """
    Ước lượng fault geometry từ phân bố động đất
    Sử dụng clustering để tìm các fault lines
    """
    n = len(df)
    geometry_features = {}

    # 1. Fault depth (continuous feature - simplified from 3 one-hot features)
    print("  Computing fault depth...")
    depths = df['depth'].values
    geometry_features['fault_depth_km'] = depths

    # 2. Local strike direction (hướng đứt gãy)
    print("  Computing local strike direction...")
    local_strike = np.zeros(n)
    local_dip = np.zeros(n)

    for i in tqdm(range(n), desc="  Strike/Dip"):
        # Tính local strike từ PCA của nearby events
        radius_km = 50
        radius_m = radius_km * 1000

        # Find nearby events
        nearby_indices = kdtree.query_ball_point(coords[i], radius_m)

        if len(nearby_indices) < 10:
            local_strike[i] = 0
            local_dip[i] = 90  # Default vertical
            continue

        # Get coordinates of nearby events
        nearby_coords = coords[nearby_indices]

        # PCA để tìm hướng chính của fault
        # Center the coordinates
        centered = nearby_coords - nearby_coords.mean(axis=0)

        # Covariance matrix
        cov = np.cov(centered.T)

        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # Sort by eigenvalue
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]

        # First eigenvector = strike direction
        strike_vector = eigenvectors[:, 0]

        # Convert to azimuth (degrees from North)
        x, y, z = strike_vector
        strike_rad = np.arctan2(y, x)
        strike_deg = np.degrees(strike_rad)

        # Normalize to 0-360
        if strike_deg < 0:
            strike_deg += 360

        local_strike[i] = strike_deg

        # Dip: estimated from depth variation
        nearby_depths = depths[nearby_indices]
        if len(nearby_depths) > 0:
            depth_std = np.std(nearby_depths)
            # High depth variation = steep dip
            local_dip[i] = 90 - np.clip(depth_std, 0, 60)
        else:
            local_dip[i] = 90

    geometry_features['fault_strike_deg'] = local_strike
    geometry_features['fault_dip_deg'] = local_dip

    # Note: fault_curvature removed - unreliable estimator from seismicity data

    # 3. Estimated fault length (khoảng cách tới event xa nhất trong cluster)
    print("  Computing estimated fault length...")
    fault_length = np.zeros(n)

    for i in tqdm(range(n), desc="  Fault length"):
        radius_km = 100
        radius_m = radius_km * 1000

        nearby_indices = kdtree.query_ball_point(coords[i], radius_m)

        if len(nearby_indices) < 2:
            fault_length[i] = 10  # Default 10 km
            continue

        # Maximum distance between any two nearby events
        nearby_coords = coords[nearby_indices]

        # Find max distance
        max_dist = 0
        for j in range(len(nearby_coords)):
            for k in range(j+1, len(nearby_coords)):
                dist = np.linalg.norm(nearby_coords[j] - nearby_coords[k])
                if dist > max_dist:
                    max_dist = dist

        fault_length[i] = max_dist / 1000  # Convert to km

    geometry_features['fault_length_km'] = fault_length

    # Note: fault_complexity and dist_to_fault_intersection removed
    # - fault_complexity: subjective, not well-defined
    # - dist_to_fault_intersection: unreliable estimator from seismicity

    return geometry_features

# Tính fault geometry features
geometry_features = estimate_fault_geometry_from_seismicity(
    df_work, cartesian_coords, kdtree_cartesian
)

# Thêm vào dataframe
for key, values in geometry_features.items():
    df_work[key] = values

print(f"  ✓ Fault geometry features added: {len(geometry_features)} features")

# ============================================================================
# SUMMARY AND SAVE
# ============================================================================
print("\n" + "="*70)
print(" SUMMARY ")
print("="*70)

# Tất cả features mới (OPTIMIZED - removed redundant features)
new_features = [
    # Aftershock features (mainshock_id removed - just an identifier)
    'is_aftershock', 'mainshock_mag',
    # Fault proximity (20th neighbor removed - redundant with 5th and 10th)
    'dist_to_5th_neighbor_km', 'dist_to_10th_neighbor_km',
    'seismicity_density_100km',
    # Stress features
    'coulomb_stress_proxy',
    # Regional features
    'regional_b_value', 'seismic_gap_days', 'regional_max_mag_5yr',
]

# Thêm stress tensor features (stress_mean and stress_deviatoric removed - multicollinearity)
stress_feature_names = [
    'stress_sigma_1_mpa',
    'stress_sigma_3_mpa',
    'stress_tau_max_mpa',
    'stress_rate_mpa_per_year',
    'stress_drop_recent_mpa'
]

# Thêm fault geometry features (OPTIMIZED - removed unreliable/subjective features)
geometry_feature_names = [
    'fault_depth_km',          # Continuous feature (replaced 3 one-hot features)
    'fault_strike_deg',
    'fault_dip_deg',
    'fault_length_km'
]

all_new_features = new_features + stress_feature_names + geometry_feature_names

print(f"\n✓ Total features added: {len(all_new_features)} (OPTIMIZED)")
print(f"  - Base features: {len(new_features)}")
print(f"  - Stress tensor features: {len(stress_feature_names)} (reduced from 7)")
print(f"  - Fault geometry features: {len(geometry_feature_names)} (reduced from 9)")

# Hiển thị thống kê
print(f"\nStress Tensor Features:")
for feat in stress_feature_names:
    values = df_work[feat].values
    print(f"  {feat}: mean={values.mean():.2f}, std={values.std():.2f}")

print(f"\nFault Geometry Features:")
for feat in geometry_feature_names:
    values = df_work[feat].values
    print(f"  {feat}: mean={values.mean():.2f}, std={values.std():.2f}")

# Save
output_file = '/home/haind/Desktop/earthquake-sequence-mining/haind/features_advanced.csv'
df_work.to_csv(output_file, index=False)

print(f"\n✓ SAVED: {output_file}")
print(f"  Rows: {len(df_work):,}")
print(f"  Columns: {len(df_work.columns)}")

print("\n" + "="*70)
print(" COMPLETE! ")
print("="*70)
print("\nOPTIMIZED FEATURES (Removed redundant/unreliable features):")
print("  [Removed]")
print("    - mainshock_id: identifier only, no predictive value")
print("    - dist_to_20th_neighbor_km: redundant with 5th and 10th")
print("    - stress_mean_mpa: exact function of sigma_1 and sigma_3")
print("    - stress_deviatoric_mpa: r≈0.99 with tau_max, highly correlated")
print("    - fault_is_shallow/intermediate/deep: replaced with fault_depth_km")
print("    - fault_curvature, fault_complexity, dist_to_fault_intersection: unreliable")
print("\n  [Stress Tensor - 5 features]")
print("    - stress_sigma_1_mpa: Maximum principal stress")
print("    - stress_sigma_3_mpa: Minimum principal stress")
print("    - stress_tau_max_mpa: Maximum shear stress")
print("    - stress_rate_mpa_per_year: Stress accumulation rate")
print("    - stress_drop_recent_mpa: Recent stress drop")
print("\n  [Fault Geometry - 4 features]")
print("    - fault_depth_km: Continuous depth feature (replaced 3 one-hot)")
print("    - fault_strike_deg: Strike direction (0-360°)")
print("    - fault_dip_deg: Dip angle (0-90°)")
print("    - fault_length_km: Estimated fault length")
print("="*70)
