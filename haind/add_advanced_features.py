"""
Add Advanced Features for Earthquake Forecasting - OPTIMIZED VERSION
Features: Fault Line, Aftershock Detection, Coulomb Stress, Regional Features

Optimizations:
- Vectorized operations with NumPy
- Efficient spatial queries with KD-tree
- Progress bars for tracking
- Optimized Gardner-Knopoff algorithm

Author: haind
Project: Earthquake Sequence Mining
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
print("\n[1/5] Loading data...")
df = pd.read_csv('/home/haind/Desktop/earthquake-sequence-mining/dongdat.csv')
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)

# Filter for M >= 3.0
df_work = df[df['mag'] >= 3.0].copy().reset_index(drop=True)
print(f"  Working with {len(df_work):,} events (M >= 3.0)")

# ============================================================================
# PRE-COMPUTE SPATIAL DATA WITH KD-TREE
# ============================================================================
print("\n[2/5] Building spatial index (KD-tree)...")

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
print("\n[3/5] Aftershock detection (Gardner-Knopoff, memory-optimized)...")

# Get arrays
times = df_work['time'].values
mags = df_work['mag'].values

# Initialize arrays
is_aftershock = np.zeros(len(df_work), dtype=bool)
mainshock_id = np.full(len(df_work), -1, dtype=np.int64)
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
                mainshock_id[j] = i
                mainshock_mag[j] = mag_i

# Add to dataframe
df_work['is_aftershock'] = is_aftershock
df_work['mainshock_id'] = mainshock_id
df_work['mainshock_mag'] = mainshock_mag

n_aftershocks = is_aftershock.sum()
print(f"\n  ✓ Aftershocks detected: {n_aftershocks:,} ({n_aftershocks/len(df_work)*100:.1f}%)")

# ============================================================================
# FEATURE 2: FAULT PROXIMITY FEATURES (Vectorized)
# ============================================================================
print("\n[4/5] Fault proximity features (using seismicity density)...")

# Distance to k-th nearest neighbor (vectorized with KD-tree)
for k in [5, 10, 20]:
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
print("\n[5/5] Regional seismicity features...")

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
# SUMMARY AND SAVE
# ============================================================================
print("\n" + "="*70)
print(" SUMMARY ")
print("="*70)

new_features = [
    'is_aftershock', 'mainshock_id', 'mainshock_mag',
    'dist_to_5th_neighbor_km', 'dist_to_10th_neighbor_km', 'dist_to_20th_neighbor_km',
    'seismicity_density_100km',
    'coulomb_stress_proxy',
    'regional_b_value', 'seismic_gap_days', 'regional_max_mag_5yr'
]

print(f"\n✓ {len(new_features)} new features added")

# Save
output_file = '/home/haind/Desktop/earthquake-sequence-mining/haind/features_advanced.csv'
df_work.to_csv(output_file, index=False)

print(f"\n✓ SAVED: {output_file}")
print(f"  Rows: {len(df_work):,}")
print(f"  Columns: {len(df_work.columns)}")

print("\n" + "="*70)
print(" COMPLETE! ")
print("="*70)
