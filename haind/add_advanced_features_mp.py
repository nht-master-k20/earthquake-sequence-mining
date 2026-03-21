"""
Add Advanced Features for Earthquake Forecasting - MULTIPROCESSING VERSION
Features: Fault Line, Aftershock Detection, Coulomb Stress, Regional Features
NEW: Stress Tensor Features, Fault Geometry Features
NEW: Checkpoint system + MULTIPROCESSING for speed

IMPORTANT: Process ALL magnitudes (not just M >= 3.0) to preserve foreshocks

Optimizations:
- Vectorized operations with NumPy
- Efficient spatial queries with KD-tree
- Progress bars for tracking
- Optimized Gardner-Knopoff algorithm
- CHECKPOINT: Save progress after each step
- MULTIPROCESSING: Parallel processing for CPU-intensive tasks

Author: haind
Project: Earthquake Sequence Mining
Updated: 2025-03-21 (Added Multiprocessing)
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
from tqdm import tqdm
import warnings
import pickle
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count, shared_memory
from functools import partial
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
# Số CPU cores sử dụng (để -1 để dùng tất cả, hoặc số cụ thể)
N_CORES = max(1, cpu_count() - 1)  # Giữ 1 core cho system

# Chunk size cho multiprocessing
CHUNK_SIZE = 10000

print(f"Using {N_CORES} CPU cores for multiprocessing")

# ============================================================================
# CHECKPOINT CONFIGURATION
# ============================================================================
CHECKPOINT_DIR = Path('/home/haind/Desktop/earthquake-sequence-mining/haind/checkpoints')
CHECKPOINT_DIR.mkdir(exist_ok=True)

CHECKPOINT_FILE = CHECKPOINT_DIR / 'add_features_mp_checkpoint.pkl'

def save_checkpoint(step, data):
    """Lưu trạng thái sau mỗi bước"""
    checkpoint = {
        'step': step,
        'df_work': data['df_work'],
        'cartesian_coords': data.get('cartesian_coords'),
        'kdtree_cartesian': data.get('kdtree_cartesian'),
        'completed_steps': data.get('completed_steps', [])
    }
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"  ✓ Checkpoint saved at step {step}")

def load_checkpoint():
    """Tải checkpoint nếu có"""
    if not CHECKPOINT_FILE.exists():
        return None

    try:
        with open(CHECKPOINT_FILE, 'rb') as f:
            checkpoint = pickle.load(f)
        print(f"  ✓ Found checkpoint from step {checkpoint['step']}")
        print(f"    Completed steps: {checkpoint.get('completed_steps', [])}")
        return checkpoint
    except Exception as e:
        print(f"  ✗ Error loading checkpoint: {e}")
        return None

def clear_checkpoint():
    """Xóa checkpoint khi hoàn thành"""
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        print("  ✓ Checkpoint cleared")

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
    """Process một chunk cho b-value calculation"""
    indices, df_work, cartesian_coords, kdtree_cartesian, mags = args

    results = []
    for idx in indices:
        current_time = df_work.loc[idx, 'time']
        current_coord = cartesian_coords[idx]

        # Spatial window
        radius_km = 100
        radius_m = radius_km * 1000
        nearby_indices = kdtree_cartesian.query_ball_point(current_coord, radius_m)

        # Time window: 1 year before
        time_window = pd.Timedelta(days=365)
        time_mask = (df_work.loc[nearby_indices, 'time'] >= current_time - time_window) & \
                    (df_work.loc[nearby_indices, 'time'] <= current_time)
        nearby_indices = [nearby_indices[i] for i, mask in enumerate(time_mask) if mask]

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
            gap_days = time_delta.astype('timedelta64[s]').astype(float) / 86400
            results.append(gap_days)
        else:
            results.append(365 * 10)

    return chunk_start, results

def process_regional_max_mag_chunk(args):
    """Process một chunk cho regional max magnitude"""
    chunk_start, chunk_end, df_work, cartesian_coords, mags, kdtree_cartesian, radius_m = args

    results = []
    for i in range(chunk_start, chunk_end):
        current_time = df_work.loc[i, 'time']
        time_window = pd.Timedelta(days=365*5)

        nearby_indices = kdtree_cartesian.query_ball_point(cartesian_coords[i], radius_m)
        time_mask = (df_work.loc[nearby_indices, 'time'] >= current_time - time_window) & \
                    (df_work.loc[nearby_indices, 'time'] <= current_time)
        nearby_indices = [nearby_indices[i] for i, mask in enumerate(time_mask) if mask]

        if len(nearby_indices) > 0:
            results.append(mags[nearby_indices].max())
        else:
            results.append(df_work.loc[i, 'mag'])

    return chunk_start, results

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
    """Process một chunk cho fault geometry calculation"""
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

        # Fault length
        if len(nearby_coords) >= 2:
            max_dist = 0
            for j in range(len(nearby_coords)):
                for k in range(j+1, len(nearby_coords)):
                    dist = np.linalg.norm(nearby_coords[j] - nearby_coords[k])
                    if dist > max_dist:
                        max_dist = dist
            fault_length[idx] = max_dist / 1000
        else:
            fault_length[idx] = 10

    return chunk_start, local_strike, local_dip, fault_length


# ============================================================================
# MAIN EXECUTION
# ============================================================================

print("="*70)
print(" ADVANCED FEATURES - MULTIPROCESSING VERSION ")
print("="*70)

# ============================================================================
# CHECKPOINT: LOAD OR START FRESH
# ============================================================================
checkpoint = load_checkpoint()

if checkpoint is not None:
    df_work = checkpoint['df_work']
    cartesian_coords = checkpoint.get('cartesian_coords')
    kdtree_cartesian = checkpoint.get('kdtree_cartesian')
    completed_steps = checkpoint.get('completed_steps', [])
    print(f"\n  Resuming from checkpoint...")
    print(f"  Data loaded: {len(df_work):,} events")
else:
    df_work = None
    cartesian_coords = None
    kdtree_cartesian = None
    completed_steps = []

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
if 1 not in completed_steps:
    print("\n[1/8] Loading data...")
    df = pd.read_csv('/home/haind/Desktop/earthquake-sequence-mining/dongdat.csv')
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    df_work = df.copy().reset_index(drop=True)
    print(f"  Working with {len(df_work):,} events")
    print(f"  Magnitude range: {df_work['mag'].min():.1f} - {df_work['mag'].max():.1f}")

    completed_steps.append(1)
    save_checkpoint(1, {'df_work': df_work, 'completed_steps': completed_steps})
else:
    print("\n[1/8] ✓ Data already loaded (from checkpoint)")

# ============================================================================
# STEP 2: BUILD SPATIAL INDEX
# ============================================================================
if 2 not in completed_steps:
    print("\n[2/8] Building spatial index (KD-tree)...")

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

    completed_steps.append(2)
    save_checkpoint(2, {
        'df_work': df_work,
        'cartesian_coords': cartesian_coords,
        'kdtree_cartesian': kdtree_cartesian,
        'completed_steps': completed_steps
    })
else:
    print("\n[2/8] ✓ KD-tree already built (from checkpoint)")

# ============================================================================
# FEATURE 1: AFTERSHOCK DETECTION (MULTIPROCESSING)
# ============================================================================
if 'aftershock' not in completed_steps:
    print("\n[3/8] Aftershock detection (MULTIPROCESSING)...")

    times = df_work['time'].values
    mags = df_work['mag'].values
    time_numeric = times.astype(np.int64) / 1e9

    n_events = len(df_work)
    chunk_size = CHUNK_SIZE

    # Prepare arguments for multiprocessing
    args_list = []
    for chunk_start in range(0, n_events, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_events)
        args_list.append((chunk_start, chunk_end, times, mags, time_numeric, cartesian_coords))

    # Process with multiprocessing
    is_aftershock = np.zeros(n_events, dtype=bool)
    mainshock_mag = mags.copy()

    print(f"  Processing with {N_CORES} cores...")
    with Pool(N_CORES) as pool:
        results = list(tqdm(
            pool.imap(process_aftershock_chunk, args_list),
            total=len(args_list),
            desc="  Declustering"
        ))

    # Combine results
    for chunk_start, chunk_end, aftershocks, mainshocks in results:
        is_aftershock[chunk_start:chunk_end] = aftershocks
        mainshock_mag[chunk_start:chunk_end] = mainshocks

    df_work['is_aftershock'] = is_aftershock
    df_work['mainshock_mag'] = mainshock_mag

    print(f"  ✓ Aftershock detection complete")
    print(f"    Aftershocks: {is_aftershock.sum():,} ({is_aftershock.sum()/len(df_work)*100:.1f}%)")

    completed_steps.append('aftershock')
    save_checkpoint('aftershock', {
        'df_work': df_work,
        'cartesian_coords': cartesian_coords,
        'kdtree_cartesian': kdtree_cartesian,
        'completed_steps': completed_steps
    })
else:
    print("\n[3/8] ✓ Aftershock detection already done (from checkpoint)")

# ============================================================================
# FEATURE 2: DISTANCE TO NEIGHBORS
# ============================================================================
if 'distance_neighbors' not in completed_steps:
    print("\n[4/8] Computing distance to neighbors...")

    n_events = len(df_work)

    # Distance to 5th and 10th neighbor
    print("  Computing distance to 5th and 10th neighbor...")
    dists_5th, idxs_5th = kdtree_cartesian.query(cartesian_coords, k=6)
    dist_to_5th_neighbor_km = dists_5th[:, 5] / 1000

    dists_10th, idxs_10th = kdtree_cartesian.query(cartesian_coords, k=11)
    dist_to_10th_neighbor_km = dists_10th[:, 10] / 1000

    # Seismicity density (MULTIPROCESSING)
    print("  Computing seismicity density (MULTIPROCESSING)...")
    radius_km = 100
    radius_m = radius_km * 1000

    args_list = []
    chunk_size = CHUNK_SIZE
    for chunk_start in range(0, n_events, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_events)
        args_list.append((chunk_start, chunk_end, cartesian_coords, kdtree_cartesian, radius_m))

    seismicity_density = np.zeros(n_events)

    with Pool(N_CORES) as pool:
        results = list(tqdm(
            pool.imap(process_density_chunk, args_list),
            total=len(args_list),
            desc="  Density"
        ))

    for chunk_start, counts in results:
        for i, count in enumerate(counts):
            seismicity_density[chunk_start + i] = count

    seismicity_density_100km = seismicity_density / (np.pi * radius_km**2)

    df_work['dist_to_5th_neighbor_km'] = dist_to_5th_neighbor_km
    df_work['dist_to_10th_neighbor_km'] = dist_to_10th_neighbor_km
    df_work['seismicity_density_100km'] = seismicity_density_100km

    print(f"  ✓ Distance features added")

    completed_steps.append('distance_neighbors')
    save_checkpoint('distance_neighbors', {
        'df_work': df_work,
        'cartesian_coords': cartesian_coords,
        'kdtree_cartesian': kdtree_cartesian,
        'completed_steps': completed_steps
    })
else:
    print("\n[4/8] ✓ Distance features already computed (from checkpoint)")

# ============================================================================
# FEATURE 3: COULOMB STRESS (MULTIPROCESSING)
# ============================================================================
if 'coulomb_stress' not in completed_steps:
    print("\n[5/8] Computing Coulomb stress proxy (MULTIPROCESSING)...")

    mags = df_work['mag'].values
    radius_km = 50
    radius_m = radius_km * 1000
    lookback_events = 20

    n_events = len(df_work)
    args_list = []
    chunk_size = CHUNK_SIZE

    for chunk_start in range(0, n_events, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_events)
        args_list.append((chunk_start, chunk_end, mags, cartesian_coords, kdtree_cartesian, radius_m, lookback_events))

    coulomb_stress_proxy = np.zeros(n_events)

    with Pool(N_CORES) as pool:
        results = list(tqdm(
            pool.imap(process_coulomb_stress_chunk, args_list),
            total=len(args_list),
            desc="  Coulomb stress"
        ))

    for chunk_start, stresses in results:
        for i, stress in enumerate(stresses):
            coulomb_stress_proxy[chunk_start + i] = stress

    df_work['coulomb_stress_proxy'] = coulomb_stress_proxy

    print(f"  ✓ Coulomb stress proxy added")

    completed_steps.append('coulomb_stress')
    save_checkpoint('coulomb_stress', {
        'df_work': df_work,
        'cartesian_coords': cartesian_coords,
        'kdtree_cartesian': kdtree_cartesian,
        'completed_steps': completed_steps
    })
else:
    print("\n[5/8] ✓ Coulomb stress already computed (from checkpoint)")

# ============================================================================
# FEATURE 4: REGIONAL FEATURES (MULTIPROCESSING)
# ============================================================================
if 'regional' not in completed_steps:
    print("\n[6/8] Computing regional features (MULTIPROCESSING)...")

    mags = df_work['mag'].values

    # 4.1 Regional b-value (MULTIPROCESSING)
    print("  Computing regional b-values (MULTIPROCESSING)...")
    sample_size = min(1000, len(df_work))
    sample_indices = np.linspace(0, len(df_work)-1, sample_size, dtype=int)

    # Split samples for multiprocessing
    samples_per_core = len(sample_indices) // N_CORES
    args_list = []

    for core_id in range(N_CORES):
        start_idx = core_id * samples_per_core
        if core_id == N_CORES - 1:
            end_idx = len(sample_indices)
        else:
            end_idx = start_idx + samples_per_core

        if start_idx < len(sample_indices):
            args_list.append((sample_indices[start_idx:end_idx], df_work, cartesian_coords, kdtree_cartesian, mags))

    b_values = []
    with Pool(N_CORES) as pool:
        results = pool.map(process_b_value_chunk, args_list)
        for res in results:
            b_values.extend(res)

    # Interpolate
    f_b = interp1d(sample_indices, b_values, kind='linear',
                    bounds_error=False, fill_value=(np.mean(b_values), np.mean(b_values)))
    df_work['regional_b_value'] = f_b(np.arange(len(df_work)))
    print(f"  ✓ Mean b-value: {np.mean(b_values):.2f}")

    # 4.2 Seismic gap (MULTIPROCESSING)
    print("  Seismic gap calculation (MULTIPROCESSING)...")
    m5_indices = df_work[df_work['mag'] >= 5.0].index.tolist()
    m5_coords = cartesian_coords[m5_indices]
    m5_times = df_work.loc[m5_indices, 'time'].values

    if len(m5_coords) > 0:
        kdtree_m5 = cKDTree(m5_coords)

        n_events = len(df_work)
        args_list = []
        chunk_size = CHUNK_SIZE

        for chunk_start in range(0, n_events, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_events)
            args_list.append((chunk_start, chunk_end, df_work, cartesian_coords, m5_coords, m5_times, kdtree_m5))

        seismic_gaps = np.zeros(n_events)

        with Pool(N_CORES) as pool:
            results = list(tqdm(
                pool.imap(process_seismic_gap_chunk, args_list),
                total=len(args_list),
                desc="  Seismic gaps"
            ))

        for chunk_start, gaps in results:
            for i, gap in enumerate(gaps):
                seismic_gaps[chunk_start + i] = gap

        df_work['seismic_gap_days'] = seismic_gaps
        print(f"  ✓ Median gap: {np.median(seismic_gaps):.0f} days")
    else:
        df_work['seismic_gap_days'] = 365 * 10

    # 4.3 Regional max magnitude (MULTIPROCESSING)
    print("  Computing regional max magnitude (MULTIPROCESSING)...")
    radius_km = 200
    radius_m = radius_km * 1000

    n_events = len(df_work)
    args_list = []
    chunk_size = CHUNK_SIZE

    for chunk_start in range(0, n_events, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_events)
        args_list.append((chunk_start, chunk_end, df_work, cartesian_coords, mags, kdtree_cartesian, radius_m))

    regional_max_mag = np.zeros(n_events)

    with Pool(N_CORES) as pool:
        results = list(tqdm(
            pool.imap(process_regional_max_mag_chunk, args_list),
            total=len(args_list),
            desc="  Regional max mag"
        ))

    for chunk_start, mags in results:
        for i, mag in enumerate(mags):
            regional_max_mag[chunk_start + i] = mag

    df_work['regional_max_mag_5yr'] = regional_max_mag

    print(f"  ✓ Regional features added")

    completed_steps.append('regional')
    save_checkpoint('regional', {
        'df_work': df_work,
        'cartesian_coords': cartesian_coords,
        'kdtree_cartesian': kdtree_cartesian,
        'completed_steps': completed_steps
    })
else:
    print("\n[6/8] ✓ Regional features already computed (from checkpoint)")

# ============================================================================
# FEATURE 5: STRESS TENSOR (MULTIPROCESSING)
# ============================================================================
if 'stress_tensor' not in completed_steps:
    print("\n[7/8] Stress tensor features (MULTIPROCESSING)...")

    mags = df_work['mag'].values
    times = df_work['time'].values
    lookback_events = 50
    time_window_events = 20
    n_events = len(df_work)

    # Stress tensor components
    print("  Computing stress tensor components (MULTIPROCESSING)...")
    args_list = []
    chunk_size = CHUNK_SIZE

    for chunk_start in range(0, n_events, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_events)
        args_list.append((chunk_start, chunk_end, mags, times, lookback_events))

    sigma_1 = np.zeros(n_events)
    sigma_3 = np.zeros(n_events)
    tau_max = np.zeros(n_events)

    with Pool(N_CORES) as pool:
        results = list(tqdm(
            pool.imap(process_stress_tensor_chunk, args_list),
            total=len(args_list),
            desc="  Stress tensor"
        ))

    for chunk_start, s1, s3, tm in results:
        for i in range(len(s1)):
            sigma_1[chunk_start + i] = s1[i]
            sigma_3[chunk_start + i] = s3[i]
            tau_max[chunk_start + i] = tm[i]

    df_work['stress_sigma_1_mpa'] = sigma_1 / 1e6
    df_work['stress_sigma_3_mpa'] = sigma_3 / 1e6
    df_work['stress_tau_max_mpa'] = tau_max / 1e6

    # Stress rate (MULTIPROCESSING)
    print("  Computing stress rate (MULTIPROCESSING)...")
    args_list = []

    for chunk_start in range(0, n_events, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_events)
        args_list.append((chunk_start, chunk_end, mags, times, time_window_events))

    stress_rate = np.zeros(n_events)

    with Pool(N_CORES) as pool:
        results = list(tqdm(
            pool.imap(process_stress_rate_chunk, args_list),
            total=len(args_list),
            desc="  Stress rate"
        ))

    for chunk_start, rates in results:
        for i, rate in enumerate(rates):
            stress_rate[chunk_start + i] = rate

    df_work['stress_rate_mpa_per_year'] = stress_rate

    # Stress drop
    print("  Computing recent stress drop...")
    stress_drop_recent = np.zeros(n_events)

    for i in tqdm(range(n_events), desc="  Stress drop"):
        if i < 10:
            continue
        prev_indices = np.arange(max(0, i - 10), i)
        prev_mags = mags[prev_indices]
        if len(prev_mags) > 0:
            stress_drop_recent[i] = 3 * 10**(1.5 * np.max(prev_mags))

    df_work['stress_drop_recent_mpa'] = stress_drop_recent

    print(f"  ✓ Stress tensor features added: 5 features")

    completed_steps.append('stress_tensor')
    save_checkpoint('stress_tensor', {
        'df_work': df_work,
        'cartesian_coords': cartesian_coords,
        'kdtree_cartesian': kdtree_cartesian,
        'completed_steps': completed_steps
    })
else:
    print("\n[7/8] ✓ Stress tensor already computed (from checkpoint)")

# ============================================================================
# FEATURE 6: FAULT GEOMETRY (MULTIPROCESSING)
# ============================================================================
if 'fault_geometry' not in completed_steps:
    print("\n[8/8] Fault geometry features (MULTIPROCESSING)...")

    depths = df_work['depth'].values
    radius_km = 50
    radius_m = radius_km * 1000
    n_events = len(df_work)

    # Use chunks
    args_list = []
    chunk_size = CHUNK_SIZE

    for chunk_start in range(0, n_events, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_events)
        args_list.append((chunk_start, chunk_end, df_work, cartesian_coords, kdtree_cartesian, depths, radius_m))

    local_strike = np.zeros(n_events)
    local_dip = np.zeros(n_events)
    fault_length = np.zeros(n_events)

    print("  Computing fault geometry (MULTIPROCESSING)...")
    with Pool(N_CORES) as pool:
        results = list(tqdm(
            pool.imap(process_fault_geometry_chunk, args_list),
            total=len(args_list),
            desc="  Fault geometry"
        ))

    for chunk_start, strike, dip, length in results:
        for i in range(len(strike)):
            local_strike[chunk_start + i] = strike[i]
            local_dip[chunk_start + i] = dip[i]
            fault_length[chunk_start + i] = length[i]

    df_work['fault_depth_km'] = depths
    df_work['fault_strike_deg'] = local_strike
    df_work['fault_dip_deg'] = local_dip
    df_work['fault_length_km'] = fault_length

    print(f"  ✓ Fault geometry features added: 4 features")

    completed_steps.append('fault_geometry')
    save_checkpoint('fault_geometry', {
        'df_work': df_work,
        'cartesian_coords': cartesian_coords,
        'kdtree_cartesian': kdtree_cartesian,
        'completed_steps': completed_steps
    })
else:
    print("\n[8/8] ✓ Fault geometry already computed (from checkpoint)")

# ============================================================================
# SUMMARY AND SAVE
# ============================================================================
print("\n" + "="*70)
print(" SUMMARY ")
print("="*70)

all_new_features = [
    'is_aftershock', 'mainshock_mag',
    'dist_to_5th_neighbor_km', 'dist_to_10th_neighbor_km',
    'seismicity_density_100km', 'coulomb_stress_proxy',
    'regional_b_value', 'seismic_gap_days', 'regional_max_mag_5yr',
    'stress_sigma_1_mpa', 'stress_sigma_3_mpa', 'stress_tau_max_mpa',
    'stress_rate_mpa_per_year', 'stress_drop_recent_mpa',
    'fault_depth_km', 'fault_strike_deg', 'fault_dip_deg', 'fault_length_km'
]

print(f"\nFeatures added ({len(all_new_features)} total):")
for feat in all_new_features:
    if feat in df_work.columns:
        print(f"  ✓ {feat}")

# Save to CSV
output_file = '/home/haind/Desktop/earthquake-sequence-mining/haind/features_advanced.csv'
df_work.to_csv(output_file, index=False)

print(f"\n✓ SAVED: {output_file}")
print(f"  Rows: {len(df_work):,}")
print(f"  Columns: {len(df_work.columns)}")

# Clear checkpoint
clear_checkpoint()

print("\n" + "="*70)
print(" COMPLETE! ")
print("="*70)
print("\nMultiprocessing version completed successfully!")
print(f"  Used {N_CORES} CPU cores")
print(f"  Total advanced features: {len(all_new_features)}")
print("="*70)
