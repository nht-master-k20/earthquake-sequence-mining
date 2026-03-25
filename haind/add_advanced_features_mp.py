"""
Add Advanced Features for Earthquake Forecasting - LSTM OPTIMIZED VERSION
Task: Predict time-to-next and magnitude-of-next earthquake using LSTM

Features (29 total):
- ORIGINAL (10): time, latitude, longitude, depth, mag, sig, mmi, cdi, felt, region_code
- CORE (5): Aftershock, Density, Coulomb Stress, B-value
- SEQUENCE (6): Spatio-temporal clustering features
- LSTM CRITICAL (5): Temporal intervals only
- TARGETS (3): time_to_next, next_mag, next_mag_binary

Data Sorting:
- PRIMARY: region_code (grid-based, ~50km)
- SECONDARY: time (chronological within each region)

Region Code Calculation:
- Format: R{lat_int}_{lon_int}
- Grid size: 0.5° ≈ 55km
- Purpose: Regional earthquake pattern analysis

Optimizations:
- Vectorized operations with NumPy
- Numba JIT compilation for loops
- Efficient spatial queries with KD-tree
- Binary search for time-based queries
- Checkpoint system enabled (resume from failures)
- CSV version system (7 versions)

Author: haind
Project: Earthquake Sequence Mining
Updated: 2025-03-25
Version: 4.0 (Optimized with Numba + vectorization)
"""

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
from tqdm import tqdm
import time
import warnings
import os
import pickle
import json
from pathlib import Path
from datetime import datetime
from numba import jit, prange
import bisect

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = BASE_DIR / 'checkpoints'
CHECKPOINT_DIR.mkdir(exist_ok=True)

CHECKPOINT_META_FILE = CHECKPOINT_DIR / 'checkpoint_meta.json'
CHECKPOINT_DATA_FILE = CHECKPOINT_DIR / 'checkpoint_data.pkl'

# ============================================================================
# TIME RANGE CONFIGURATION
# ============================================================================
# Data filtering by year range
START_YEAR = 2000
END_YEAR = 2026  # Exclusive (will process 2000-2025)

# ============================================================================
# CHECKPOINT HELPER FUNCTIONS
# ============================================================================
def save_checkpoint(df, completed_steps, current_step_name):
    checkpoint_data = {
        'df': df,
        'completed_steps': completed_steps,
        'last_step': current_step_name,
        'timestamp': datetime.now().isoformat()
    }

    with open(CHECKPOINT_DATA_FILE, 'wb') as f:
        pickle.dump(checkpoint_data, f)

    meta_data = {
        'completed_steps': completed_steps,
        'last_step': current_step_name,
        'timestamp': datetime.now().isoformat(),
        'n_rows': len(df),
        'n_cols': len(df.columns)
    }

    with open(CHECKPOINT_META_FILE, 'w') as f:
        json.dump(meta_data, f, indent=2)

    print(f"  ✓ Checkpoint: {current_step_name} ({len(df):,} rows, {len(df.columns)} cols)")

def load_checkpoint():
    if CHECKPOINT_META_FILE.exists() and CHECKPOINT_DATA_FILE.exists():
        try:
            with open(CHECKPOINT_META_FILE, 'r') as f:
                meta_data = json.load(f)

            print(f"  Resuming from: {meta_data['last_step']} ({meta_data['timestamp']})")

            with open(CHECKPOINT_DATA_FILE, 'rb') as f:
                checkpoint_data = pickle.load(f)

            return checkpoint_data
        except Exception as e:
            print(f"  Warning: Could not load checkpoint: {e}")
            return None
    return None

# ============================================================================
# CSV VERSION SYSTEM
# ============================================================================
VERSION_DIR = BASE_DIR / 'csv_versions'
VERSION_DIR.mkdir(exist_ok=True)

def save_version_csv(df, version, step_name, description=""):
    """
    Save DataFrame as a versioned CSV file.
    Format: v{version:02d}_{step_name}.csv

    Args:
        df: DataFrame to save
        version: Version number (integer)
        step_name: Name of the step
        description: Optional description for the version

    Returns:
        filepath: Path to saved file
    """
    filename = f"v{version:02d}_{step_name}.csv"
    filepath = VERSION_DIR / filename

    # Save full dataframe (including intermediate columns)
    df.to_csv(filepath, index=False)

    # Save version metadata
    meta_file = VERSION_DIR / f"v{version:02d}_meta.json"
    metadata = {
        'version': version,
        'step_name': step_name,
        'description': description,
        'timestamp': datetime.now().isoformat(),
        'n_rows': len(df),
        'n_cols': len(df.columns),
        'columns': list(df.columns)
    }
    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ CSV v{version:02d}: {filename} ({len(df):,} rows, {len(df.columns)} cols)")
    return filepath

def get_latest_version():
    """Get the latest version number from csv_versions directory."""
    version_files = list(VERSION_DIR.glob('v*_meta.json'))
    if not version_files:
        return 0
    versions = []
    for f in version_files:
        try:
            v = int(f.stem.split('_')[0][1:])
            versions.append(v)
        except:
            pass
    return max(versions) if versions else 0

def save_intermediate_file(df, step_name):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"features_{step_name}_{timestamp}.csv"
    filepath = BASE_DIR / filename
    df.to_csv(filepath, index=False)
    print(f"  → Saved: {filename}")
    return filepath

# ============================================================================
# PROGRESS TRACKER
# ============================================================================
class ProgressTracker:
    """Track overall progress of feature engineering pipeline."""

    def __init__(self, total_steps=7):
        self.total_steps = total_steps
        self.completed_steps = 0
        self.step_names = [
            'Load Data',
            'Build Spatial Index',
            'Spatio-temporal Clustering',
            'Core Features',
            'LSTM Temporal Features',
            'Target Variables',
            'Final Summary'
        ]
        self.start_time = time.time()
        self.step_times = []

    def start_step(self, step_num):
        """Mark the start of a step."""
        self.current_step_start = time.time()
        percentage = (step_num / self.total_steps) * 100
        elapsed = time.time() - self.start_time

        print("\n" + "="*70)
        print(f" STEP {step_num}/{self.total_steps}: {self.step_names[step_num-1]} ")
        print("="*70)
        print(f" Overall Progress: {percentage:.1f}% | Elapsed: {self._format_time(elapsed)}")

        if self.completed_steps > 0:
            avg_time = elapsed / self.completed_steps
            remaining = avg_time * (self.total_steps - self.completed_steps)
            print(f" Estimated Time Remaining: {self._format_time(remaining)}")

    def complete_step(self):
        """Mark the completion of a step."""
        step_time = time.time() - self.current_step_start
        self.step_times.append(step_time)
        self.completed_steps += 1

        print(f"\n  ✓ Step completed in {self._format_time(step_time)}")

        if self.completed_steps < self.total_steps:
            avg_time = sum(self.step_times) / len(self.step_times)
            remaining = avg_time * (self.total_steps - self.completed_steps)
            print(f"  → Average time per step: {self._format_time(avg_time)}")
            print(f"  → Estimated remaining: {self._format_time(remaining)}")

    def _format_time(self, seconds):
        """Format seconds to readable time."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"

    def finish(self):
        """Mark the completion of all steps."""
        total_time = time.time() - self.start_time
        print("\n" + "="*70)
        print(" PIPELINE COMPLETE! ")
        print("="*70)
        print(f" Total Time: {self._format_time(total_time)}")
        print(f" Average per Step: {self._format_time(total_time/self.total_steps)}")
        print("="*70)

# ============================================================================
# NUMBA OPTIMIZED FUNCTIONS WITH PROGRESS SUPPORT
# ============================================================================

def compute_clustering_with_progress(times_numeric, mags, cartesian_coords,
                                      distance_threshold, time_window_sec,
                                      desc="Clustering"):
    """
    Wrapper for clustering with tqdm progress bar.
    Processes in chunks to show progress.
    """
    n_events = len(times_numeric)
    chunk_size = max(5000, n_events // 50)  # Show at least 50 progress updates

    sequence_ids = np.zeros(n_events, dtype=np.int64)
    seq_positions = np.zeros(n_events, dtype=np.int64)
    is_seq_mainshock = np.zeros(n_events, dtype=np.int64)

    for chunk_start in tqdm(range(0, n_events, chunk_size), desc=desc, unit="chunks"):
        chunk_end = min(chunk_start + chunk_size, n_events)

        # Process this chunk using the Numba function
        seq_ids_chunk, seq_pos_chunk, is_main_chunk = compute_clustering_vectorized(
            times_numeric[chunk_start:chunk_end],
            mags[chunk_start:chunk_end],
            cartesian_coords[chunk_start:chunk_end],
            distance_threshold, time_window_sec, chunk_end - chunk_start
        )

        sequence_ids[chunk_start:chunk_end] = seq_ids_chunk
        seq_positions[chunk_start:chunk_end] = seq_pos_chunk
        is_seq_mainshock[chunk_start:chunk_end] = is_main_chunk

    # Remap sequence IDs to be consecutive across chunks
    unique_ids = np.unique(sequence_ids)
    unique_ids = unique_ids[unique_ids > 0]  # Exclude 0
    id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_ids, 1)}
    id_mapping[0] = 0

    vectorized_map = np.vectorize(lambda x: id_mapping.get(x, 0))
    sequence_ids = vectorized_map(sequence_ids)

    return sequence_ids, seq_positions, is_seq_mainshock


def compute_aftershock_with_progress(times_numeric, mags, desc="Aftershock"):
    """
    Wrapper for aftershock detection with tqdm progress bar.
    """
    n_events = len(times_numeric)
    chunk_size = max(10000, n_events // 20)

    is_aftershock = np.zeros(n_events, dtype=np.int64)
    mainshock_mag = mags.copy()

    for chunk_start in tqdm(range(0, n_events, chunk_size), desc=desc, unit="chunks"):
        chunk_end = min(chunk_start + chunk_size, n_events)

        aft_chunk, mag_chunk = compute_aftershock_vectorized(
            times_numeric[chunk_start:chunk_end],
            mags[chunk_start:chunk_end],
            chunk_end - chunk_start
        )

        is_aftershock[chunk_start:chunk_end] = aft_chunk
        mainshock_mag[chunk_start:chunk_end] = mag_chunk

    return is_aftershock, mainshock_mag


def compute_coulomb_stress_with_progress(mags, desc="Coulomb stress", window_size=20):
    """
    Wrapper for Coulomb stress with tqdm progress bar.
    """
    n_events = len(mags)
    chunk_size = max(5000, n_events // 20)

    coulomb_stress = np.zeros(n_events, dtype=np.float64)

    for chunk_start in tqdm(range(0, n_events, chunk_size), desc=desc, unit="chunks"):
        chunk_end = min(chunk_start + chunk_size, n_events)

        # Process this chunk
        stress_chunk = compute_coulomb_stress_optimized(
            mags[chunk_start:chunk_end],
            chunk_end - chunk_start,
            window_size
        )

        coulomb_stress[chunk_start:chunk_end] = stress_chunk

    return coulomb_stress


def compute_b_value_with_progress(mags, sample_indices, n_events, desc="B-value", window_size=10000):
    """
    Wrapper for b-value with tqdm progress bar.
    """
    n_samples = len(sample_indices)

    b_values = []
    for idx_pos in tqdm(range(n_samples), desc=desc, unit="samples"):
        idx = sample_indices[idx_pos]
        b_val = compute_b_value_optimized(mags, np.array([idx]), n_events, window_size)[0]
        b_values.append(b_val)

    return np.array(b_values)


# ============================================================================
# CORE NUMBA JIT FUNCTIONS (called by wrappers above)
# ============================================================================
@jit(nopython=True, parallel=False)
def compute_clustering_vectorized(times_numeric, mags, cartesian_coords,
                                   distance_threshold, time_window_sec, n_events):
    """
    Optimized clustering algorithm using vectorized operations.
    Returns sequence_ids, seq_positions, is_seq_mainshock
    """
    sequence_ids = np.zeros(n_events, dtype=np.int64)
    seq_positions = np.zeros(n_events, dtype=np.int64)
    is_seq_mainshock = np.zeros(n_events, dtype=np.int64)

    current_seq_id = 0

    for i in range(n_events):
        if sequence_ids[i] > 0:
            continue

        current_time = times_numeric[i]
        current_mag = mags[i]
        current_cartesian = cartesian_coords[i]

        time_window_start = current_time - time_window_sec

        # Find candidate indices within time window
        candidate_indices = np.zeros(n_events, dtype=np.int64)
        n_candidates = 0

        for j in range(n_events):
            if times_numeric[j] < current_time and times_numeric[j] >= time_window_start:
                candidate_indices[n_candidates] = j
                n_candidates += 1

        found_sequence = False

        if n_candidates > 0:
            # Check spatial distance
            for k in range(n_candidates):
                idx = candidate_indices[k]
                dx = current_cartesian[0] - cartesian_coords[idx][0]
                dy = current_cartesian[1] - cartesian_coords[idx][1]
                dz = current_cartesian[2] - cartesian_coords[idx][2]
                dist = np.sqrt(dx*dx + dy*dy + dz*dz)

                if dist <= distance_threshold and mags[idx] > current_mag:
                    # Find most recent larger event
                    most_recent_idx = idx
                    for k2 in range(k + 1, n_candidates):
                        idx2 = candidate_indices[k2]
                        dx2 = current_cartesian[0] - cartesian_coords[idx2][0]
                        dy2 = current_cartesian[1] - cartesian_coords[idx2][1]
                        dz2 = current_cartesian[2] - cartesian_coords[idx2][2]
                        if np.sqrt(dx2*dx2 + dy2*dy2 + dz2*dz2) <= distance_threshold and mags[idx2] > current_mag:
                            most_recent_idx = idx2

                    if sequence_ids[most_recent_idx] > 0:
                        sequence_ids[i] = sequence_ids[most_recent_idx]
                        # Count position
                        pos = 0
                        for j in range(i):
                            if sequence_ids[j] == sequence_ids[most_recent_idx]:
                                pos += 1
                        seq_positions[i] = pos + 1
                        found_sequence = True
                    break

        if not found_sequence:
            current_seq_id += 1
            sequence_ids[i] = current_seq_id
            seq_positions[i] = 1
            is_seq_mainshock[i] = 1

    return sequence_ids, seq_positions, is_seq_mainshock


@jit(nopython=True)
def compute_aftershock_vectorized(times_numeric, mags, n_events):
    """
    Optimized aftershock detection using vectorized operations.
    """
    is_aftershock = np.zeros(n_events, dtype=np.int64)
    mainshock_mag = mags.copy()

    for i in range(n_events):
        mag_i = mags[i]
        time_i = times_numeric[i]
        time_window_sec = 10 ** (0.5 * mag_i - 1.0)

        max_candidate_mag = mag_i
        found_larger = False

        for j in range(n_events):
            if times_numeric[j] >= time_i and times_numeric[j] <= time_i + time_window_sec:
                if mags[j] > mag_i:
                    found_larger = True
                    if mags[j] > max_candidate_mag:
                        max_candidate_mag = mags[j]

        if found_larger:
            is_aftershock[i] = 1
            mainshock_mag[i] = max_candidate_mag

    return is_aftershock, mainshock_mag


@jit(nopython=True)
def compute_coulomb_stress_optimized(mags, n_events, window_size=20):
    """
    Optimized Coulomb stress computation.
    """
    coulomb_stress = np.zeros(n_events, dtype=np.float64)

    for i in range(n_events):
        start_idx = max(0, i - window_size)
        stress_sum = 0.0

        for j in range(start_idx, i):
            stress_sum += 10**(1.5 * mags[j])

        coulomb_stress[i] = stress_sum

    return coulomb_stress


@jit(nopython=True)
def compute_b_value_optimized(mags, sample_indices, n_events, window_size=10000):
    """
    Optimized b-value computation.
    """
    b_values = np.zeros(len(sample_indices), dtype=np.float64)

    for idx_pos in range(len(sample_indices)):
        idx = sample_indices[idx_pos]
        start_idx = max(0, idx - window_size)

        # Count valid magnitudes
        count_total = 0
        count_valid = 0
        mag_sum = 0.0
        mag_min = 10.0

        for j in range(start_idx, idx + 1):
            if j < n_events:
                count_total += 1
                if mags[j] >= 3.0:
                    count_valid += 1
                    mag_sum += mags[j]
                    if mags[j] < mag_min:
                        mag_min = mags[j]

        if count_total >= 50 and count_valid >= 20 and mag_min < 10.0:
            mag_mean = mag_sum / count_valid
            b_values[idx_pos] = (mag_mean - mag_min)**(-1) * np.log10(np.e)
        else:
            b_values[idx_pos] = 1.0

    return b_values


# ============================================================================
# MAIN EXECUTION
# ============================================================================
print("="*70)
print(" ADVANCED FEATURES - LSTM OPTIMIZED + CHECKPOINT (V4.0) ")
print("="*70)

ALL_STEPS = [
    'step1_load_data',
    'step2_build_index',
    'step3_clustering',
    'step4_core_features',
    'step5_lstm_temporal',
    'step6_targets'
]

# Initialize progress tracker
progress = ProgressTracker(total_steps=7)

# ============================================================================
# CHECKPOINT SYSTEM
# ============================================================================
print("\n[Checkpoint System]")
checkpoint = load_checkpoint()

if checkpoint:
    df_work = checkpoint['df']
    completed_steps = checkpoint['completed_steps']
    current_version = get_latest_version()
    progress.completed_steps = len(completed_steps)
    print(f"  Skipping: {completed_steps}")
    print(f"  Latest CSV version: v{current_version:02d}")
    print(f"  Resuming from step {progress.completed_steps + 1}/7")
else:
    completed_steps = []
    current_version = 0
    print("  No checkpoint found - starting fresh")
    print(f"  CSV versions will be saved to: {VERSION_DIR}")

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
if 'step1_load_data' not in completed_steps:
    progress.start_step(1)
    df = pd.read_csv(BASE_DIR.parent / 'dongdat.csv')
    df['time'] = pd.to_datetime(df['time'])

    # Filter by year range
    print(f"\n  Filtering data: {START_YEAR}-{END_YEAR-1}")
    original_count = len(df)
    df = df[(df['time'].dt.year >= START_YEAR) & (df['time'].dt.year < END_YEAR)]
    filtered_count = len(df)
    removed_count = original_count - filtered_count
    print(f"  Original: {original_count:,} rows")
    print(f"  Filtered: {filtered_count:,} rows ({filtered_count/original_count*100:.1f}%)")
    print(f"  Removed: {removed_count:,} rows")

    # Calculate region_code based on geographic grid (~50km)
    # Grid size: 0.5 degrees ≈ 55km
    GRID_SIZE_DEG = 0.5  # ~55km

    def calculate_region_code(lat, lon, grid_size=GRID_SIZE_DEG):
        """
        Calculate region code based on lat/lon grid.
        Format: R{lat_int:03d}_{lon_int:03d}
        """
        # Offset to handle negative coordinates
        lat_offset = 90  # to make all latitudes positive (0-180)
        lon_offset = 180  # to make all longitudes positive (0-360)

        lat_int = int((lat + lat_offset) / grid_size)
        lon_int = int((lon + lon_offset) / grid_size)

        return f"R{lat_int:03d}_{lon_int:03d}"

    df['region_code'] = df.apply(
        lambda row: calculate_region_code(row['latitude'], row['longitude']),
        axis=1
    )

    # Sort by REGION first, then by TIME within each region
    # This allows LSTM to learn regional earthquake patterns
    df = df.sort_values(['region_code', 'time']).reset_index(drop=True)
    df_work = df.copy().reset_index(drop=True)

    # Handle NaN values for sparse features (mmi, cdi, felt)
    # These features have low availability (<2%) but are valuable when present
    sparse_features = ['mmi', 'cdi', 'felt']
    for feat in sparse_features:
        if feat in df_work.columns:
            nan_count = df_work[feat].isna().sum()
            if nan_count > 0:
                df_work[feat] = df_work[feat].fillna(0)
                print(f"  {feat}: filled {nan_count:,} NaN values with 0")

    # Show region statistics
    n_regions = df_work['region_code'].nunique()
    print(f"  Events: {len(df_work):,} | Regions: {n_regions:,}")
    print(f"  Mag: {df_work['mag'].min():.1f} - {df_work['mag'].max():.1f}")
    print(f"  Grid size: {GRID_SIZE_DEG}° (~{GRID_SIZE_DEG * 111:.0f}km)")
    print(f"  Sorting: region_code → time")

    current_version += 1
    save_version_csv(df_work, current_version, 'step1_load_data',
                     description=f"Data sorted by region_code then time (grid: {GRID_SIZE_DEG}°), NaN filled for sparse features")
    save_checkpoint(df_work, ['step1_load_data'], 'step1_load_data')
    completed_steps = ['step1_load_data']
    progress.complete_step()
else:
    print("\n[1/7] ✓ SKIPPED")
    progress.completed_steps += 1

# ============================================================================
# STEP 2: BUILD SPATIAL INDEX
# ============================================================================
if 'step2_build_index' not in completed_steps:
    progress.start_step(2)

    coords = df_work[['latitude', 'longitude']].values
    lat_rad = np.radians(coords[:, 0])
    lon_rad = np.radians(coords[:, 1])
    R = 6371.0

    x = R * np.cos(lat_rad) * np.cos(lon_rad)
    y = R * np.cos(lat_rad) * np.sin(lon_rad)
    z = R * np.sin(lat_rad)
    cartesian_coords = np.column_stack([x, y, z])
    kdtree_cartesian = cKDTree(cartesian_coords)

    df_work['_cartesian_x'] = x
    df_work['_cartesian_y'] = y
    df_work['_cartesian_z'] = z

    current_version += 1
    save_version_csv(df_work, current_version, 'step2_spatial_index',
                     description="Data with cartesian coordinates for KD-tree spatial queries")
    save_checkpoint(df_work, completed_steps + ['step2_build_index'], 'step2_build_index')
    completed_steps.append('step2_build_index')
    progress.complete_step()
else:
    print("\n[2/7] ✓ SKIPPED")
    progress.completed_steps += 1
    x = df_work['_cartesian_x'].values
    y = df_work['_cartesian_y'].values
    z = df_work['_cartesian_z'].values
    cartesian_coords = np.column_stack([x, y, z])
    kdtree_cartesian = cKDTree(cartesian_coords)

# ============================================================================
# STEP 3: SPATIO-TEMPORAL CLUSTERING (OPTIMIZED)
# ============================================================================
# Define earthquake sequences based on spatial and temporal proximity
# - Distance threshold: 50km
# - Time window: 72 hours
# ============================================================================
if 'step3_clustering' not in completed_steps:
    progress.start_step(3)

    # Clustering parameters
    DISTANCE_THRESHOLD_KM = 50.0  # km
    TIME_WINDOW_HOURS = 72.0  # hours
    TIME_WINDOW_SEC = TIME_WINDOW_HOURS * 3600

    times_numeric = df_work['time'].values.astype(np.int64) / 1e9
    mags = df_work['mag'].values
    n_events = len(df_work)

    print(f"  Parameters: distance < {DISTANCE_THRESHOLD_KM}km, time < {TIME_WINDOW_HOURS}h")

    # Use optimized clustering function with progress bar
    print(f"  Running clustering on {n_events:,} events...")

    sequence_ids, seq_positions, is_seq_mainshock = compute_clustering_with_progress(
        times_numeric, mags, cartesian_coords,
        DISTANCE_THRESHOLD_KM, TIME_WINDOW_SEC,
        desc="  Clustering progress"
    )
    print(f"  ✓ Clustering completed!")

    df_work['sequence_id'] = sequence_ids
    df_work['seq_position'] = seq_positions
    df_work['is_seq_mainshock'] = is_seq_mainshock

    # Compute additional sequence-level features - OPTIMIZED
    print("  Computing sequence-level features...")

    # Use groupby for efficiency
    seq_info = df_work[df_work['sequence_id'] > 0].groupby('sequence_id').agg({
        'mag': 'first',  # Mainshock mag is first in sequence
        'sequence_id': 'count'  # Sequence length
    }).rename(columns={'mag': 'mainshock_mag', 'sequence_id': 'seq_len'})

    # Map to dataframe
    seq_mainshock_mag_map = seq_info['mainshock_mag'].to_dict()
    seq_length_map = seq_info['seq_len'].to_dict()

    df_work['seq_mainshock_mag'] = df_work['sequence_id'].map(seq_mainshock_mag_map).fillna(0)
    df_work['seq_length'] = df_work['sequence_id'].map(seq_length_map).fillna(0).astype(int)

    # Time since sequence start - OPTIMIZED
    mainshock_times = df_work[df_work['is_seq_mainshock'] == 1].set_index('sequence_id')['time'].to_dict()

    def get_mainshock_time(seq_id):
        return mainshock_times.get(seq_id, df_work['time'].iloc[0])

    mainshock_time_series = df_work['sequence_id'].apply(get_mainshock_time)
    df_work['time_since_seq_start_sec'] = (df_work['time'] - mainshock_time_series).dt.total_seconds().fillna(0)

    # Print statistics
    n_sequences = df_work['sequence_id'].nunique() - 1  # Exclude 0
    n_mainshocks = df_work['is_seq_mainshock'].sum()
    avg_seq_length = df_work[df_work['sequence_id'] > 0].groupby('sequence_id').size().mean()

    print(f"  ✓ Found {n_sequences:,} sequences")
    print(f"  ✓ Mainshocks: {n_mainshocks:,}")
    print(f"  ✓ Avg sequence length: {avg_seq_length:.1f} events")

    current_version += 1
    save_version_csv(df_work, current_version, 'step3_spatial_temporal_clustering',
                     description=f"Spatio-temporal clustering: {n_sequences} sequences, {DISTANCE_THRESHOLD_KM}km/{TIME_WINDOW_HOURS}h")
    save_checkpoint(df_work, completed_steps + ['step3_clustering'], 'step3_clustering')
    completed_steps.append('step3_clustering')
    progress.complete_step()
else:
    print("\n[3/7] ✓ SKIPPED")
    progress.completed_steps += 1

# ============================================================================
# STEP 4: CORE FEATURES (OPTIMIZED)
# ============================================================================
if 'step4_core_features' not in completed_steps:
    progress.start_step(4)

    times_numeric = df_work['time'].values.astype(np.int64) / 1e9
    mags = df_work['mag'].values
    n_events = len(df_work)

    # 4.1 Aftershock detection - OPTIMIZED with Numba + tqdm
    print("  [1/4] Computing aftershock features...")
    is_aftershock, mainshock_mag = compute_aftershock_with_progress(
        times_numeric, mags,
        desc="     Aftershock detection"
    )
    df_work['is_aftershock'] = is_aftershock
    df_work['mainshock_mag'] = mainshock_mag
    print("       ✓ Done!")

    # 4.2 Seismicity density - OPTIMIZED
    print("  [2/4] Computing seismicity density...")
    start_t = time.time()
    lat_bins = np.linspace(df_work['latitude'].min(), df_work['latitude'].max(), 100)
    lon_bins = np.linspace(df_work['longitude'].min(), df_work['longitude'].max(), 100)

    df_work['lat_bin'] = pd.cut(df_work['latitude'], bins=lat_bins, labels=False)
    df_work['lon_bin'] = pd.cut(df_work['longitude'], bins=lon_bins, labels=False)
    df_work['spatial_bin'] = df_work['lat_bin'] * 100 + df_work['lon_bin']

    density_map = df_work['spatial_bin'].value_counts()
    df_work['seismicity_density'] = df_work['spatial_bin'].map(density_map).fillna(1)
    df_work['seismicity_density_100km'] = df_work['seismicity_density'] / 100
    print(f"       → Done in {time.time() - start_t:.1f}s")

    # 4.3 Coulomb stress proxy - OPTIMIZED with Numba + tqdm
    print("  [3/4] Computing Coulomb stress...")
    coulomb_stress = compute_coulomb_stress_with_progress(
        mags,
        desc="     Coulomb stress",
        window_size=20
    )
    df_work['coulomb_stress_proxy'] = coulomb_stress
    print("       ✓ Done!")

    # 4.4 Regional b-value - OPTIMIZED with Numba + tqdm
    print("  [4/4] Computing b-value...")
    sample_size = min(1000, n_events)
    sample_indices = np.linspace(0, n_events-1, sample_size, dtype=int)

    b_values = compute_b_value_with_progress(
        mags, sample_indices, n_events,
        desc="     B-value",
        window_size=10000
    )

    f_b = interp1d(sample_indices, b_values, kind='linear',
                    bounds_error=False, fill_value=(np.mean(b_values), np.mean(b_values)))
    df_work['regional_b_value'] = f_b(np.arange(n_events))
    print("       ✓ Done!")

    current_version += 1
    save_version_csv(df_work, current_version, 'step4_core_features',
                     description="Core features: is_aftershock, mainshock_mag, seismicity_density, coulomb_stress, regional_b_value")
    save_checkpoint(df_work, completed_steps + ['step4_core_features'], 'step4_core_features')
    completed_steps.append('step4_core_features')
    progress.complete_step()
else:
    print("\n[4/7] ✓ SKIPPED")
    progress.completed_steps += 1

# ============================================================================
# STEP 5: LSTM TEMPORAL FEATURES (OPTIMIZED)
# ============================================================================
if 'step5_lstm_temporal' not in completed_steps:
    progress.start_step(5)

    n_events = len(df_work)  # Define n_events for this step

    # 5.1 Time since last event - ALREADY OPTIMIZED (vectorized)
    print("  [1/3] Computing time since last event...")
    start_t = time.time()
    df_work['time_since_last_event'] = df_work['time'].diff().dt.total_seconds().fillna(0)
    print(f"       → Done in {time.time() - start_t:.1f}s")

    # 5.2 Time since last M5+ - OPTIMIZED with binary search
    print("  [2/3] Computing time since M5+...")
    start_t = time.time()
    m5_indices = df_work[df_work['mag'] >= 5.0].index.tolist()
    print(f"       Found {len(m5_indices)} M5+ events")

    if len(m5_indices) > 0:
        m5_times = df_work.loc[m5_indices, 'time'].values

        # Use binary search for efficient time lookup with progress
        last_m5_time = []
        current_times = df_work['time'].values

        for i in tqdm(range(len(df_work)), desc="       Processing", leave=False):
            current_time = current_times[i]
            # Find the most recent M5+ event using binary search
            idx = bisect.bisect_left(m5_times, current_time) - 1
            if idx >= 0:
                # Convert numpy timedelta64 to seconds
                time_diff = current_time - m5_times[idx]
                last_m5_time.append(time_diff.astype('timedelta64[s]').astype(int))
            else:
                last_m5_time.append(365 * 24 * 3600)  # Default: 1 year

        df_work['time_since_last_M5'] = last_m5_time
    else:
        df_work['time_since_last_M5'] = 365 * 24 * 3600  # Default: 1 year
    print(f"       → Done in {time.time() - start_t:.1f}s")

    # 5.3 Interval sequence (last 5 intervals) - OPTIMIZED
    print("  [3/3] Computing interval lags...")
    start_t = time.time()
    time_diffs = df_work['time'].diff().dt.total_seconds().fillna(0).values

    # Vectorized computation for interval lags using np.roll
    # interval_lag1 = most recent interval, interval_lag2 = second most recent, etc.
    for lag in range(1, 6):
        # Shift time_diffs by 'lag' positions and fill beginning with 0
        lag_values = np.roll(time_diffs, shift=lag)
        lag_values[:lag] = 0  # Fill first 'lag' positions with 0
        df_work[f'interval_lag{lag}'] = lag_values
    print(f"       → Done in {time.time() - start_t:.1f}s")

    current_version += 1
    save_version_csv(df_work, current_version, 'step5_lstm_temporal',
                   description="LSTM temporal features: time intervals, intervals lag features")
    save_checkpoint(df_work, completed_steps + ['step5_lstm_temporal'], 'step5_lstm_temporal')
    completed_steps.append('step5_lstm_temporal')
    progress.complete_step()
else:
    print("\n[5/7] ✓ SKIPPED")
    progress.completed_steps += 1

# ============================================================================
# STEP 6: TARGET VARIABLES
# ============================================================================
if 'step6_targets' not in completed_steps:
    progress.start_step(6)

    df_work['target_time_to_next'] = df_work['time'].diff(-1).dt.total_seconds().abs().fillna(0)
    df_work['target_next_mag'] = df_work['mag'].shift(-1).fillna(df_work['mag'].iloc[-1])
    df_work['target_next_mag_binary'] = (df_work['target_next_mag'] >= 5.0).astype(int)

    current_version += 1
    save_version_csv(df_work, current_version, 'step6_targets',
                   description="Final dataset with all features and targets: time_to_next, next_mag, next_mag_binary")
    save_checkpoint(df_work, completed_steps + ['step6_targets'], 'step6_targets')
    completed_steps.append('step6_targets')
    progress.complete_step()
else:
    print("\n[6/7] ✓ SKIPPED")
    progress.completed_steps += 1

# ============================================================================
# SELECT FINAL FEATURES AND SAVE
# ============================================================================
progress.start_step(7)
print("\n" + "="*70)
print(" SUMMARY ")
print("="*70)

original_features = ['time', 'latitude', 'longitude', 'depth', 'mag',
                     'sig', 'mmi', 'cdi', 'felt', 'region_code']
core_features = ['is_aftershock', 'mainshock_mag', 'seismicity_density_100km',
                 'coulomb_stress_proxy', 'regional_b_value']
sequence_features = ['sequence_id', 'seq_position', 'is_seq_mainshock',
                     'seq_mainshock_mag', 'seq_length', 'time_since_seq_start_sec']
lstm_features = ['time_since_last_event', 'time_since_last_M5',
                 'interval_lag1', 'interval_lag2', 'interval_lag3', 'interval_lag4', 'interval_lag5']
target_features = ['target_time_to_next', 'target_next_mag', 'target_next_mag_binary']

final_features = original_features + core_features + sequence_features + lstm_features + target_features
df_final = df_work[final_features].copy()

print(f"\nFeatures: {len(final_features)} total")
print(f"  Original: 10 | Core: 5 | Sequence: 6 | LSTM: 5 | Targets: 3")

# Save to CSV
output_file = BASE_DIR / 'features_lstm.csv'
df_final.to_csv(output_file, index=False)

# Save as version CSV
current_version += 1
save_version_csv(df_final, current_version, 'final_features',
               description=f"Final features: {len(final_features)} total (10 original + 5 core + 6 sequence + 5 LSTM + 3 targets)")

print(f"\n✓ SAVED: {output_file}")
print(f"  Rows: {len(df_final):,} | Columns: {len(df_final.columns)}")
print(f"  Size: {os.path.getsize(output_file) / 1024 / 1024:.1f} MB")

progress.complete_step()
progress.finish()

print(f"\nFeatures ready for LSTM training!")
print(f"Main file: {output_file}")
print(f"Version files saved in: {VERSION_DIR}")
print(f"Total versions created: {current_version}")
print(f"Total: {len(final_features)} features")
