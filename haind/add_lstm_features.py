"""
Add LSTM-Specific Features for Earthquake Prediction
Task: Predict time-to-next and magnitude-of-next earthquake

Features:
- Core features (5): aftershock, density, coulomb stress, b-value
- LSTM Critical: Temporal intervals, rolling statistics, AMR, targets
- Targets: time_to_next, next_mag

Author: haind
Project: Earthquake Sequence Mining
Created: 2025-03-23
"""

import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = BASE_DIR.parent / 'dongdat.csv'
OUTPUT_FILE = BASE_DIR / 'features_lstm.csv'

print("="*70)
print(" LSTM FEATURES FOR EARTHQUAKE PREDICTION ")
print("="*70)
print(f"Data file: {DATA_FILE}")
print(f"Output file: {OUTPUT_FILE}")

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[1/6] Loading data...")
df = pd.read_csv(DATA_FILE)
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)
print(f"  Total events: {len(df):,}")

# ============================================================================
# STEP 2: CORE FEATURES (5 essential features)
# ============================================================================
print("\n[2/6] Computing core features...")

# 2.1 Aftershock detection (simplified Gardner-Knopoff)
print("  Computing aftershock detection...")
times_numeric = df['time'].values.astype(np.int64) / 1e9
mags = df['mag'].values

n_events = len(df)
is_aftershock = np.zeros(n_events, dtype=bool)
mainshock_mag = mags.copy()

# Simplified: Check if there's a larger event within time/distance window
# Using time-only for speed (can add spatial later if needed)
for i in tqdm(range(n_events), desc="  Aftershock"):
    mag_i = mags[i]
    time_i = times_numeric[i]

    # Time window based on magnitude
    time_window_sec = 10 ** (0.5 * mag_i - 1.0)

    # Find events in time window AFTER this event
    time_mask = (times_numeric >= time_i) & (times_numeric <= time_i + time_window_sec)
    candidate_indices = np.where(time_mask)[0]

    if len(candidate_indices) > 1:
        # Check for larger mainshock
        candidate_mags = mags[candidate_indices]
        larger_mainshock = candidate_mags > mag_i

        if np.any(larger_mainshock):
            is_aftershock[i] = True
            mainshock_mag[i] = candidate_mags[larger_mainshock].max()

df['is_aftershock'] = is_aftershock
df['mainshock_mag'] = mainshock_mag
print(f"  Aftershocks: {is_aftershock.sum():,} ({is_aftershock.sum()/len(df)*100:.1f}%)")

# 2.2 Seismicity density (100km radius - simplified)
print("  Computing seismicity density...")
# Simplified: Count events in spatial bins (can use kdtree for accuracy)
lat_bins = np.linspace(df['latitude'].min(), df['latitude'].max(), 100)
lon_bins = np.linspace(df['longitude'].min(), df['longitude'].max(), 100)

df['lat_bin'] = pd.cut(df['latitude'], bins=lat_bins, labels=False)
df['lon_bin'] = pd.cut(df['longitude'], bins=lon_bins, labels=False)
df['spatial_bin'] = df['lat_bin'] * 100 + df['lon_bin']

# Count events in same spatial bin (approximation for 100km)
density_map = df['spatial_bin'].value_counts()
df['seismicity_density'] = df['spatial_bin'].map(density_map).fillna(1)

# Normalize by area (approximate)
df['seismicity_density_100km'] = df['seismicity_density'] / 100  # Normalized
print(f"  Mean density: {df['seismicity_density_100km'].mean():.2f}")

# 2.3 Coulomb stress proxy (simplified - based on recent events)
print("  Computing Coulomb stress proxy...")
coulomb_stress = np.zeros(n_events)

for i in tqdm(range(min(10000, n_events)), desc="  Coulomb stress"):
    # Look at last 20 events
    start_idx = max(0, i - 20)
    prev_mags = mags[start_idx:i]

    if len(prev_mags) > 0:
        # Stress is proportional to seismic moment
        stress_contributions = 10**(1.5 * prev_mags)
        coulomb_stress[i] = np.sum(stress_contributions)

# Interpolate for rest (for speed)
if n_events > 10000:
    from scipy.interpolate import interp1d
    sample_indices = np.arange(0, n_events, n_events // 1000)
    f_coulomb = interp1d(sample_indices, coulomb_stress[sample_indices],
                          kind='linear', bounds_error=False,
                          fill_value=(coulomb_stress[:1000].mean(), coulomb_stress[-1000:].mean()))
    coulomb_stress = f_coulomb(np.arange(n_events))

df['coulomb_stress_proxy'] = coulomb_stress
print(f"  Mean stress: {coulomb_stress.mean():.2e}")

# 2.4 Regional b-value (simplified)
print("  Computing regional b-value...")
sample_size = min(1000, n_events)
sample_indices = np.linspace(0, n_events-1, sample_size, dtype=int)

b_values = []
for idx in tqdm(sample_indices, desc="  B-value"):
    # Use recent events (1 year approx)
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

# Interpolate
from scipy.interpolate import interp1d
f_b = interp1d(sample_indices, b_values, kind='linear',
                bounds_error=False, fill_value=(np.mean(b_values), np.mean(b_values)))
df['regional_b_value'] = f_b(np.arange(n_events))
print(f"  Mean b-value: {np.mean(b_values):.2f}")

# ============================================================================
# STEP 3: LSTM CRITICAL FEATURES - Temporal Intervals
# ============================================================================
print("\n[3/6] Computing LSTM temporal features...")

# 3.1 Time since last event
df['time_since_last_event'] = df['time'].diff().dt.total_seconds()
df['time_since_last_event'] = df['time_since_last_event'].fillna(0)
print("  ✓ time_since_last_event")

# 3.2 Time since last M5+ event
print("  Computing time since last M5+...")
m5_times = df[df['mag'] >= 5.0]['time']
last_m5_time = pd.Series(index=df.index, dtype=float)

for i in tqdm(range(len(df)), desc="  Time since M5"):
    current_time = df.loc[i, 'time']
    past_m5 = m5_times[m5_times < current_time]
    if len(past_m5) > 0:
        last_m5_time[i] = (current_time - past_m5.max()).total_seconds()
    else:
        last_m5_time[i] = 365 * 24 * 3600  # Default: 1 year

df['time_since_last_M5'] = last_m5_time
print("  ✓ time_since_last_M5")

# 3.3 Interval sequence (last 5 intervals) - CRITICAL for LSTM
print("  Computing interval sequences...")
intervals_seq = []
for i in tqdm(range(len(df)), desc="  Interval seq"):
    if i < 5:
        intervals_seq.append([0] * 5)
    else:
        recent_times = df.loc[i-5:i, 'time'].values
        diffs = np.diff(recent_times).astype('timedelta64[s]').astype(float)
        # Pad if needed
        diffs_padded = np.pad(diffs, (5-len(diffs), 0), mode='constant')
        intervals_seq.append(diffs_padded)

# Create columns for each lag
for lag in range(5):
    df[f'interval_lag{lag+1}'] = [seq[lag] for seq in intervals_seq]
print("  ✓ interval_lag1 to interval_lag5")

# ============================================================================
# STEP 4: LSTM ROLLING STATISTICS
# ============================================================================
print("\n[4/6] Computing rolling statistics...")

# Convert time to numeric for rolling calculations
df['time_numeric'] = df['time'].astype(np.int64) / 1e9

# Define rolling windows (in seconds)
windows = {
    '1h': 3600,
    '6h': 6 * 3600,
    '24h': 24 * 3600,
    '7d': 7 * 24 * 3600,
    '30d': 30 * 24 * 3600
}

for window_name, window_sec in windows.items():
    print(f"  Computing rolling {window_name}...")

    count_list = []
    mean_mag_list = []
    max_mag_list = []

    for i in tqdm(range(len(df)), desc=f"   {window_name}"):
        current_time = df.loc[i, 'time_numeric']
        min_time = current_time - window_sec

        # Find events in time window
        time_mask = (df['time_numeric'] >= min_time) & (df['time_numeric'] <= current_time)
        events_in_window = df[time_mask]

        count_list.append(len(events_in_window))

        if len(events_in_window) > 0:
            mean_mag_list.append(events_in_window['mag'].mean())
            max_mag_list.append(events_in_window['mag'].max())
        else:
            mean_mag_list.append(0)
            max_mag_list.append(0)

    df[f'rolling_count_{window_name}'] = count_list
    df[f'rolling_mean_mag_{window_name}'] = mean_mag_list
    df[f'rolling_max_mag_{window_name}'] = max_mag_list

print("  ✓ Rolling statistics computed")

# ============================================================================
# STEP 5: PRECURSOR FEATURES (AMR, Z-value)
# ============================================================================
print("\n[5/6] Computing precursor features...")

# 5.1 AMR (Accelerating Moment Release)
print("  Computing AMR (Accelerating Moment Release)...")
# Seismic moment: M0 = 10^(1.5*M + 9.1) in N⋅m
df['seismic_moment'] = 10 ** (1.5 * df['mag'] + 9.1)
df['cumulative_moment'] = df['seismic_moment'].cumsum()

# Compute AMR as slope of log(cumulative_moment) over last 100 events
amr_values = []
for i in tqdm(range(len(df)), desc="  AMR"):
    if i < 100:
        amr_values.append(0)
    else:
        recent_moment = df.loc[i-100:i, 'cumulative_moment'].values
        log_moment = np.log10(recent_moment)
        # Linear fit: slope indicates acceleration
        slope = np.polyfit(range(100), log_moment, 1)[0]
        amr_values.append(slope)

df['AMR'] = amr_values
print("  ✓ AMR computed")

# 5.2 Z-value (statistical significance of rate change)
print("  Computing Z-value...")
z_values = []
window_events = 100  # Compare recent 100 vs previous 100

for i in tqdm(range(len(df)), desc="  Z-value"):
    if i < window_events * 2:
        z_values.append(0)
    else:
        # Recent window
        recent_idx = range(i - window_events, i)
        recent_count = len(recent_idx)

        # Previous window
        prev_idx = range(i - window_events * 2, i - window_events)
        prev_count = len(prev_idx)

        # Compare rates (events per time)
        if recent_count > 0 and prev_count > 0:
            recent_time = df.loc[i-1, 'time_numeric'] - df.loc[i-window_events, 'time_numeric']
            prev_time = df.loc[i-window_events-1, 'time_numeric'] - df.loc[i-window_events*2, 'time_numeric']

            if recent_time > 0 and prev_time > 0:
                rate_recent = recent_count / recent_time
                rate_prev = prev_count / prev_time

                # Z-test for rate difference
                if rate_prev > 0:
                    z = (rate_recent - rate_prev) / np.sqrt(rate_prev / window_events)
                    z_values.append(z)
                else:
                    z_values.append(0)
            else:
                z_values.append(0)
        else:
            z_values.append(0)

df['Z_value'] = z_values
print("  ✓ Z-value computed")

# 5.3 Benioff strain (alternative to moment release)
print("  Computing Benioff strain...")
# Benioff strain: Σ√E where E = 10^(1.5M + 4.8) in joules
df['energy'] = 10 ** (1.5 * df['mag'] + 4.8)
df['benioff_strain'] = np.sqrt(df['energy']).cumsum()
print("  ✓ Benioff strain computed")

# ============================================================================
# STEP 6: TARGET VARIABLES
# ============================================================================
print("\n[6/6] Creating target variables...")

# 6.1 Time to next earthquake
df['target_time_to_next'] = df['time'].diff(-1).dt.total_seconds().abs()
df['target_time_to_next'] = df['target_time_to_next'].fillna(0)  # Last event
print("  ✓ target_time_to_next")

# 6.2 Magnitude of next earthquake
df['target_next_mag'] = df['mag'].shift(-1)
df['target_next_mag'] = df['target_next_mag'].fillna(df['mag'].iloc[-1])
print("  ✓ target_next_mag")

# 6.3 Binary target: next earthquake >= M5.0
df['target_next_mag_binary'] = (df['target_next_mag'] >= 5.0).astype(int)
print("  ✓ target_next_mag_binary")

# ============================================================================
# SELECT FINAL FEATURES
# ============================================================================
print("\n" + "="*70)
print(" SELECTING FINAL FEATURES ")
print("="*70)

# Core features (5)
core_features = [
    'is_aftershock',
    'mainshock_mag',
    'seismicity_density_100km',
    'coulomb_stress_proxy',
    'regional_b_value'
]

# LSTM critical features
lstm_features = [
    'time_since_last_event',
    'time_since_last_M5',
    'interval_lag1', 'interval_lag2', 'interval_lag3', 'interval_lag4', 'interval_lag5',
    'rolling_count_1h', 'rolling_count_6h', 'rolling_count_24h',
    'rolling_count_7d', 'rolling_count_30d',
    'rolling_mean_mag_1h', 'rolling_mean_mag_6h', 'rolling_mean_mag_24h',
    'rolling_mean_mag_7d', 'rolling_mean_mag_30d',
    'rolling_max_mag_1h', 'rolling_max_mag_6h', 'rolling_max_mag_24h',
    'rolling_max_mag_7d', 'rolling_max_mag_30d',
    'AMR', 'Z_value', 'benioff_strain'
]

# Targets
target_features = [
    'target_time_to_next',
    'target_next_mag',
    'target_next_mag_binary'
]

# Original features to keep
original_features = ['time', 'latitude', 'longitude', 'depth', 'mag']

# Combine all
final_features = original_features + core_features + lstm_features + target_features

# Create final dataframe
df_final = df[final_features].copy()

print(f"\nFinal features: {len(final_features)}")
print("  - Original: 5")
print("  - Core: 5")
print("  - LSTM-specific: 27")
print("  - Targets: 3")

# ============================================================================
# SAVE OUTPUT
# ============================================================================
print("\n" + "="*70)
print(" SAVING OUTPUT ")
print("="*70)

# Save to CSV
df_final.to_csv(OUTPUT_FILE, index=False)

print(f"\n✓ SAVED: {OUTPUT_FILE}")
print(f"  Rows: {len(df_final):,}")
print(f"  Columns: {len(df_final.columns)}")
print(f"  Size: {os.path.getsize(OUTPUT_FILE) / 1024 / 1024:.1f} MB")

# Print feature list
print(f"\nFEATURE LIST:")
print("\n1. ORIGINAL FEATURES:")
for feat in original_features:
    print(f"   - {feat}")

print("\n2. CORE FEATURES (5):")
for feat in core_features:
    print(f"   - {feat}")

print("\n3. LSTM CRITICAL FEATURES (27):")
print("   Temporal Intervals:")
for feat in ['time_since_last_event', 'time_since_last_M5'] + [f'interval_lag{i}' for i in range(1, 6)]:
    print(f"   - {feat}")
print("   Rolling Counts:")
for feat in [f'rolling_count_{w}' for w in ['1h', '6h', '24h', '7d', '30d']]:
    print(f"   - {feat}")
print("   Rolling Mean Mag:")
for feat in [f'rolling_mean_mag_{w}' for w in ['1h', '6h', '24h', '7d', '30d']]:
    print(f"   - {feat}")
print("   Rolling Max Mag:")
for feat in [f'rolling_max_mag_{w}' for w in ['1h', '6h', '24h', '7d', '30d']]:
    print(f"   - {feat}")
print("   Precursors:")
for feat in ['AMR', 'Z_value', 'benioff_strain']:
    print(f"   - {feat}")

print("\n4. TARGET VARIABLES (3):")
for feat in target_features:
    print(f"   - {feat}")

print("\n" + "="*70)
print(" COMPLETE! ")
print("="*70)
print("\nReady for LSTM training!")
print("\nNext steps:")
print("  1. Use sequences of 50-100 events as LSTM input")
print("  2. Predict: target_time_to_next, target_next_mag")
print("  3. Consider splitting by time (train on early data, test on recent)")
print("="*70)
