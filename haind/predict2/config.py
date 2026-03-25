"""
Configuration for Split LSTM Models
TimeLSTM and MagLSTM configurations

Author: haind
Date: 2025-03-25
"""

import os
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = BASE_DIR.parent
MODEL_DIR = BASE_DIR / 'models'
LOG_DIR = BASE_DIR / 'logs'
DATA_PROCESSED_DIR = BASE_DIR / 'data_processed'

# Original features file (input to split_data.py)
FEATURES_FILE = str(DATA_DIR / 'features_lstm.csv')

# Split data files (output from split_data.py)
FEATURES_TIME_FILE = str(DATA_PROCESSED_DIR / 'features_time.csv')
FEATURES_MAG_FILE = str(DATA_PROCESSED_DIR / 'features_mag.csv')

# Create directories
os.makedirs(str(MODEL_DIR), exist_ok=True)
os.makedirs(str(LOG_DIR), exist_ok=True)
os.makedirs(str(DATA_PROCESSED_DIR), exist_ok=True)

# =============================================================================
# MODEL PARAMETERS
# =============================================================================
SEQUENCE_LENGTH = 5
BATCH_SIZE = 64
EPOCHS = 50
TRAIN_TEST_SPLIT = 0.8
VALIDATION_SPLIT = 0.1
LEARNING_RATE = 0.001
DROPOUT = 0.3
EARLY_STOPPING_PATIENCE = 15

# =============================================================================
# FEATURES (per FEATURES_GUIDE.html)
# =============================================================================

# Time Model Features (23 input features)
# Goal: Predict probability of earthquake in next 7 days (binary classification)
TIME_FEATURES = [
    # Original (4 features)
    'mag', 'depth', 'sig', 'region_code',
    # Core (4 features)
    'is_aftershock', 'mainshock_mag', 'seismicity_density_100km', 'coulomb_stress_proxy',
    # Sequence (6 features)
    'sequence_id', 'seq_position', 'is_seq_mainshock', 'seq_mainshock_mag',
    'seq_length', 'time_since_seq_start_sec',
    # LSTM Temporal (9 features)
    'time_since_last_event', 'time_since_last_M5',
    'interval_lag1', 'interval_lag2', 'interval_lag3',
    'interval_lag4', 'interval_lag5'
]
# Excluded: latitude, longitude (spatial features less important for time prediction)

# Mag Model Features (25 input features)
# Goal: Predict next_mag (magnitude)
MAG_FEATURES = [
    # Original (9 features)
    'mag', 'depth', 'latitude', 'longitude', 'sig', 'mmi', 'cdi', 'felt', 'region_code',
    # Core (5 features)
    'is_aftershock', 'mainshock_mag', 'seismicity_density_100km',
    'coulomb_stress_proxy', 'regional_b_value',
    # Sequence (5 features)
    'seq_position', 'is_seq_mainshock', 'seq_mainshock_mag',
    'seq_length', 'time_since_seq_start_sec',
    # LSTM Temporal (9 features)
    'time_since_last_event', 'time_since_last_M5',
    'interval_lag1', 'interval_lag2', 'interval_lag3',
    'interval_lag4', 'interval_lag5'
]
# Excluded: sequence_id (less important for mag prediction)

# Targets
TIME_TARGET = 'target_quake_in_7days'  # Binary: 1 if earthquake within 7 days, else 0
MAG_TARGET = 'target_next_mag'

# =============================================================================
# FEATURE INFO
# =============================================================================
print(f"{'='*70}")
print(" PREDICT2 CONFIGURATION - SPLIT MODELS")
print(f"{'='*70}")
print(f"Base dir: {str(BASE_DIR)}")
print(f"Features file: {FEATURES_FILE}")
print(f"Model dir: {str(MODEL_DIR)}")
print(f"")
print(f"TIME Model: {len(TIME_FEATURES)} input features -> {TIME_TARGET}")
print(f"MAG Model:  {len(MAG_FEATURES)} input features -> {MAG_TARGET}")
print(f"{'='*70}")
