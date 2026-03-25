"""
Earthquake Prediction Pipeline
Complete flow: Training → Prediction

Author: haind
Project: Earthquake Sequence Mining
Date: 2025-03-25
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = BASE_DIR.parent
FEATURES_FILE = str(DATA_DIR / 'features_lstm.csv')
MODEL_DIR = BASE_DIR / 'models'
LOG_DIR = BASE_DIR / 'logs'

# Create directories
os.makedirs(str(MODEL_DIR), exist_ok=True)
os.makedirs(str(LOG_DIR), exist_ok=True)

# Model parameters
SEQUENCE_LENGTH = 5  # Number of past events to consider
BATCH_SIZE = 32
EPOCHS = 50
TRAIN_TEST_SPLIT = 0.8
VALIDATION_SPLIT = 0.1

# Features (31 total - 3 targets = 28 input features)
ORIGINAL_FEATURES = ['time', 'latitude', 'longitude', 'depth', 'mag',
                     'sig', 'mmi', 'cdi', 'felt', 'region_code']
CORE_FEATURES = ['is_aftershock', 'mainshock_mag', 'seismicity_density_100km',
                 'coulomb_stress_proxy', 'regional_b_value']
SEQUENCE_FEATURES = ['sequence_id', 'seq_position', 'is_seq_mainshock',
                     'seq_mainshock_mag', 'seq_length', 'time_since_seq_start_sec']
LSTM_FEATURES = ['time_since_last_event', 'time_since_last_M5',
                 'interval_lag1', 'interval_lag2', 'interval_lag3',
                 'interval_lag4', 'interval_lag5']
TARGET_FEATURES = ['target_time_to_next', 'target_next_mag', 'target_next_mag_binary']

# All features except time (for model input)
INPUT_FEATURES = [f for f in (ORIGINAL_FEATURES + CORE_FEATURES + SEQUENCE_FEATURES + LSTM_FEATURES) if f != 'time']

print(f"✓ Configuration loaded")
print(f"  Data dir: {str(DATA_DIR)}")
print(f"  Features file: {FEATURES_FILE}")
print(f"  Input features: {len(INPUT_FEATURES)}")
