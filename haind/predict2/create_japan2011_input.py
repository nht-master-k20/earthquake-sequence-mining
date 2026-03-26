"""
Tạo input cho predict.py - Tìm vùng có mainshock >= M5.0 + foreshocks
Scan toàn bộ data, tìm vùng có chuỗi events với mainshock lớn, lấy foreshocks trước đó

Usage:
    python predict2/create_japan2011_input.py

Output: predict2/input_events.json
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# Filter params
MIN_MAINSHOCK_MAG = 5.0  # Mainshock >= M5.0
DAYS_BEFORE = 30  # Lấy 30 ngày trước mainshock
REGION_RADIUS = 2.0  # 2 độ quanh mainshock
MIN_FORESHOCK_MAG = 4.0  # Foreshocks >= M4.0
MIN_EVENTS = 5  # Tối thiểu 5 foreshocks


def find_mainshock_regions():
    """Tìm tất cả regions có mainshock >= M5.0"""
    print("Scanning for M5.0+ mainshocks...")

    # Get data file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    data_file = os.path.join(base_dir, 'features_lstm.csv')

    df = pd.read_csv(data_file)
    df['time'] = pd.to_datetime(df['time'])

    # Find mainshocks >= M5.0
    mainshocks = df[df['mag'] >= MIN_MAINSHOCK_MAG].copy()
    print(f"Found {len(mainshocks)} events >= M{MIN_MAINSHOCK_MAG}")

    if len(mainshocks) == 0:
        print("ERROR: No mainshock >= M5.0 found in data!")
        return []

    # Group by region (2° radius around each mainshock)
    regions = []
    processed_indices = set()

    for idx, mainshock in mainshocks.iterrows():
        if idx in processed_indices:
            continue

        # Get events in this region (2° radius)
        lat, lon = mainshock['latitude'], mainshock['longitude']
        region_events = df[
            (df['latitude'].between(lat - REGION_RADIUS, lat + REGION_RADIUS)) &
            (df['longitude'].between(lon - REGION_RADIUS, lon + REGION_RADIUS))
        ].copy()

        # Mark as processed
        processed_indices.update(region_events.index.tolist())

        # Count foreshocks (30 ngày before, >= M4.5)
        mainshock_time = mainshock['time']
        start_time = mainshock_time - timedelta(days=DAYS_BEFORE)

        foreshocks = region_events[
            (region_events['time'] >= start_time) &
            (region_events['time'] < mainshock_time) &
            (region_events['mag'] >= MIN_FORESHOCK_MAG)
        ].copy()

        # Sort by time (oldest first = earliest foreshock)
        foreshocks = foreshocks.sort_values('time').reset_index(drop=True)

        if len(foreshocks) >= MIN_EVENTS:
            regions.append({
                'mainshock': mainshock,
                'foreshocks': foreshocks,
                'n_foreshocks': len(foreshocks)
            })

    print(f"Found {len(regions)} regions with >= {MIN_EVENTS} foreshocks")
    return regions


def select_best_region(regions):
    """Chọn region có nhiều foreshocks nhất và mainshock lớn nhất"""
    if not regions:
        return None

    # Sort by: số foreshocks (giảm dần), rồi mainshock mag (giảm dần)
    regions.sort(key=lambda x: (-x['n_foreshocks'], -x['mainshock']['mag']))

    return regions[0]


def dataframe_to_events(mainshock, foreshocks_df):
    """Chuyển dataframe foreshocks sang list events với đầy đủ features

    CHI LAY 50 EVENTS GAN NHAT TRUOC MAINSHOCK
    - Tránh quá nhiều dự đoán thừa
    - Tập trung vào events gần mainshock nhất
    """
    # CHỈ LẤY 50 EVENTS GẦN NHẤT
    MAX_EVENTS = 50
    if len(foreshocks_df) > MAX_EVENTS:
        print(f"WARNING: Co {len(foreshocks_df)} foreshocks, chi lay {MAX_EVENTS} events gan nhat!")
        foreshocks_df = foreshocks_df.tail(MAX_EVENTS).reset_index(drop=True)

    events = []
    mainshock_time = pd.to_datetime(mainshock['time'])

    for idx, row in foreshocks_df.iterrows():
        hours_before = (mainshock_time - row['time']).total_seconds() / 3600

        event = {
            'time': row['time'].strftime('%Y-%m-%dT%H:%M:%S'),
            'mag': row['mag'],
            'depth': row['depth'],
            'latitude': row['latitude'],
            'longitude': row['longitude'],
            'sig': int(row['sig']) if pd.notna(row['sig']) else int(10 ** row['mag']),
            'mmi': int(row['mmi']) if pd.notna(row['mmi']) else int(row['mag'] * 1.5),
            'cdi': int(row['cdi']) if pd.notna(row['cdi']) else int(row['mag'] * 1.2),
            'felt': int(row['felt']) if pd.notna(row['felt']) else int(10 ** row['mag']),
            'region_code': str(row['region_code']) if pd.notna(row['region_code']) else f"{row['latitude']:.0f}_{row['longitude']:.0f}",

            # Core features
            'is_aftershock': int(row['is_aftershock']) if pd.notna(row['is_aftershock']) else 0,
            'mainshock_mag': row['mainshock_mag'] if pd.notna(row['mainshock_mag']) else row['mag'],
            'seismicity_density_100km': row['seismicity_density_100km'] if pd.notna(row['seismicity_density_100km']) else 50,
            'coulomb_stress_proxy': row['coulomb_stress_proxy'] if pd.notna(row['coulomb_stress_proxy']) else 0.5,
            'regional_b_value': row['regional_b_value'] if pd.notna(row['regional_b_value']) else 1.0,

            # Sequence features
            'sequence_id': int(row['sequence_id']) if pd.notna(row['sequence_id']) else 1,
            'seq_position': int(row['seq_position']) if pd.notna(row['seq_position']) else idx + 1,
            'is_seq_mainshock': int(row['is_seq_mainshock']) if pd.notna(row['is_seq_mainshock']) else (1 if row['mag'] >= 6 else 0),
            'seq_mainshock_mag': row['seq_mainshock_mag'] if pd.notna(row['seq_mainshock_mag']) else row['mag'],
            'seq_length': len(foreshocks_df),
            'time_since_seq_start_sec': row['time_since_seq_start_sec'] if pd.notna(row['time_since_seq_start_sec']) else idx * 3600,

            # Temporal features
            'time_since_last_event': row['time_since_last_event'] if pd.notna(row['time_since_last_event']) else 3600,
            'time_since_last_M5': row['time_since_last_M5'] if pd.notna(row['time_since_last_M5']) else 86400,
            'interval_lag1': row['interval_lag1'] if pd.notna(row['interval_lag1']) else 3600,
            'interval_lag2': row['interval_lag2'] if pd.notna(row['interval_lag2']) else 7200,
            'interval_lag3': row['interval_lag3'] if pd.notna(row['interval_lag3']) else 10800,
            'interval_lag4': row['interval_lag4'] if pd.notna(row['interval_lag4']) else 14400,
            'interval_lag5': row['interval_lag5'] if pd.notna(row['interval_lag5']) else 18000,

            # Target (mainshock M5.0)
            'target_quake_in_7days': 1,
            'target_next_mag': mainshock['mag']
        }
        events.append(event)

    return events


def show_summary(mainshock, foreshocks_df):
    """Hiển thị tóm tắt"""
    print(f"\n{'='*90}")
    print(" MAINSHOCK & FORESHOCKS (DATA THẬT) ".center(90))
    print(f"{'='*90}\n")

    print(f"MAINSHOCK:")
    print(f"  Magnitude: M{mainshock['mag']:.1f}")
    print(f"  Time: {mainshock['time']}")
    print(f"  Location: {mainshock['latitude']:.4f}°N, {mainshock['longitude']:.4f}°E")
    print(f"  Depth: {mainshock['depth']:.1f} km")
    print()

    print(f"FORESHOCKS ({len(foreshocks_df)} events, >= M{MIN_FORESHOCK_MAG}):")
    print(f"{'#':<4} {'Mag':<8} {'Depth':<10} {'Truoc':<15} {'Lat':<10} {'Lon':<11}")
    print("-" * 90)

    for i, (_, row) in enumerate(foreshocks_df.iterrows()):
        event_time = pd.to_datetime(row['time'])
        mainshock_time = pd.to_datetime(mainshock['time'])
        hours_before = (mainshock_time - event_time).total_seconds() / 3600
        days = int(hours_before // 24)
        hours = hours_before % 24

        time_str = f"{days}d {hours:.1f}h" if days > 0 else f"{hours:.1f}h"
        print(f"{i+1:<4} {row['mag']:<8.2f} {row['depth']:<10.1f} {time_str:<15} "
              f"{row['latitude']:<10.4f} {row['longitude']:<11.4f}")

    print(f"\n{'='*90}")
    print(f"TARGET: Mainshock M{mainshock['mag']:.1f}")
    print(f"{'='*90}\n")


def save_input_file(events):
    """Lưu events vào input_events.json"""
    output_file = os.path.join(os.path.dirname(__file__), 'input_events.json')

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(events, f, ensure_ascii=False, indent=2)

    print(f"Da luu {len(events)} events vao: {output_file}")
    return output_file


def main():
    """Main function"""
    print(f"\n{'='*90}")
    print(" TIM VUNG CO MAINSHOCK M5.0+ + FORESHOCKS ".center(90))
    print(f"{'='*90}\n")

    # Tìm regions có mainshock M9+
    regions = find_mainshock_regions()

    if not regions:
        print("\nERROR: Khong tim thay region nao co mainshock M5.0!")
        return

    # Chọn region tốt nhất
    best_region = select_best_region(regions)

    mainshock = best_region['mainshock']
    foreshocks_df = best_region['foreshocks']

    # Chuyển sang events
    events = dataframe_to_events(mainshock, foreshocks_df)

    # Hiển thị tóm tắt
    show_summary(mainshock, foreshocks_df)

    # Lưu file
    output_file = save_input_file(events)

    print(f"\nFile da san sang! Chay predict:")
    print(f"  python predict2/predict.py")
    print()


if __name__ == "__main__":
    main()
