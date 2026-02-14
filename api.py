#!/usr/bin/env python3
"""
Flask API to serve earthquake CSV data for visualization
"""

import os
import json
import pandas as pd
from flask import Flask, jsonify, send_from_directory

app = Flask(__name__, static_folder='app_demo')

# Get the parent directory (earthquake-sequence-mining)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')


@app.route('/')
def index():
    """Serve the demo homepage"""
    return send_from_directory('app_demo')


@app.route('/api/years')
def get_years():
    """Get list of years that have data"""
    years = []
    if os.path.exists(DATA_DIR):
        for item in os.listdir(DATA_DIR):
            item_path = os.path.join(DATA_DIR, item)
            if os.path.isdir(item_path) and item.isdigit():
                years.append(item)
    return jsonify(sorted(years))


@app.route('/api/events/<year>')
def get_events(year):
    """Get events for a specific year"""
    csv_files = []
    year_dir = os.path.join(DATA_DIR, year)

    if not os.path.exists(year_dir):
        return jsonify([])

    # Find all CSV files in the year directory
    for file in os.listdir(year_dir):
        if file.endswith('.csv'):
            csv_files.append(os.path.join(year_dir, file))

    if not csv_files:
        return jsonify([])

    # Read the first (or only) CSV file
    csv_file = csv_files[0]
    try:
        df = pd.read_csv(csv_file)

        # Convert to list of dicts
        events = []
        for _, row in df.iterrows():
            event = {
                'time': str(row.get('time', '')),
                'place': str(row.get('place', '')),
                'mag': float(row.get('mag', 0)) if pd.notna(row.get('mag')) else 0,
                'depth': float(row.get('depth', 0)) if pd.notna(row.get('depth')) else 0,
                'lat': float(row.get('lat', 0)) if pd.notna(row.get('lat')) else 0,
                'lon': float(row.get('lon', 0)) if pd.notna(row.get('lon')) else 0,
            }
            events.append(event)

        return jsonify(events)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats')
def get_stats():
    """Get overall statistics across all years"""
    stats = {}
    total_events = 0
    all_mags = []
    all_depths = []

    if os.path.exists(DATA_DIR):
        for year in sorted(os.listdir(DATA_DIR)):
            year_dir = os.path.join(DATA_DIR, year)
            if os.path.isdir(year_dir) and year.isdigit():
                # Find CSV file
                csv_file = os.path.join(year_dir, f"earthquakes_{year}_all.csv")
                if not os.path.exists(csv_file):
                    # Try other CSV files
                    for file in os.listdir(year_dir):
                        if file.endswith('.csv'):
                            csv_file = os.path.join(year_dir, file)
                            break

                if os.path.exists(csv_file):
                    try:
                        df = pd.read_csv(csv_file)
                        stats[year] = {
                            'count': len(df),
                            'avg_mag': float(df['mag'].mean()) if 'mag' in df.columns else 0,
                            'max_mag': float(df['mag'].max()) if 'mag' in df.columns else 0,
                            'avg_depth': float(df['depth'].mean()) if 'depth' in df.columns else 0,
                        }
                        total_events += len(df)
                        if 'mag' in df.columns:
                            all_mags.extend(df['mag'].dropna().tolist())
                        if 'depth' in df.columns:
                            all_depths.extend(df['depth'].dropna().tolist())
                    except Exception:
                        pass

    if all_mags:
        stats['overall'] = {
            'total_events': total_events,
            'avg_mag': round(sum(all_mags) / len(all_mags), 1),
            'max_mag': round(max(all_mags), 1),
            'avg_depth': round(sum(all_depths) / len(all_depths), 1),
        }

    return jsonify(stats)


if __name__ == '__main__':
    print("=" * 60)
    print("Earthquake Visualization API")
    print("=" * 60)
    print("Starting Flask server...")
    print("Open: http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    app.run(host='0.0.0.0', port=5000, debug=True)
