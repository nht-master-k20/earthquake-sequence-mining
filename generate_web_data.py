#!/usr/bin/env python3
"""
Generate static data file for web visualization from all CSV files in /data
"""

import os
import json
import glob
import pandas as pd
from datetime import datetime

DATA_DIR = "data"
OUTPUT_FILE = "app_demo/data.js"


def generate_web_data():
    """Generate data.js from all CSV files in data/"""
    all_data = {}
    all_events = []
    all_years = []

    print("=" * 60)
    print("GENERATING WEB DATA")
    print("=" * 60)

    # Find all year directories
    if not os.path.exists(DATA_DIR):
        print(f"✗ {DATA_DIR}/ directory not found!")
        return

    for item in sorted(os.listdir(DATA_DIR)):
        year_path = os.path.join(DATA_DIR, item)
        if os.path.isdir(year_path) and item.isdigit():
            year = item
            all_years.append(year)

            # Find CSV files in this year directory
            csv_files = glob.glob(os.path.join(year_path, "*.csv"))
            if csv_files:
                csv_file = csv_files[0]  # Use first CSV file
                try:
                    df = pd.read_csv(csv_file)

                    # Convert to list of dicts
                    events = []
                    for _, row in df.iterrows():
                        event = {
                            'time': str(row.get('time', '')),
                            'place': str(row.get('place', '')),
                            'mag': float(row.get('mag', 0)) if pd.notna(row.get('mag', None)) else 0,
                            'depth': float(row.get('depth', 0)) if pd.notna(row.get('depth', None)) else 0,
                            'lat': float(row.get('lat', 0)) if pd.notna(row.get('lat', None)) else 0,
                            'lon': float(row.get('lon', 0)) if pd.notna(row.get('lon', None)) else 0,
                        }
                        events.append(event)

                    all_data[year] = events
                    all_events.extend(events)
                    print(f"✓ {year}: {len(events)} events")
                except Exception as e:
                    print(f"✗ {year}: Error reading CSV - {e}")
            else:
                print(f"⚠ {year}: No CSV file found")

    # Calculate overall stats
    total_events = len(all_events)
    all_mags = [e['mag'] for e in all_events if e['mag'] > 0]
    all_depths = [e['depth'] for e in all_events if e['depth'] > 0]

    stats = {
        'total_events': total_events,
        'avg_mag': round(sum(all_mags) / len(all_mags), 1) if all_mags else 0,
        'max_mag': round(max(all_mags), 1) if all_mags else 0,
        'avg_depth': round(sum(all_depths) / len(all_depths), 1) if all_depths else 0,
    }

    # Create JavaScript file
    js_content = f"""// Auto-generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
// Run: python generate_web_data.py

const earthquakeData = {{
    years: {json.dumps(all_years, ensure_ascii=False)},
    data: {json.dumps(all_data, ensure_ascii=False)},
    stats: {json.dumps(stats, ensure_ascii=False)}
}};
"""

    # Write to file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(js_content)

    print("\n" + "=" * 60)
    print(f"✓ Generated: {OUTPUT_FILE}")
    print(f"  Total years: {len(all_years)}")
    print(f"  Total events: {total_events}")
    print("=" * 60)


if __name__ == "__main__":
    generate_web_data()
