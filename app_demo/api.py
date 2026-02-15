#!/usr/bin/env python3
"""
FastAPI backend for earthquake data API
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import glob
import os
from pathlib import Path
from datetime import datetime

app = FastAPI(title="Earthquake Sequence Mining API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = "data"


def get_available_years() -> list[str]:
    """Get list of years that have data"""
    if not os.path.exists(DATA_DIR):
        return []
    years = []
    for item in sorted(os.listdir(DATA_DIR)):
        year_path = os.path.join(DATA_DIR, item)
        if os.path.isdir(year_path) and item.isdigit():
            csv_files = glob.glob(os.path.join(year_path, "*.csv"))
            if csv_files:
                years.append(item)
    return years


def read_year_data(year: str) -> list:
    """Read earthquake data for a specific year"""
    year_path = os.path.join(DATA_DIR, year)
    csv_files = glob.glob(os.path.join(year_path, "*.csv"))
    if not csv_files:
        return []
    try:
        df = pd.read_csv(csv_files[0])
        events = []
        for idx, row in df.iterrows():
            try:
                # Parse time - handle both Unix timestamp and ISO format
                time_str = str(row.get('time', ''))
                # Try Unix timestamp (milliseconds)
                try:
                    time_obj = pd.to_datetime(int(time_str), unit='ms', errors='coerce')
                except (ValueError, TypeError):
                    # Try ISO format
                    time_obj = pd.to_datetime(time_str, errors='coerce')

                if pd.isna(time_obj):
                    continue

                event = {
                    'time': time_obj,
                    'place': str(row.get('place', 'Unknown')),
                    'mag': float(row.get('mag', 0)) if pd.notna(row.get('mag', None)) else 0,
                    'depth': float(row.get('depth', 0)) if pd.notna(row.get('depth', None)) else 0,
                    'lat': float(row.get('lat', 0)) if pd.notna(row.get('lat', None)) else 0,
                    'lon': float(row.get('lon', 0)) if pd.notna(row.get('lon', None)) else 0,
                }
                events.append(event)
            except Exception as e:
                print(f"Error parsing row {idx}: {e}")
                continue

        # Sort by datetime BEFORE formatting
        events.sort(key=lambda x: x['time'])

        # Format time after sorting
        for event in events:
            event['time'] = event['time'].strftime('%d/%m/%Y %H:%M:%S')

        return events
    except Exception as e:
        print(f"Error reading {year}: {e}")
        return []


def calculate_stats(data: list) -> dict:
    """Calculate statistics from data"""
    mags = [d['mag'] for d in data if d['mag'] > 0]
    depths = [d['depth'] for d in data if d['depth'] > 0]

    stats = {
        'total_events': len(data),
        'avg_mag': round(sum(mags) / len(mags), 1) if mags else 0,
        'max_mag': round(max(mags), 1) if mags else 0,
        'avg_depth': round(sum(depths) / len(depths), 1) if depths else 0,
    }
    return stats


def calculate_charts(data: list) -> dict:
    """Calculate chart data from data"""
    mag_ranges = {'0-3': 0, '3-5': 0, '5-7': 0, '7+': 0}
    depth_ranges = {'0-50km': 0, '50-100km': 0, '100-300km': 0, '300km+': 0}
    month_counts = [0] * 12

    for d in data:
        # Magnitude ranges
        mag = d['mag']
        if mag < 3:
            mag_ranges['0-3'] += 1
        elif mag < 5:
            mag_ranges['3-5'] += 1
        elif mag < 7:
            mag_ranges['5-7'] += 1
        else:
            mag_ranges['7+'] += 1

        # Depth ranges
        depth = d['depth']
        if depth < 50:
            depth_ranges['0-50km'] += 1
        elif depth < 100:
            depth_ranges['50-100km'] += 1
        elif depth < 300:
            depth_ranges['100-300km'] += 1
        else:
            depth_ranges['300km+'] += 1

        # Month counts
        try:
            time_obj = datetime.strptime(d['time'], '%d/%m/%Y %H:%M:%S')
            month_counts[time_obj.month - 1] += 1
        except:
            pass

    return {
        'mag_ranges': mag_ranges,
        'depth_ranges': depth_ranges,
        'month_counts': month_counts
    }


@app.get("/")
def read_root():
    return {"message": "Earthquake Sequence Mining API", "docs": "/docs"}


@app.get("/api/years")
def get_years():
    """Get all available years"""
    return {"years": get_available_years()}


@app.get("/api/data/{year}")
def get_year_data(year: str):
    """Get earthquake data for a specific year with stats and charts"""
    data = read_year_data(year)

    return {
        "year": year,
        "count": len(data),
        "data": data,
        "stats": calculate_stats(data),
        "charts": calculate_charts(data)
    }


@app.get("/api/stats")
def get_stats():
    """Get overall statistics from all data"""
    all_events = []
    all_mags = []
    all_depths = []

    for year in get_available_years():
        events = read_year_data(year)
        all_events.extend(events)
        for e in events:
            if e['mag'] > 0:
                all_mags.append(e['mag'])
            if e['depth'] > 0:
                all_depths.append(e['depth'])

    stats = {
        'total_events': len(all_events),
        'avg_mag': round(sum(all_mags) / len(all_mags), 1) if all_mags else 0,
        'max_mag': round(max(all_mags), 1) if all_mags else 0,
        'avg_depth': round(sum(all_depths) / len(all_depths), 1) if all_depths else 0,
    }
    return stats


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8386)
