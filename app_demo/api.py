#!/usr/bin/env python3
"""
FastAPI backend for earthquake data API
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import glob
import os
import json
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

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def get_available_years() -> list[str]:
    """Get list of years that have data"""
    if not os.path.exists(DATA_DIR):
        return []
    years = []
    for item in sorted(os.listdir(DATA_DIR)):
        year_path = os.path.join(DATA_DIR, item)
        if os.path.isdir(year_path) and item.isdigit():
            json_files = glob.glob(os.path.join(year_path, "event_*.json"))
            if json_files:
                years.append(item)
    return years


def read_year_data(year: str) -> list:
    """Read earthquake data for a specific year from JSON files"""
    year_path = os.path.join(DATA_DIR, year)
    json_files = glob.glob(os.path.join(year_path, "event_*.json"))
    if not json_files:
        return []

    events = []
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract feature from GeoJSON
            if "features" in data and data["features"]:
                feature = data["features"][0]
            elif data.get("type") == "Feature":
                feature = data
            else:
                continue

            props = feature["properties"]
            coords = feature.get("geometry", {}).get("coordinates", [0, 0, 0])

            # Parse time (Unix timestamp in milliseconds)
            time_ms = props.get("time", 0)
            try:
                time_obj = datetime.fromtimestamp(time_ms / 1000)
            except:
                continue

            mag = props.get('mag')
            depth = coords[2] if len(coords) > 2 else None
            lat = coords[1] if len(coords) > 1 else None
            lon = coords[0] if len(coords) > 0 else None

            event = {
                'time': time_obj,
                'place': props.get('place') or '--',
                'mag': mag if mag is not None else '--',
                'depth': depth if depth is not None else '--',
                'lat': lat if lat is not None else '--',
                'lon': lon if lon is not None else '--',
            }
            events.append(event)

        except Exception:
            continue

    # Sort by datetime BEFORE formatting
    events.sort(key=lambda x: x['time'])

    # Format time after sorting
    for event in events:
        event['time'] = event['time'].strftime('%d/%m/%Y %H:%M:%S')

    return events


def calculate_stats(data: list) -> dict:
    """Calculate statistics from data"""
    # Filter out '--' and None values, convert to float
    mags = []
    depths = []
    for d in data:
        mag = d.get('mag')
        depth = d.get('depth')
        if isinstance(mag, (int, float)) and mag > 0:
            mags.append(mag)
        if isinstance(depth, (int, float)) and depth > 0:
            depths.append(depth)

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
        # Magnitude ranges (skip '--')
        mag = d['mag']
        if isinstance(mag, (int, float)):
            if mag < 3:
                mag_ranges['0-3'] += 1
            elif mag < 5:
                mag_ranges['3-5'] += 1
            elif mag < 7:
                mag_ranges['5-7'] += 1
            else:
                mag_ranges['7+'] += 1

        # Depth ranges (skip '--')
        depth = d['depth']
        if isinstance(depth, (int, float)):
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
            mag = e['mag']
            depth = e['depth']
            if isinstance(mag, (int, float)) and mag > 0:
                all_mags.append(mag)
            if isinstance(depth, (int, float)) and depth > 0:
                all_depths.append(depth)

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
