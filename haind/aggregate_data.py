#!/usr/bin/env python3
"""
Aggregate Data - Tổng hợp dữ liệu động đất từ thư mục /data
Tạo file CSV để visualize nhanh thay vì load 1.5M JSON files

Usage:
    python aggregate_data.py                    # Aggregate tất cả
    python aggregate_data.py --min-mag 4        # Chỉ M4.0+
    python aggregate_data.py --output data.csv
"""

import os
import sys
import glob
import csv
import json
import argparse
from collections import defaultdict
from datetime import datetime


def extract_mag_from_filename(filepath):
    """
    Extract magnitude từ filename: event_<mag>_<id>.json

    Returns:
        float: magnitude hoặc None nếu không parse được
    """
    basename = os.path.basename(filepath)
    # Format: event_<mag>_<id>.json
    name = basename.replace('event_', '').replace('.json', '')
    parts = name.split('_')
    if len(parts) >= 1:
        try:
            return float(parts[0])
        except ValueError:
            return None
    return None


def get_mag_range(mag):
    """Get magnitude range label"""
    if mag < 2:
        return "M0.0-2.0"
    elif mag < 3:
        return "M2.0-3.0"
    elif mag < 4:
        return "M3.0-4.0"
    elif mag < 5:
        return "M4.0-5.0"
    elif mag < 6:
        return "M5.0-6.0"
    elif mag < 7:
        return "M6.0-7.0"
    elif mag < 8:
        return "M7.0-8.0"
    else:
        return "M8.0+"


def get_depth_range(depth):
    """Get depth range label"""
    if depth < 35:
        return "Shallow (0-35km)"
    elif depth < 70:
        return "Intermediate (35-70km)"
    elif depth < 300:
        return "Deep (70-300km)"
    else:
        return "Very Deep (300km+)"


def aggregate_data(data_dir="../data", min_mag=None, max_mag=None, output_csv="data_summary.csv"):
    """
    Tổng hợp dữ liệu từ tất cả JSON files và lưu CSV

    CSV columns:
    - mag: độ lớn
    - mag_range: khoảng độ lớn
    - depth: độ sâu
    - depth_range: khoảng độ sâu
    - year: năm
    - month: tháng
    - lat: vĩ độ
    - lon: kinh độ
    - place: địa điểm
    """
    # Find all event JSON files
    pattern = os.path.join(data_dir, "**", "event_*.json")
    all_files = glob.glob(pattern, recursive=True)

    print(f"Tìm thấy {len(all_files)} files JSON...")

    rows = []
    stats = {
        "by_year": defaultdict(int),
        "by_mag_range": defaultdict(int),
        "by_depth_range": defaultdict(int),
    }

    processed = 0
    for filepath in all_files:
        # Extract mag from filename
        mag = extract_mag_from_filename(filepath)

        if mag is None:
            continue

        # Filter by magnitude
        if min_mag is not None and mag < min_mag:
            continue
        if max_mag is not None and mag > max_mag:
            continue

        # Read JSON to get more details
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract feature
            if "features" in data and data["features"]:
                feature = data["features"][0]
            elif data.get("type") == "Feature":
                feature = data
            else:
                continue

            props = feature.get("properties", {})
            coords = feature.get("geometry", {}).get("coordinates", [0, 0, 0])

            time_ms = props.get("time")
            year = None
            month = None
            if time_ms:
                try:
                    dt = datetime.fromtimestamp(time_ms / 1000)
                    year = dt.year
                    month = dt.month
                    stats["by_year"][year] += 1
                except:
                    pass

            depth = coords[2] if len(coords) > 2 else 0
            lat = coords[1] if len(coords) > 1 else 0
            lon = coords[0] if len(coords) > 0 else 0
            place = props.get("place", "")

            mag_range = get_mag_range(mag)
            depth_range = get_depth_range(depth)

            # Add row
            rows.append({
                "mag": mag,
                "mag_range": mag_range,
                "depth": round(depth, 2),
                "depth_range": depth_range,
                "year": year,
                "month": month,
                "lat": round(lat, 4),
                "lon": round(lon, 4),
                "place": place
            })

            stats["by_mag_range"][mag_range] += 1
            stats["by_depth_range"][depth_range] += 1

            processed += 1

            # Progress every 10000 files
            if processed % 10000 == 0:
                print(f"  Đã xử lý {processed} files...")

        except Exception as e:
            # Debug first error
            if processed == 0:
                print(f"  First error: {e} on {filepath}")
            continue

    print(f"\nĐã xử lý xong {processed} files")

    # Write CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "mag", "mag_range", "depth", "depth_range",
            "year", "month", "lat", "lon", "place"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Đã lưu {len(rows)} rows vào: {output_csv}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Tổng hợp dữ liệu động đất từ thư mục /data thành CSV"
    )
    parser.add_argument("--data-dir", default="data", help="Thư mục dữ liệu (default: data)")
    parser.add_argument("--output", default="data_summary.csv", help="File CSV output (default: data_summary.csv)")
    parser.add_argument("--min-mag", type=float, default=None, help="Độ lớn tối thiểu")
    parser.add_argument("--max-mag", type=float, default=None, help="Độ lớn tối đa")

    args = parser.parse_args()

    print("=" * 60)
    print("AGGREGATE EARTHQUAKE DATA TO CSV")
    print("=" * 60)
    print(f"Data dir: {args.data_dir}")
    print(f"Output: {args.output}")
    if args.min_mag or args.max_mag:
        mag_filter = f" (M{args.min_mag}+ - M{args.max_mag})" if args.min_mag and args.max_mag else \
                     f" (M{args.min_mag}+)" if args.min_mag else \
                     f" (M{args.max_mag})" if args.max_mag else ""
        print(f"Filter: {mag_filter}")
    print("=" * 60)
    print()

    # Aggregate
    stats = aggregate_data(args.data_dir, args.min_mag, args.max_mag, args.output)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Năm: {len(stats['by_year'])}")
    if stats['by_year']:
        print(f"Năm đầu: {min(stats['by_year'].keys())}")
        print(f"Năm cuối: {max(stats['by_year'].keys())}")
    print()
    print("Theo độ lớn:")
    for mag_range, count in sorted(stats['by_mag_range'].items()):
        print(f"  {mag_range}: {count:,}")
    print()
    print("Theo độ sâu:")
    for depth_range, count in sorted(stats['by_depth_range'].items()):
        print(f"  {depth_range}: {count:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
