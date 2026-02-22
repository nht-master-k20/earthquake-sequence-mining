#!/usr/bin/env python3
"""
Kiểm tra và liệt kê các event bị thiếu file JSON
Gọi trực tiếp USGS API để lấy danh sách events

Usage:
    python check_missing_events.py --all
    python check_missing_events.py --all --autofill 1    # Tự động crawl missing
"""

import os
import sys
import glob
import argparse
import requests
import time
import json
from collections import defaultdict


def get_api_events(year, min_magnitude=None, max_magnitude=None):
    """
    Lấy danh sách event IDs từ USGS API

    Args:
        year: Năm cần kiểm tra
        min_magnitude: Độ lớn tối thiểu
        max_magnitude: Độ lớn tối đa

    Returns:
        set: Set của event IDs từ API
    """
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    params = {
        "format": "csv",
        "starttime": f"{year}-01-01",
        "endtime": f"{year}-12-31",
        "orderby": "time-asc"
    }

    if min_magnitude is not None:
        params["minmagnitude"] = min_magnitude
    if max_magnitude is not None:
        params["maxmagnitude"] = max_magnitude

    try:
        r = requests.get(url, params=params, timeout=30)

        if r.status_code == 429:
            print(f"  ⏳ Rate limited! Waiting 15s...")
            time.sleep(15)
            r = requests.get(url, params=params, timeout=30)

        if r.status_code != 200:
            return set()

        # Parse CSV to extract event IDs
        lines = r.text.split('\n')
        event_ids = set()

        # Find column index of 'id'
        header = lines[0].split(',')
        try:
            id_col = header.index('id')
        except ValueError:
            return set()

        for line in lines[1:]:  # Skip header
            if not line.strip():
                continue
            cols = line.split(',')
            if len(cols) > id_col:
                event_id = cols[id_col].strip()
                if event_id and event_id != '':
                    event_ids.add(event_id)

        return event_ids

    except Exception:
        return set()


def get_json_event_ids(year_dir, min_mag=None, max_mag=None):
    """
    Lấy event IDs từ JSON files trong thư mục
    Filter theo magnitude nếu có
    Unknown mag files: chỉ đếm khi KHÔNG có filter, HOẶC khi min-mag = 0
    """
    json_files = glob.glob(os.path.join(year_dir, "event_*.json"))
    event_ids = set()

    for json_file in json_files:
        basename = os.path.basename(json_file)
        # Format: event_<mag>_<id>.json
        name = basename.replace('event_', '').replace('.json', '')
        parts = name.split('_')
        if len(parts) >= 2:
            # Extract magnitude từ filename
            try:
                mag = float(parts[0])
                # Filter theo magnitude
                if min_mag is not None and mag < min_mag:
                    continue
                if max_mag is not None and mag > max_mag:
                    continue
            except ValueError:
                # Không parse được mag (unknown/None)
                # Nếu có filter (trừ min-mag=0) thì BỎ QUA, không filter thì vẫn giữ
                if min_mag is not None and min_mag != 0:
                    continue
                if max_mag is not None:
                    continue

            event_id = '_'.join(parts[1:])  # Bỏ magnitude, giữ lại ID
            event_ids.add(event_id)

    return event_ids


def crawl_missing_events(year, missing_events, min_mag=None, max_mag=None):
    """
    Crawl các events bị thiếu

    Args:
        year: Năm cần crawl
        missing_events: List event IDs bị thiếu
        min_mag: Minimum magnitude filter
        max_mag: Maximum magnitude filter

    Returns:
        int: Số events crawl thành công
    """
    if not missing_events:
        return 0

    print(f"\n  🔄 Auto-crawling {len(missing_events)} missing events...")

    success_count = 0

    for event_id in missing_events:
        year_dir = os.path.join("data", str(year))
        os.makedirs(year_dir, exist_ok=True)

        # Check if JSON file already exists (by event ID only, ignore magnitude)
        existing_files = glob.glob(os.path.join(year_dir, f"event_*_{event_id}.json"))
        if existing_files:
            continue  # Skip, already have this event

        try:
            url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
            params = {"eventid": event_id, "format": "geojson"}
            r = requests.get(url, params=params, timeout=30)

            if r.status_code == 429:
                print(f"    Rate limited on {event_id}, waiting 10s...")
                time.sleep(10)
                r = requests.get(url, params=params, timeout=30)

            if r.status_code == 200:
                data = r.json()

                # Get feature from response
                if "features" in data and data["features"]:
                    feature = data["features"][0]
                elif data.get("type") == "Feature":
                    feature = data
                else:
                    continue

                props = feature["properties"]
                mag = props.get("mag", 0)
                mag_str = str(mag) if mag is not None else "unknown"

                # Save JSON
                json_filename = f"event_{mag_str}_{event_id}.json"
                json_path = os.path.join(year_dir, json_filename)

                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

                place = props.get('place', 'Unknown')
                print(f"    ✓ {event_id} (M{mag_str}): {place}")

                success_count += 1

                # Delay to avoid rate limit
                time.sleep(1)

        except Exception as e:
            print(f"    ✗ {event_id}: {e}")

    print(f"  ✓ Crawled {success_count} events")
    return success_count


def check_year(year_dir, min_mag=None, max_mag=None, autofill=False):
    """Kiểm tra event thiếu cho 1 năm"""
    year = os.path.basename(year_dir)

    # Lấy event IDs từ JSON files (filter theo magnitude)
    json_event_ids = get_json_event_ids(year_dir, min_mag, max_mag)

    # Lấy event IDs từ USGS API (với filter min/max mag)
    api_event_ids = get_api_events(year, min_mag, max_mag)

    json_count = len(json_event_ids)
    api_count = len(api_event_ids)
    missing_count = api_count - json_count

    # Nếu không filter mag, hoặc min-mag = 0, đếm unknown mag riêng
    if (min_mag is None and max_mag is None) or min_mag == 0:
        unknown_count = count_unknown_mag(year_dir)
        print(f"{year}: api={api_count}, json={json_count} (unknown: {unknown_count}), missing={missing_count}")
    else:
        print(f"{year}: api={api_count}, json={json_count}, missing={missing_count}")

    # Events có trong API nhưng KHÔNG có JSON
    missing = sorted(api_event_ids - json_event_ids)

    # Auto-fill nếu được yêu cầu
    if autofill and missing:
        crawl_missing_events(year, missing, min_mag, max_mag)

    return year, len(missing)


def count_unknown_mag(year_dir):
    """Đếm số files có magnitude unknown"""
    json_files = glob.glob(os.path.join(year_dir, "event_*.json"))
    count = 0
    for json_file in json_files:
        basename = os.path.basename(json_file)
        name = basename.replace('event_', '').replace('.json', '')
        parts = name.split('_')
        if len(parts) >= 1:
            try:
                float(parts[0])
            except ValueError:
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Kiểm tra event thiếu JSON bằng cách gọi USGS API",
        epilog="""
Examples:
  # Check tất cả các năm
  python check_missing_events.py --all

  # Check năm 1975, auto-fill missing
  python check_missing_events.py 1975 --autofill 1

  # Check events M4.0+ (tất cả các năm)
  python check_missing_events.py --all --min-mag 4 --autofill 1

  # Check events M4.0 - M6.0 (tất cả các năm)
  python check_missing_events.py --all --min-mag 4 --max-mag 6 --autofill 1

  # Re-crawl null mag events to check if API has updated magnitude
  python check_missing_events.py 1976 --get-null-mag
        """
    )
    parser.add_argument("year", type=str, nargs='*', help="Năm cần kiểm tra (hoặc dùng --all)")
    parser.add_argument("--all", action="store_true", help="Kiểm tra tất cả các năm")
    parser.add_argument("--min-mag", type=float, default=None, help="Lọc theo độ lớn tối thiểu")
    parser.add_argument("--max-mag", type=float, default=None, help="Lọc theo độ lớn tối đa")
    parser.add_argument("--autofill", type=int, default=0, help="Tự động crawl missing events (0=off, 1=on)")
    parser.add_argument("--output-dir", type=str, default="data")

    args = parser.parse_args()

    # Validate: cần ít nhất một năm hoặc --all
    if not args.year and not args.all:
        parser.error("Vui lòng cung cấp: NĂM hoặc --all")
        return 1

    # Xác định danh sách năm
    if args.all:
        data_dir = args.output_dir
        years_to_check = []
        if os.path.exists(data_dir):
            for item in os.listdir(data_dir):
                item_path = os.path.join(data_dir, item)
                if os.path.isdir(item_path) and item.isdigit():
                    years_to_check.append(item)
        years_to_check.sort()
    else:
        years_to_check = args.year if args.year else []

    print("=" * 60)
    print("CHECKING MISSING JSON FILES")
    print("=" * 60)
    print(f"Years: {len(years_to_check)}")
    if args.min_mag or args.max_mag:
        mag_filter = f" (M{args.min_mag}+-M{args.max_mag})" if args.min_mag and args.max_mag else \
                   f" (M{args.min_mag}+)" if args.min_mag else \
                   f" (M{args.max_mag})" if args.max_mag else ""
        print(f"Filter: {mag_filter}")
    print(f"Auto-fill: {'ON' if args.autofill else 'OFF'}")
    print("=" * 60)

    total_missing = 0
    for year in years_to_check:
        full_path = os.path.join(args.output_dir, year)
        if os.path.isdir(full_path):
            _, missing_count = check_year(
                full_path,
                args.min_mag,
                args.max_mag,
                autofill=(args.autofill == 1)
            )
            total_missing += missing_count

    print("\n" + "=" * 60)
    print(f"TOTAL MISSING: {total_missing}")
    print("=" * 60)


if __name__ == "__main__":
    main()
