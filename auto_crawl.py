#!/usr/bin/env python3
"""
Auto Crawl - Tự động crawl earthquake data từ USGS API
Kiểm tra và tự động download các event bị thiếu

Usage:
    python auto_crawl.py --all                    # Crawl tất cả các năm
    python auto_crawl.py 1950                     # Crawl năm 1950
    python auto_crawl.py --all --no-autofill      # Chỉ kiểm tra, không crawl
"""

import os
import sys
import glob
import argparse
import requests
import time
import json
import csv
from io import StringIO
from collections import defaultdict

# CẤU HÌNH DELAY (giây)
CRAWL_DELAY = 0.8          # Delay khi crawl mỗi event (giảm để tăng tốc độ)
FETCH_DELAY = 0.5          # Delay khi fetch mỗi magnitude range
RETRY_DELAY_429 = 15       # Delay khi bị rate limit 429 (fetch list)
RETRY_EVENT_429 = 10       # Delay khi bị rate limit 429 (crawl event)


def get_api_events(year, min_magnitude=None, max_magnitude=None):
    """
    Lấy danh sách event IDs từ USGS API
    LUÔN LUÔN split theo magnitude ranges để đảm bảo không bị bỏ sót events

    Args:
        year: Năm cần kiểm tra
        min_magnitude: Độ lớn tối thiểu
        max_magnitude: Độ lớn tối đa

    Returns:
        set: Set của event IDs từ API
    """
    # LUÔN LUÔN split theo magnitude ranges
    return get_api_events_by_mag_ranges(year, min_magnitude, max_magnitude)


def get_api_events_by_mag_ranges(year, min_magnitude=None, max_magnitude=None):
    """
    Lấy event IDs bằng cách split theo magnitude ranges [0,0.5), [0.5,1), [1,1.5), ...
    Chia nhỏ hơn để tránh vượt quá limit 20000 events/request của USGS API
    """
    all_event_ids = set()

    # Xác định các range cần crawl - chia nhỏ thành 0.5
    mag_ranges = []
    for i in range(20):  # 0.0, 0.5, 1.0, 1.5, ... 9.5, 10.0
        mag_ranges.append((i * 0.5, (i + 1) * 0.5))
    mag_ranges.append((10, 11))  # (10,11) để lấy cả giá trị 10.0+

    for low, high in mag_ranges:
        # Apply user filters nếu có
        range_min = low
        range_max = high
        if min_magnitude is not None:
            range_min = max(low, min_magnitude)
        if max_magnitude is not None:
            range_max = min(high, max_magnitude)

        if range_min >= range_max:
            continue

        range_str = f"M{range_min}-M{range_max}" if range_max < 10 else f"M{range_min}-M10"
        print(f"  Fetching {range_str}...", end=" ")

        url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
        params = {
            "format": "csv",
            "starttime": f"{year}-01-01",
            "endtime": f"{year}-12-31",
            "orderby": "time-asc",
            "minmagnitude": range_min,
            "maxmagnitude": range_max
        }

        try:
            r = requests.get(url, params=params, timeout=30)

            if r.status_code == 429:
                time.sleep(RETRY_DELAY_429)
                r = requests.get(url, params=params, timeout=30)

            if r.status_code != 200:
                print(f"✗ Error {r.status_code}")
                time.sleep(FETCH_DELAY)
                continue

            # Parse CSV đúng cách với csv.DictReader
            reader = csv.DictReader(StringIO(r.text))
            range_count = 0
            for row in reader:
                if 'id' in row and row['id']:
                    event_id = row['id'].strip()
                    if event_id:
                        all_event_ids.add(event_id)
                        range_count += 1

            print(f"✓ {range_count} events")
            time.sleep(FETCH_DELAY)

        except Exception as e:
            print(f"✗ Error: {e}")
            time.sleep(FETCH_DELAY)

    return all_event_ids


def get_json_event_ids(year_dir, min_mag=None, max_mag=None, exclude_unknown=True):
    """
    Lấy event IDs từ JSON files trong thư mục
    Filter theo magnitude nếu có
    Unknown mag files: mặc định LOẠI BỎ (exclude_unknown=True)

    Args:
        year_dir: Đường dẫn thư mục năm
        min_mag: Độ lớn tối thiểu
        max_mag: Độ lớn tối đa
        exclude_unknown: Có loại bỏ unknown mag không (default True)

    Returns:
        set: Set của event IDs
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
                # Mặc định LOẠI BỎ unknown mag
                if exclude_unknown:
                    continue
                # Nếu không exclude unknown, check filter
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
    index = 1

    for event_id in missing_events:
        year_dir = os.path.join("data", str(year))
        os.makedirs(year_dir, exist_ok=True)

        # Check if JSON file already exists (by event ID only, ignore magnitude)
        existing_files = glob.glob(os.path.join(year_dir, f"event_*_{event_id}.json"))

        # Skip nếu file đã tồn tại
        if existing_files:
            print(f"    [{index}] ⊗ {event_id} - skipped (already exists)")
            index += 1
            continue

        try:
            url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
            params = {"eventid": event_id, "format": "geojson"}
            r = requests.get(url, params=params, timeout=30)

            if r.status_code == 429:
                print(f"    Rate limited on {event_id}, waiting {RETRY_EVENT_429}s...")
                time.sleep(RETRY_EVENT_429)
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

                # BỎ QUA nếu magnitude là None/unknown
                if mag is None:
                    continue

                mag_str = str(mag)

                # Save JSON
                json_filename = f"event_{mag_str}_{event_id}.json"
                json_path = os.path.join(year_dir, json_filename)

                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

                place = props.get('place', 'Unknown')
                print(f"    [{index}] ✓ {event_id} (M{mag_str}): {place}")

                success_count += 1
                index += 1

                # Delay to avoid rate limit
                time.sleep(CRAWL_DELAY)

        except Exception as e:
            print(f"    [{index}] ✗ {event_id}: {e}")
            index += 1

    print(f"  ✓ Crawled {success_count} events")
    return success_count


def check_year(year_dir, min_mag=None, max_mag=None, autofill=False):
    """Kiểm tra event thiếu cho 1 năm"""
    year = os.path.basename(year_dir)

    # Lấy event IDs từ JSON files (chỉ lấy các file có mag hợp lệ)
    json_event_ids = get_json_event_ids(year_dir, min_mag, max_mag, exclude_unknown=True)

    # Lấy event IDs từ USGS API (với filter min/max mag)
    api_event_ids = get_api_events(year, min_mag, max_mag)

    json_count = len(json_event_ids)
    api_count = len(api_event_ids)
    missing_count = api_count - json_count

    # Hiển thị kết quả
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
        description="Auto crawl earthquake data từ USGS API",
        epilog="""
Examples:
  # Crawl tất cả các năm
  python auto_crawl.py --all

  # Crawl năm 1975
  python auto_crawl.py 1975

  # Chỉ kiểm tra, không crawl
  python auto_crawl.py --all --no-autofill

  # Crawl events M4.0+ (tất cả các năm)
  python auto_crawl.py --all --min-mag 4

  # Crawl events M4.0 - M6.0 (tất cả các năm)
  python auto_crawl.py --all --min-mag 4 --max-mag 6
        """
    )
    parser.add_argument("year", type=str, nargs='*', help="Năm cần crawl (hoặc dùng --all)")
    parser.add_argument("--all", action="store_true", help="Crawl tất cả các năm")
    parser.add_argument("--min-mag", type=float, default=None, help="Lọc theo độ lớn tối thiểu")
    parser.add_argument("--max-mag", type=float, default=None, help="Lọc theo độ lớn tối đa")
    parser.add_argument("--no-autofill", action="store_true", help="Chỉ kiểm tra, không tự động crawl")
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
    print("AUTO CRAWL")
    print("=" * 60)
    print(f"Years: {len(years_to_check)}")
    if args.min_mag or args.max_mag:
        mag_filter = f" (M{args.min_mag}+-M{args.max_mag})" if args.min_mag and args.max_mag else \
                   f" (M{args.min_mag}+)" if args.min_mag else \
                   f" (M{args.max_mag})" if args.max_mag else ""
        print(f"Filter: {mag_filter}")
    autofill_enabled = not args.no_autofill
    print(f"Auto-crawl: {'ON' if autofill_enabled else 'OFF'}")
    print("=" * 60)

    total_missing = 0
    for year in years_to_check:
        full_path = os.path.join(args.output_dir, year)
        if os.path.isdir(full_path):
            _, missing_count = check_year(
                full_path,
                args.min_mag,
                args.max_mag,
                autofill=autofill_enabled
            )
            total_missing += missing_count

    print("\n" + "=" * 60)
    if autofill_enabled and total_missing > 0:
        print(f"TOTAL CRAWLED: {total_missing} events")
    elif total_missing > 0:
        print(f"TOTAL MISSING: {total_missing} events (use --autofill or remove --no-autofill to crawl)")
    else:
        print(f"ALL EVENTS COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
