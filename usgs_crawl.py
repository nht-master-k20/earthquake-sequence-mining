#!/usr/bin/env python3
"""
USGS Earthquake Event Crawler
Crawl earthquake data by year from USGS API
Saves JSON files for each event

Usage:
    python usgs_crawl.py 2023
    python usgs_crawl.py 2023 --min-mag 6.5
"""

import os
import sys
import argparse
import time
import requests
import pandas as pd
import glob
import json
from io import StringIO
from datetime import datetime


def get_event_list(year, min_magnitude=None, max_magnitude=None, limit=None):
    """
    Lấy danh sách events từ USGS API

    Args:
        year: Năm cần crawl
        min_magnitude: Độ lớn tối thiểu (None = tất cả)
        max_magnitude: Độ lớn tối đa (None = tất cả)
        limit: Giới hạn số lượng (None = không giới hạn)

    Returns:
        DataFrame: Danh sách events hoặc None
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

    if limit:
        params["limit"] = limit

    try:
        print(f"\nFetching earthquake list for {year}...")
        r = requests.get(url, params=params, timeout=30)

        if r.status_code == 429:
            print(f"⏳ Rate limited! Waiting 15s before retry...")
            time.sleep(15)
            r = requests.get(url, params=params, timeout=30)
            if r.status_code != 200:
                print(f"✗ Failed to fetch event list: HTTP {r.status_code}")
                return None

        if r.status_code != 200:
            print(f"✗ Failed to fetch event list: HTTP {r.status_code}")
            return None

        df = pd.read_csv(StringIO(r.text))
        mag_str = f"M{min_magnitude}+" if min_magnitude is not None else ""
        if max_magnitude is not None:
            mag_str += f"-M{max_magnitude}"
        if not mag_str:
            mag_str = "all magnitudes"

        if len(df) >= 20000:
            print(f"⚠ Found {len(df)} events ({mag_str}) - API limit reached!")
            print(f"  Please use --min-mag or split by month manually")
            return None
        else:
            print(f"✓ Found {len(df)} events ({mag_str})")

        return df

    except Exception as e:
        print(f"✗ Error fetching event list: {e}")
        return None


def get_event_list_by_month(year, min_magnitude=None, max_magnitude=None, limit=None):
    """
    Lấy danh sách events theo tháng (cho năm có >20k events)
    """
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    all_dfs = []

    for month in range(1, 13):
        import calendar
        last_day = calendar.monthrange(year, month)[1]
        starttime = f"{year}-{month:02d}-01"
        endtime = f"{year}-{month:02d}-{last_day}"

        params = {
            "format": "csv",
            "starttime": starttime,
            "endtime": endtime,
            "orderby": "time-asc"
        }

        if min_magnitude is not None:
            params["minmagnitude"] = min_magnitude
        if max_magnitude is not None:
            params["maxmagnitude"] = max_magnitude

        if limit:
            params["limit"] = limit

        try:
            print(f"  Fetching {year}-{month:02d}...", end=" ")
            r = requests.get(url, params=params, timeout=30)

            if r.status_code == 429:
                print(f"⏳ Rate limited! Waiting 10s...", end=" ")
                time.sleep(10)
                r = requests.get(url, params=params, timeout=30)
                if r.status_code != 200:
                    print(f"✗ HTTP {r.status_code}")
                    continue
            elif r.status_code != 200:
                print(f"✗ HTTP {r.status_code}")
                continue

            df = pd.read_csv(StringIO(r.text))
            print(f"✓ {len(df)} events")
            all_dfs.append(df)

        except Exception as e:
            print(f"✗ Error: {e}")

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"\n✓ Total: {len(combined_df)} events")
        return combined_df
    else:
        return None


def crawl_event(event_id, output_dir="data"):
    """
    Crawl chi tiết 1 event và lưu JSON

    Args:
        event_id: ID của event
        output_dir: Thư mục lưu JSON

    Returns:
        bool: True nếu thành công
    """
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    max_retries = 3

    for attempt in range(max_retries):
        try:
            params = {"eventid": event_id, "format": "geojson"}
            r = requests.get(url, params=params, timeout=30)

            if r.status_code == 429:
                if attempt < max_retries - 1:
                    wait_time = 10 + (attempt * 10)
                    time.sleep(wait_time)
                    continue
                else:
                    return False

            if r.status_code != 200:
                return False

            data = r.json()

            # Get feature from response
            if "features" in data and data["features"]:
                feature = data["features"][0]
            elif data.get("type") == "Feature":
                feature = data
            else:
                return False

            props = feature["properties"]
            mag = props.get("mag", 0)
            mag_str = str(mag) if mag is not None else "unknown"
            place = props.get("place", "Unknown")

            # Save JSON with format: event_<mag>_<id>.json
            json_filename = f"event_{mag_str}_{event_id}.json"
            json_path = os.path.join(output_dir, json_filename)

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Print event info
            print(f"  ✓ {event_id} (M{mag_str}): {place}")

            return True

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            continue

    return False


def crawl_year(year, min_mag, max_mag, output_dir, limit=None):
    """
    Crawl dữ liệu cho một năm

    Args:
        year: Năm cần crawl
        min_mag: Độ lớn tối thiểu (None = tất cả)
        max_mag: Độ lớn tối đa (None = tất cả)
        output_dir: Thư mục output gốc
        limit: Giới hạn số lượng events

    Returns:
        int: Số events đã crawl
    """
    mag_str = f"M{min_mag}+" if min_mag is not None else ""
    if max_mag is not None:
        mag_str += f"-M{max_mag}"
    if not mag_str:
        mag_str = "all"

    print(f"\n{'=' * 60}")
    print(f"CRAWLING YEAR: {year}")
    print(f"Magnitude: {mag_str}")
    print("=" * 60)

    # Lấy danh sách events
    df = get_event_list(year, min_mag, max_mag, limit)

    if df is None or len(df) == 0:
        print(f"✗ No events found for {year}!")
        return 0

    # Nếu API limit reached
    if len(df) >= 20000:
        print(f"\n⚠ API limit reached, switching to month-by-month fetching...")
        df = get_event_list_by_month(year, min_mag, max_mag, limit)
        if df is None or len(df) == 0:
            print(f"✗ Failed to fetch events by month!")
            return 0

    year_dir = os.path.join(output_dir, str(year))
    os.makedirs(year_dir, exist_ok=True)
    print(f"Output: {year_dir}/")

    # Crawl từng event
    print(f"\nCrawling {len(df)} events...")
    success_count = 0
    skip_count = 0
    error_count = 0

    for idx, row in df.iterrows():
        event_id = row.get('id')
        if not event_id or pd.isna(event_id):
            continue

        # Check nếu JSON đã tồn tại
        import glob
        existing_files = glob.glob(os.path.join(year_dir, f"event_*_{event_id}.json"))

        if existing_files:
            skip_count += 1
            # Print skip info (giới hạn in để không spam)
            if skip_count <= 10 or skip_count % 500 == 0:
                print(f"  ⊗ {event_id} - skipped (already exists)")
            continue

        # Crawl event
        if crawl_event(event_id, output_dir=year_dir):
            success_count += 1
        else:
            error_count += 1
            print(f"  ✗ {event_id} - failed to crawl")

        # Progress every 50 events
        if (idx + 1) % 50 == 0:
            print(f"  Progress: {idx + 1}/{len(df)} processed, {success_count} new, {skip_count} skipped, {error_count} errors")

        # Delay để tránh rate limit
        time.sleep(1)

    print(f"\n{'=' * 60}")
    print(f"SUMMARY FOR {year}")
    print("=" * 60)
    print(f"Total events: {len(df)}")
    print(f"Successfully crawled: {success_count}")
    print(f"Skipped (already have JSON): {skip_count}")
    print(f"Errors: {error_count}")

    return success_count


def main():
    parser = argparse.ArgumentParser(
        description="USGS Earthquake Event Crawler - JSON only"
    )
    parser.add_argument("year", type=int, nargs='?', help="Năm cần crawl")
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--min-mag", type=float, default=None)
    parser.add_argument("--max-mag", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="data")

    args = parser.parse_args()

    # Xác định danh sách năm
    years = []

    if args.all:
        if not args.start_year:
            print("✗ --all requires --start-year")
            return 1
        end_year = datetime.now().year
        years = list(range(args.start_year, end_year + 1))

    elif args.start_year and args.end_year:
        years = list(range(args.start_year, args.end_year + 1))

    elif args.year:
        years = [args.year]

    elif args.start_year:
        years = list(range(args.start_year, datetime.now().year + 1))

    else:
        parser.print_help()
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    mag_str = f"M{args.min_mag}+" if args.min_mag is not None else ""
    if args.max_mag is not None:
        mag_str += f"-M{args.max_mag}"
    if not mag_str:
        mag_str = "all"

    print("=" * 60)
    print(f"USGS EARTHQUAKE CRAWLER (JSON only)")
    print("=" * 60)
    print(f"Years: {years[0]} - {years[-1]} ({len(years)} years)")
    print(f"Magnitude: {mag_str}")
    print(f"Limit: {args.limit if args.limit else 'No limit'}")
    print(f"Output: {args.output_dir}/{{year}}/event_<mag>_<id>.json")
    print("=" * 60)

    for year in years:
        crawl_year(
            year=year,
            min_mag=args.min_mag,
            max_mag=args.max_mag,
            output_dir=args.output_dir,
            limit=args.limit
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
