#!/usr/bin/env python3
"""
USGS Earthquake Event Crawler
Crawl earthquake data by year from USGS API

Usage:
    python usgs_crawl.py 2023
    python usgs_crawl.py 2023 --min-mag 6.5
"""

import os
import sys
import argparse
import time
import json
import requests
import pandas as pd
from io import StringIO
from datetime import datetime


def crawl_event(event_id, output_dir="data", save_json=True):
    """
    Crawl chi tiết event bằng USGS API

    Args:
        event_id: ID của sự kiện (ví dụ: us6000s4z9)
        output_dir: Thư mục lưu file
        save_json: Có lưu file JSON không

    Returns:
        dict: Thông tin event hoặc None nếu lỗi
    """
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    try:
        params = {"eventid": event_id, "format": "geojson"}
        r = requests.get(url, params=params, timeout=30)

        if r.status_code != 200:
            print(f"  ✗ HTTP {r.status_code}")
            return None

        data = r.json()

        # Xử lý 2 format response khác nhau
        if "features" in data and data["features"]:
            feature = data["features"][0]
        elif data.get("type") == "Feature":
            feature = data
        else:
            print(f"  ✗ Not found: {event_id}")
            return None

        props = feature["properties"]
        coords = feature["geometry"]["coordinates"]

        result = {
            "id": props.get("id"),
            "time": pd.to_datetime(props.get("time", 0), unit="ms"),
            "place": props.get("place"),
            "mag": props.get("mag"),
            "depth": coords[2],
            "lat": coords[1],
            "lon": coords[0],
            "felt": props.get("felt"),
            "cdi": props.get("cdi"),
            "mmi": props.get("mmi"),
            "alert": props.get("alert"),
            "tsunami": props.get("tsunami"),
            "sig": props.get("sig"),
            "url": props.get("url")
        }

        print(f"  ✓ {result['place']} - M{result['mag']}")

        # Lưu JSON nếu được yêu cầu
        # Format: event_<mag>_<id>.json (ví dụ: event_6.3_us70006vkq.json)
        if save_json:
            mag = result.get('mag', 'unknown')
            mag_str = f"{mag:.1f}" if mag is not None else "unknown"
            json_filename = f"event_{mag_str}_{result['id']}.json"
            json_path = os.path.join(output_dir, json_filename)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        return result

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None


def get_event_ids(year, min_magnitude=None, limit=None):
    """
    Lấy danh sách event IDs theo năm

    Args:
        year: Năm cần crawl
        min_magnitude: Độ lớn tối thiểu (None = tất cả)
        limit: Giới hạn số lượng (None = không giới hạn)

    Returns:
        DataFrame: Danh sách events
    """
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    params = {
        "format": "csv",
        "starttime": f"{year}-01-01",
        "endtime": f"{year}-12-31",
        "orderby": "time-asc"
    }

    # Chỉ thêm minmagnitude nếu được chỉ định
    if min_magnitude is not None:
        params["minmagnitude"] = min_magnitude

    if limit:
        params["limit"] = limit

    try:
        print(f"\nFetching earthquake list for {year}...")
        r = requests.get(url, params=params, timeout=30)

        if r.status_code != 200:
            print(f"✗ Failed to fetch event list: HTTP {r.status_code}")
            return None

        df = pd.read_csv(StringIO(r.text))
        mag_str = f"M>={min_magnitude}" if min_magnitude is not None else "all magnitudes"
        print(f"✓ Found {len(df)} events ({mag_str})")
        return df

    except Exception as e:
        print(f"✗ Error fetching event list: {e}")
        return None


def crawl_multiple_events(event_ids, output_dir="data", save_json=True, delay=0.5):
    """
    Crawl nhiều event IDs

    Args:
        event_ids: List các event ID
        output_dir: Thư mục lưu file
        save_json: Có lưu tất cả JSON không
        delay: Delay giữa các requests (giây)

    Returns:
        list: Danh sách kết quả
    """
    results = []
    total = len(event_ids)

    for i, event_id in enumerate(event_ids, 1):
        print(f"[{i}/{total}] {event_id}", end=" ")
        result = crawl_event(event_id, output_dir=output_dir, save_json=save_json)

        if result:
            results.append(result)

        # Delay để tránh rate limit
        if i < total and delay > 0:
            time.sleep(delay)

    return results


def crawl_year(year, min_mag, output_dir, save_json, delay, limit=None):
    """
    Crawl dữ liệu cho một năm

    Args:
        year: Năm cần crawl
        min_mag: Độ lớn tối thiểu (None = tất cả)
        output_dir: Thư mục output gốc
        save_json: Có lưu JSON không
        delay: Delay giữa requests
        limit: Giới hạn số lượng events

    Returns:
        DataFrame: Kết quả hoặc None
    """
    # Tạo chuỗi mô tả min_mag
    mag_str = f"M{min_mag}+" if min_mag is not None else "all"
    print(f"\n{'=' * 60}")
    print(f"CRAWLING YEAR: {year}")
    print(f"Min Magnitude: {mag_str}")
    print("=" * 60)

    # Lấy danh sách event IDs
    df = get_event_ids(year, min_mag, limit)

    if df is None or len(df) == 0:
        print(f"✗ No events found for {year}!")
        return None

    # Chỉ tạo thư mục khi có events
    year_dir = os.path.join(output_dir, str(year))
    os.makedirs(year_dir, exist_ok=True)
    print(f"Output: {year_dir}/")

    print(f"\nEvent IDs:")
    print(df[['time', 'id', 'place', 'mag']].to_string(index=False))

    # Crawl tất cả events
    print(f"\n{'=' * 60}")
    print("CRAWLING EVENTS...")
    print("=" * 60)

    all_results = crawl_multiple_events(
        df['id'].tolist(),
        output_dir=year_dir,  # Lưu vào thư mục năm
        save_json=save_json,
        delay=delay
    )

    # Báo cáo kết quả
    print(f"\n{'=' * 60}")
    print(f"SUMMARY FOR {year}")
    print("=" * 60)
    print(f"Total events found: {len(df)}")
    print(f"Successfully crawled: {len(all_results)}")
    print(f"Failed: {len(df) - len(all_results)}")

    # Lưu CSV riêng cho từng năm
    if all_results:
        results_df = pd.DataFrame(all_results)
        # Tên file CSV: earthquakes_2023_all.csv hoặc earthquakes_2023_M6.0+.csv
        csv_name = f"earthquakes_{year}_{mag_str}.csv"
        csv_path = os.path.join(year_dir, csv_name)
        results_df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"✓ Saved CSV: {csv_path}")

        # Thêm cột year
        results_df['year'] = year
        return results_df

    return None


def main():
    parser = argparse.ArgumentParser(
        description="USGS Earthquake Event Crawler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Crawl 1 năm (tất cả độ lớn)
  python usgs_crawl.py 2023

  # Crawl nhiều năm với độ lớn tối thiểu
  python usgs_crawl.py --start-year 2020 --end-year 2023 --min-mag 5.0

  # Crawl tất cả các năm
  python usgs_crawl.py --all --start-year 2010

  # Tùy chọn khác
  python usgs_crawl.py --start-year 2020 --end-year 2023 --min-mag 6.5 --limit 50
  python usgs_crawl.py 2023 --no-json --delay 1.0
        """
    )

    parser.add_argument("year", type=int, nargs='?', help="Năm cần crawl (để trống nếu dùng --start-year/--end-year)")
    parser.add_argument("--start-year", type=int, default=None,
                        help="Năm bắt đầu (dùng với --end-year)")
    parser.add_argument("--end-year", type=int, default=None,
                        help="Năm kết thúc (dùng với --start-year)")
    parser.add_argument("--all", action="store_true",
                        help="Crawl từ start-year đến năm hiện tại")
    parser.add_argument("--min-mag", type=float, default=None,
                        help="Độ lớn tối thiểu (default: tất cả)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Giới hạn số lượng events mỗi năm (default: không giới hạn)")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Thư mục lưu file (default: data)")
    parser.add_argument("--no-json", action="store_true",
                        help="Không lưu file JSON cho từng event")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay giữa requests (giây, default: 0.5)")

    args = parser.parse_args()

    # Xác định danh sách năm cần crawl
    years = []

    # Mode 1: --all (từ start-year đến năm hiện tại)
    if args.all:
        if not args.start_year:
            print("✗ --all requires --start-year")
            return 1
        end_year = datetime.now().year
        years = list(range(args.start_year, end_year + 1))

    # Mode 2: --start-year và --end-year
    elif args.start_year and args.end_year:
        years = list(range(args.start_year, args.end_year + 1))

    # Mode 3: Chỉ 1 năm (single year mode)
    elif args.year:
        years = [args.year]

    # Mode 4: Chỉ --start-year (từ start-year đến năm hiện tại)
    elif args.start_year:
        years = list(range(args.start_year, datetime.now().year + 1))

    else:
        parser.print_help()
        print("\n✗ Vui lòng指定 năm hoặc khoảng năm!")
        return 1

    # Tạo output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Chuỗi mô tả min_mag
    mag_str = f"M{args.min_mag}+" if args.min_mag is not None else "all"
    mag_display = args.min_mag if args.min_mag is not None else "tất cả"

    print("=" * 60)
    print(f"USGS EARTHQUAKE CRAWLER")
    print("=" * 60)
    print(f"Years: {years[0]} - {years[-1]} ({len(years)} years)")
    print(f"Min Magnitude: {mag_display}")
    print(f"Limit: {args.limit if args.limit else 'No limit'}")
    print(f"Output: {args.output_dir}/{{year}}/")
    print(f"Save JSON: {not args.no_json}")
    print(f"JSON format: event_<mag>_<id>.json")
    print("=" * 60)

    # Crawl từng năm
    all_dataframes = []

    for year in years:
        df = crawl_year(
            year=year,
            min_mag=args.min_mag,
            output_dir=args.output_dir,
            save_json=not args.no_json,
            delay=args.delay,
            limit=args.limit
        )

        if df is not None:
            all_dataframes.append(df)

    # Gộp tất cả và lưu CSV tổng hợp
    print(f"\n{'=' * 60}")
    print(f"FINAL SUMMARY")
    print("=" * 60)

    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)

        # Lưu CSV tổng hợp
        mag_str = f"M{args.min_mag}+" if args.min_mag is not None else "all"
        csv_name = f"earthquakes_{years[0]}-{years[-1]}_{mag_str}.csv"
        csv_path = os.path.join(args.output_dir, csv_name)
        combined_df.to_csv(csv_path, index=False, encoding='utf-8')

        total_records = len(combined_df)
        print(f"✓ Total years crawled: {len(all_dataframes)}")
        print(f"✓ Total records: {total_records}")
        print(f"✓ Combined CSV: {csv_path}")
        print(f"\nPer-directory structure:")
        print(f"  {args.output_dir}/")
        for year in years:
            print(f"    ├── {year}/")
            print(f"    │   ├── earthquakes_{year}_{mag_str}.csv")
            if not args.no_json:
                print(f"    │   ├── event_*.json")
            print(f"    │   └── ...")
        print(f"    └── {csv_name} (combined)")

        return 0
    else:
        print("✗ No data collected!")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
