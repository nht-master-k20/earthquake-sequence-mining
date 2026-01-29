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
        if save_json:
            json_path = os.path.join(output_dir, f"event_{event_id}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        return result

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None


def get_event_ids(year, min_magnitude=6.0, limit=None):
    """
    Lấy danh sách event IDs theo năm

    Args:
        year: Năm cần crawl
        min_magnitude: Độ lớn tối thiểu
        limit: Giới hạn số lượng (None = không giới hạn)

    Returns:
        DataFrame: Danh sách events
    """
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    params = {
        "format": "csv",
        "starttime": f"{year}-01-01",
        "endtime": f"{year}-12-31",
        "minmagnitude": min_magnitude,
        "orderby": "time-asc"
    }

    if limit:
        params["limit"] = limit

    try:
        print(f"\nFetching earthquake list for {year}...")
        r = requests.get(url, params=params, timeout=30)

        if r.status_code != 200:
            print(f"✗ Failed to fetch event list: HTTP {r.status_code}")
            return None

        df = pd.read_csv(StringIO(r.text))
        print(f"✓ Found {len(df)} events (M>={min_magnitude})")
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


def main():
    parser = argparse.ArgumentParser(
        description="USGS Earthquake Event Crawler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python usgs_crawl.py 2023
  python usgs_crawl.py 2023 --min-mag 6.5
  python usgs_crawl.py 2022 --min-mag 7.0 --limit 50
  python usgs_crawl.py 2023 --no-json --output-dir results
        """
    )

    parser.add_argument("year", type=int, help="Năm cần crawl")
    parser.add_argument("--min-mag", type=float, default=6.0,
                        help="Độ lớn tối thiểu (default: 6.0)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Giới hạn số lượng events (default: không giới hạn)")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Thư mục lưu file (default: data)")
    parser.add_argument("--no-json", action="store_true",
                        help="Không lưu file JSON cho từng event")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay giữa requests (giây, default: 0.5)")

    args = parser.parse_args()

    # Tạo output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"USGS EARTHQUAKE CRAWLER - {args.year}")
    print("=" * 60)
    print(f"Year: {args.year}")
    print(f"Min Magnitude: {args.min_mag}")
    print(f"Limit: {args.limit if args.limit else 'No limit'}")
    print(f"Output: {args.output_dir}/")
    print(f"Save JSON: {not args.no_json}")
    print("=" * 60)

    # Lấy danh sách event IDs
    df = get_event_ids(args.year, args.min_mag, args.limit)

    if df is None or len(df) == 0:
        print("✗ No events found!")
        return 1

    print(f"\nEvent IDs:")
    print(df[['time', 'id', 'place', 'mag']].to_string(index=False))

    # Crawl tất cả events
    print(f"\n{'=' * 60}")
    print("CRAWLING EVENTS...")
    print("=" * 60)

    all_results = crawl_multiple_events(
        df['id'].tolist(),
        output_dir=args.output_dir,
        save_json=not args.no_json,
        delay=args.delay
    )

    # Báo cáo kết quả
    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print("=" * 60)
    print(f"Total events found: {len(df)}")
    print(f"Successfully crawled: {len(all_results)}")
    print(f"Failed: {len(df) - len(all_results)}")

    # Lưu CSV
    if all_results:
        results_df = pd.DataFrame(all_results)

        # Lưu CSV tổng hợp
        csv_name = f"earthquakes_{args.year}_M{args.min_mag}+.csv"
        csv_path = os.path.join(args.output_dir, csv_name)
        results_df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"\n✓ Saved CSV: {csv_path}")
        print(f"  Records: {len(results_df)}")
        print(f"  Columns: {len(results_df.columns)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
