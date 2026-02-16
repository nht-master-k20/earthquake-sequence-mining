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
from requests.exceptions import RequestException, ConnectionError, Timeout


def crawl_event(event_id, output_dir="data"):
    """
    Crawl chi tiết event bằng USGS API

    Args:
        event_id: ID của sự kiện (ví dụ: us6000s4z9)
        output_dir: Thư mục lưu file

    Returns:
        dict: Thông tin event hoặc None nếu lỗi
    """
    # Default values
    save_json = True
    max_retries = 3
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    # Retry logic cho lỗi mạng
    for attempt in range(max_retries):
        try:
            params = {"eventid": event_id, "format": "geojson"}
            r = requests.get(url, params=params, timeout=30)

            if r.status_code == 429:
                # Rate limit - wait longer and retry
                if attempt < max_retries - 1:
                    wait_time = 10 + (attempt * 10)  # 10s, 20s, 30s
                    print(f"  ⏳ Rate limited! Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"  ✗ Rate limited - max retries reached")
                    return None
            elif r.status_code != 200:
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
                # Dùng event_id tham số thay vì result['id'] có thể None
                event_id_for_filename = result.get('id') or event_id
                json_filename = f"event_{mag_str}_{event_id_for_filename}.json"
                json_path = os.path.join(output_dir, json_filename)
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

            return result

        except (ConnectionError, Timeout) as e:
            # Lỗi mạng/timeout - retry với exponential backoff
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 2s, 4s, 8s...
                print(f"  ⏳ Retry {attempt + 1}/{max_retries} after {wait_time}s (network error)")
                time.sleep(wait_time)
            else:
                print(f"  ✗ Failed after {max_retries} retries: {e}")
                return None

        except RequestException as e:
            # Lỗi request khác (DNS, etc.)
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"  ⏳ Retry {attempt + 1}/{max_retries} after {wait_time}s (request error)")
                time.sleep(wait_time)
            else:
                print(f"  ✗ Failed after {max_retries} retries: {e}")
                return None

        except Exception as e:
            # Lỗi khác không retry
            print(f"  ✗ Error: {e}")
            return None


def get_event_ids(year, min_magnitude=None, max_magnitude=None, limit=None):
    """
    Lấy danh sách event IDs theo năm

    Args:
        year: Năm cần crawl
        min_magnitude: Độ lớn tối thiểu (None = tất cả)
        max_magnitude: Độ lớn tối đa (None = tất cả)
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

    # Chỉ thêm min/max magnitude nếu được chỉ định
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
            # Rate limit - wait and retry
            print(f"⏳ Rate limited! Waiting 15s before retry...")
            time.sleep(15)
            # Retry once
            r = requests.get(url, params=params, timeout=30)
            if r.status_code != 200:
                print(f"✗ Failed to fetch event list: HTTP {r.status_code}")
                return None
        elif r.status_code != 200:
            print(f"✗ Failed to fetch event list: HTTP {r.status_code}")
            return None

        df = pd.read_csv(StringIO(r.text))
        mag_str = f"M{min_magnitude}+" if min_magnitude is not None else ""
        if max_magnitude is not None:
            mag_str += f"-M{max_magnitude}"
        if not mag_str:
            mag_str = "all magnitudes"

        # Check nếu API limit reached (USGS limit = 20000)
        if len(df) >= 20000:
            print(f"⚠ Found {len(df)} events ({mag_str}) - API limit reached!")
            print(f"  Tip: Use --min-mag/--max-mag to filter by magnitude, or we'll split by month...")
            # Note: Could implement pagination with offset, but splitting by month is more reliable
        else:
            print(f"✓ Found {len(df)} events ({mag_str})")

        return df

    except Exception as e:
        print(f"✗ Error fetching event list: {e}")
        return None


def get_event_ids_by_month(year, min_magnitude=None, max_magnitude=None, limit=None):
    """
    Lấy danh sách event IDs theo năm, chia nhỏ theo tháng (cho năm có >20k events)

    Args:
        year: Năm cần crawl
        min_magnitude: Độ lớn tối thiểu (None = tất cả)
        max_magnitude: Độ lớn tối đa (None = tất cả)
        limit: Giới hạn số lượng (None = không giới hạn)

    Returns:
        DataFrame: Danh sách events
    """
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    all_dfs = []

    # Chia theo tháng
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
                # Rate limit - wait and retry
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
        mag_str = f"M{min_magnitude}+" if min_magnitude is not None else ""
        if max_magnitude is not None:
            mag_str += f"-M{max_magnitude}"
        if not mag_str:
            mag_str = "all magnitudes"
        print(f"\n✓ Total: {len(combined_df)} events ({mag_str})")
        return combined_df
    else:
        return None


def crawl_multiple_events(event_ids, output_dir="data"):
    """
    Crawl nhiều event IDs

    Args:
        event_ids: List các event ID
        output_dir: Thư mục lưu file

    Returns:
        list: Danh sách kết quả
    """
    # Default values
    save_json = True
    delay = 1.0  # Increased to avoid rate limiting
    max_retries = 3
    results = []
    total = len(event_ids)
    skipped = 0

    for i, event_id in enumerate(event_ids, 1):
        # Check nếu JSON file đã tồn tại
        import glob
        existing_files = glob.glob(os.path.join(output_dir, f"event_*_{event_id}.json"))
        if existing_files:
            print(f"[{i}/{total}] {event_id} - skip (exists)")
            skipped += 1
            continue

        print(f"[{i}/{total}] {event_id}", end=" ")
        result = crawl_event(event_id, output_dir=output_dir)

        if result:
            results.append(result)

        # Delay để tránh rate limit
        if i < total and delay > 0:
            time.sleep(delay)

    if skipped > 0:
        print(f"\n✓ Skipped {skipped}/{total} events (already have JSON)")

    return results


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
        DataFrame: Kết quả hoặc None
    """
    # Default values (không còn tham số CLI)
    save_json = True
    delay = 1.0  # Increased to avoid rate limiting
    max_retries = 3
    # Tạo chuỗi mô tả magnitude range
    mag_str = f"M{min_mag}+" if min_mag is not None else ""
    if max_mag is not None:
        mag_str += f"-M{max_mag}"
    if not mag_str:
        mag_str = "all"
    print(f"\n{'=' * 60}")
    print(f"CRAWLING YEAR: {year}")
    print(f"Magnitude: {mag_str}")
    print("=" * 60)

    # Lấy danh sách event IDs
    df = get_event_ids(year, min_mag, max_mag, limit)

    if df is None or len(df) == 0:
        print(f"✗ No events found for {year}!")
        return None

    # Nếu API limit reached (>=20000), dùng method chia theo tháng
    if len(df) >= 20000:
        print(f"\n⚠ API limit reached, switching to month-by-month fetching...")
        df = get_event_ids_by_month(year, min_mag, max_mag, limit)
        if df is None or len(df) == 0:
            print(f"✗ Failed to fetch events by month!")
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
        output_dir=year_dir  # Lưu vào thư mục năm
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

  # Crawl với khoảng độ lớn (ví dụ: chỉ M 5.0 - 6.5)
  python usgs_crawl.py --start-year 2020 --end-year 2023 --min-mag 5.0 --max-mag 6.5

  # Crawl tất cả các năm
  python usgs_crawl.py --all --start-year 2010

  # Tùy chọn khác
  python usgs_crawl.py --start-year 2020 --end-year 2023 --min-mag 6.5 --max-mag 7.0 --limit 50
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
    parser.add_argument("--max-mag", type=float, default=None,
                        help="Độ lớn tối đa (default: tất cả)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Giới hạn số lượng events mỗi năm (default: không giới hạn)")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Thư mục lưu file (default: data)")

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

    # Chuỗi mô tả magnitude range
    mag_str = f"M{args.min_mag}+" if args.min_mag is not None else ""
    if args.max_mag is not None:
        mag_str += f"-M{args.max_mag}"
    if not mag_str:
        mag_str = "all"
    mag_display = mag_str if mag_str != "all" else "tất cả"

    print("=" * 60)
    print(f"USGS EARTHQUAKE CRAWLER")
    print("=" * 60)
    print(f"Years: {years[0]} - {years[-1]} ({len(years)} years)")
    print(f"Magnitude: {mag_display}")
    print(f"Limit: {args.limit if args.limit else 'No limit'}")
    print(f"Output: {args.output_dir}/{{year}}/")
    print(f"JSON format: event_<mag>_<id>.json")
    print("=" * 60)

    # Crawl từng năm
    all_dataframes = []

    for year in years:
        df = crawl_year(
            year=year,
            min_mag=args.min_mag,
            max_mag=args.max_mag,
            output_dir=args.output_dir,
            limit=args.limit
        )

        if df is not None:
            all_dataframes.append(df)

    # Gộp tất cả và lưu CSV tổng hợp
    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)

        # Lưu CSV tổng hợp
        mag_str = f"M{args.min_mag}+" if args.min_mag is not None else "all"
        csv_name = f"earthquakes_{years[0]}-{years[-1]}_{mag_str}.csv"
        csv_path = os.path.join(args.output_dir, csv_name)
        combined_df.to_csv(csv_path, index=False, encoding='utf-8')

        print(f"✓ Combined CSV: {csv_path}")
        print(f"✓ Total records: {len(combined_df)}")

        return 0
    else:
        print("✗ No data collected!")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
