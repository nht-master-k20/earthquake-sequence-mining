#!/usr/bin/env python3
"""
Kiểm tra và tự động crawl bổ sung các event bị thiếu
"""

import os
import sys
import glob
import json
import argparse
import pandas as pd
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


def get_event_id_from_json(json_file):
    """Lấy event_id từ tên file JSON"""
    basename = os.path.basename(json_file)
    # Format: event_<mag>_<id>.json
    parts = basename.replace('event_', '').replace('.json', '').split('_')
    if len(parts) >= 3:
        return parts[2]  # Return event_id (phần sau magnitude)
    return None


def list_json_files(year_dir):
    """Liệt kê tất cả file JSON trong thư mục năm"""
    pattern = os.path.join(year_dir, "event_*.json")
    return glob.glob(pattern)


def list_csv_events(csv_file):
    """Đọc danh sách event IDs từ CSV"""
    if not os.path.exists(csv_file):
        return []
    df = pd.read_csv(csv_file)
    # Thử các cột có thể chứa event ID
    for col in ['id', 'event_id', 'eventid', 'id ']:
        if col in df.columns:
            return df[col].dropna().astype(str).tolist()
    # Nếu không có cột nào, thử lấy từ URL
    return []


def crawl_single_event(event_id, output_dir, max_retries=5):
    """Crawl một event với retry"""
    # Gọi retry_failed_events.py
    result = subprocess.run(
        [sys.executable, 'retry_failed_events.py', output_dir, event_id],
        capture_output=True, text=True
    )

    # Parse output
    output = result.stdout
    if "✓" in output or "Success" in output:
        return True
    elif "✗" in output or "Failed" in output:
        return False

    # Trích xuất event_id từ tên file nếu có
    for line in output.split('\n'):
        if 'event_' in line:
            parts = line.strip().replace('event_', '').replace('.json', '').split('_')
            if len(parts) >= 3:
                return parts[2]
    return None


def check_and_supplement_year(year_dir, output_dir="data", min_year=None, max_year=None):
    """
    Kiểm tra và tự động crawl bổ sung các event bị thiếu

    Args:
        year_dir: Thư mục năm cần kiểm tra
        output_dir: Thư mục output gốc
        min_year: Năm nhỏ nhất cần kiểm tra (default: None)
        max_year: Năm lớn nhất cần kiểm tra (default: None)
    """
    year = os.path.basename(year_dir)

    # 1. Liệt kê JSON files
    json_files = list_json_files(year_dir)
    json_count = len(json_files)
    json_event_ids = set()

    for json_file in json_files:
        event_id = get_event_id_from_json(json_file)
        if event_id:
            json_event_ids.add(event_id)

    print(f"\n{'=' * 60}")
    print(f"KIỂM TRA SO KHỐP - {year}")
    print("=" * 60)
    print(f"Thư mục: {year_dir}/")
    print(f"JSON files: {json_count}")
    print(f"Event IDs từ JSON: {len(json_event_ids)}")

    # 2. Liệt kê CSV events
    csv_file = os.path.join(year_dir, f"earthquakes_{year}_all.csv")
    csv_event_ids = list_csv_events(csv_file)
    csv_count = len(csv_event_ids)

    print(f"CSV events: {csv_count}")
    print(f"Event IDs từ CSV: {len(csv_event_ids)}")

    # 3. So sánh và tìm event thiếu
    json_event_ids_list = list(json_event_ids)
    csv_event_ids_list = list(csv_event_ids)

    missing_in_csv = set(csv_event_ids) - set(json_event_ids)
    missing_in_json = set(json_event_ids) - set(csv_event_ids)

    if missing_in_csv:
        print(f"\n✗ Tháng {len(missing_in_csv)} event trong CSV KHÔNG CÓ trong JSON:")
        for eid in sorted(missing_in_csv):
            print(f"  - {eid}")
    if missing_in_json:
        print(f"\n✗ Tháng {len(missing_in_json)} event trong JSON KHÔNG CÓ trong CSV:")
        for eid in sorted(missing_in_json):
            print(f"  - {eid}")

    # 4. Tự động crawl bổ sung
    all_missing = list(missing_in_csv) + list(missing_in_json)

    if all_missing:
        print(f"\n{'=' * 60}")
        print("TỰ ĐỘNG CRAWL MISSING EVENTS...")
        print("=" * 60)

        # Sử dụng ThreadPool để crawl song song
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for event_id in all_missing:
                future = executor.submit(crawl_single_event, event_id, year_dir, 5)
                futures.append(future)

            # Đợi và hiển thị tiến trình
            total = len(all_missing)
            for i, future in enumerate(as_completed(futures), 1):
                if i % 10 == 0 or i == total:
                    print(f"[{i+1}/{total}] {event_id} ...")
                elif future.result():
                    print(f"[{i+1}/{total}] {event_id} ✓")

            # Tổng kết
            success_count = sum(1 for f in futures if f.result())
            print(f"\n{'=' * 60}")
            print("FINAL SUMMARY")
            print("=" * 60)
            print(f"Total: {total}")
            print(f"Success: {success_count}")
            print(f"Failed: {total - success_count}")

            if success_count > 0:
                print(f"\n✓ Đã crawl bổ sung {success_count} events!")

            return 0 if success_count == total else 1


def main():
    parser = argparse.ArgumentParser(
        description="Kiểm tra và tự động crawl bổ sung các event bị thiếu",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Kiểm tra năm 1900
  python check_missing_events.py 1900

  # Kiểm tra nhiều năm
  python check_missing_events.py 1900 1910 1920 1930 1940 1950

  # Kiểm tra tất cả
  python check_missing_events.py --all
        """
    )

    parser.add_argument("year", type=str, help="Năm cần kiểm tra (để kiểm tra riêng)")
    parser.add_argument("--min-year", type=int, help="Năm nhỏ nhất cần kiểm tra (optional)")
    parser.add_argument("--max-year", type=int, help="Năm lớn nhất cần kiểm tra (optional)")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Thư mục chứa data/ (default: data)")
    parser.add_argument("--no-crawl", action="store_true",
                        help="Chỉ kiểm tra, không crawl bổ sung")

    args = parser.parse_args()

    # Xác định năm cần kiểm tra
    if args.min_year is not None and args.max_year is not None:
        if args.min_year > args.max_year:
            print("✗ --min-year phải nhỏ hơn --max-year!")
            return 1

    # Xác định danh sách năm
    if args.year:
        years_to_check = [args.year]
    else:
        # Tìm tất cả thư mục năm trong data/
        data_dir = args.output_dir
        if os.path.exists(data_dir):
            years_to_check = []
            for item in os.listdir(data_dir):
                item_path = os.path.join(data_dir, item)
                if os.path.isdir(item_path) and item.isdigit():
                    years_to_check.append(os.path.basename(item_path))

        if not years_to_check:
            print("✗ Không tìm thấy thư mục năm nào trong data/")
            return 1

        # Sắp xếp
        years_to_check.sort()

        # Giới hạn phạm vi nếu có min/max_year
        if args.min_year is not None:
            years_to_check = [y for y in years_to_check if y >= args.min_year]
        if args.max_year is not None:
            years_to_check = [y for y in years_to_check if y <= args.max_year]

    print("=" * 60)
    print("KIỂM TRA SO KHỐP")
    print("=" * 60)
    print(f"Năm kiểm tra: {years_to_check[0]} - {years_to_check[-1]}")
    print(f"Thư mục data: {data_dir}/")
    print(f"Output directory: {args.output_dir}/")

    return_code = 0

    for year_dir in years_to_check:
        year = os.path.basename(year_dir)
        full_path = os.path.join(args.output_dir, year)

        print(f"\n{'=' * 60}")
        print(f"KIỂM TRA SO KHỐP - {year}")
        print("=" * 60)

        # Kiểm tra và crawl
        if args.no_crawl:
            code = check_and_supplement_year(full_path, args.output_dir,
                                         args.min_year, args.max_year)
        print(f"\n✓ Đã kiểm tra xong (không crawl)")
        else:
            code = check_and_supplement_year(full_path, args.output_dir,
                                        args.min_year, args.max_year)

        if code != 0:
            return_code = 1

    sys.exit(return_code)


if __name__ == "__main__":
    main()
