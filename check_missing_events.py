#!/usr/bin/env python3
"""
Kiểm tra và liệt kê các event bị thiếu file JSON
"""

import os
import sys
import glob
import argparse
import pandas as pd


def get_event_id_from_json(json_file):
    """Lấy event_id từ tên file JSON"""
    basename = os.path.basename(json_file)
    # Format: event_<mag>_<id>.json (event_id có thể chứa '_')
    # Bỏ "event_" và ".json", rồi split bởi '_', lấy phần cuối cùng
    name = basename.replace('event_', '').replace('.json', '')
    parts = name.split('_')
    # Phần đầu tiên là magnitude, các phần còn lại ghép lại thành event_id
    if len(parts) >= 2:
        return '_'.join(parts[1:])  # Ghép các phần sau magnitude
    return None


def list_json_files(year_dir):
    """Liệt kê tất cả file JSON trong thư mục năm"""
    pattern = os.path.join(year_dir, "event_*.json")
    return glob.glob(pattern)


def list_csv_events(year_dir):
    """Đọc danh sách event IDs từ CSV file trong thư mục"""
    # Tìm bất kỳ file CSV nào
    csv_files = glob.glob(os.path.join(year_dir, "*.csv"))
    if not csv_files:
        return [], 0

    csv_file = csv_files[0]  # Lấy file CSV đầu tiên
    df = pd.read_csv(csv_file)
    csv_count = len(df)  # Số dòng data (không tính header)

    # Thử các cột event ID trực tiếp
    for col in ['id', 'event_id', 'eventid', 'id ']:
        if col in df.columns:
            return df[col].dropna().astype(str).tolist(), csv_count

    # Nếu không có, thử extract từ URL (cột cuối cùng)
    last_col = df.columns[-1]
    if last_col not in ['id', 'event_id', 'eventid']:
        # Extract event ID từ URL: https://earthquake.usgs.gov/earthquakes/eventpage/iscgem610326299
        event_ids = []
        for url in df[last_col].dropna():
            if isinstance(url, str) and '/eventpage/' in url:
                event_id = url.split('/eventpage/')[-1]
                event_ids.append(event_id)
        return event_ids, csv_count

    return [], csv_count


def check_year(year_dir):
    """Kiểm tra và trả về danh sách event thiếu"""
    year = os.path.basename(year_dir)

    json_files = list_json_files(year_dir)
    json_event_ids = set()

    for json_file in json_files:
        event_id = get_event_id_from_json(json_file)
        if event_id:
            json_event_ids.add(event_id)

    csv_event_ids, csv_count = list_csv_events(year_dir)
    json_count = len(json_event_ids)
    missing = csv_count - json_count
    print(f"{year}: csv={csv_count}, json={json_count}, missing={missing}")

    missing = sorted(set(csv_event_ids) - json_event_ids)
    return year, missing


def main():
    parser = argparse.ArgumentParser(
        description="Kiểm tra và liệt kê các event bị thiếu file JSON"
    )
    parser.add_argument("year", type=str, nargs='*', help="Năm cần kiểm tra")
    parser.add_argument("--all", action="store_true", help="Kiểm tra tất cả các năm")
    parser.add_argument("--output-dir", type=str, default="data")

    args = parser.parse_args()

    # Xác định danh sách năm
    if args.all or not args.year:
        data_dir = args.output_dir
        years_to_check = []
        if os.path.exists(data_dir):
            for item in os.listdir(data_dir):
                item_path = os.path.join(data_dir, item)
                if os.path.isdir(item_path) and item.isdigit():
                    years_to_check.append(item)
        years_to_check.sort()
    else:
        years_to_check = args.year

    # Kiểm tra từng năm và in ra
    for year in years_to_check:
        full_path = os.path.join(args.output_dir, year)
        _, missing = check_year(full_path)
        for eid in missing:
            print(f"{year}: {eid}")


if __name__ == "__main__":
    main()
