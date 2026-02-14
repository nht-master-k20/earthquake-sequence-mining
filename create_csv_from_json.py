#!/usr/bin/env python3
"""
Tạo CSV file từ các JSON files đã có (dùng khi crawler bị interrupt và thiếu CSV)
"""

import os
import json
import glob
import argparse
import pandas as pd


def create_csv_from_json(year_dir):
    """
    Tạo CSV từ JSON files trong thư mục năm

    Args:
        year_dir: Thư mục năm (ví dụ: "data/1974")

    Returns:
        int: Số records đã tạo, hoặc 0 nếu lỗi
    """
    json_files = glob.glob(os.path.join(year_dir, "event_*.json"))

    if not json_files:
        print(f"✗ No JSON files found in {year_dir}/")
        return 0

    print(f"Found {len(json_files)} JSON files")

    # Extract data from JSON files
    results = []
    for jf in json_files:
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle 2 formats
            if "features" in data and data["features"]:
                feature = data["features"][0]
            elif data.get("type") == "Feature":
                feature = data
            else:
                continue

            props = feature["properties"]
            coords = feature["geometry"]["coordinates"]

            result = {
                "id": props.get("id"),
                "time": props.get("time"),
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
            results.append(result)
        except Exception as e:
            print(f"Error reading {jf}: {e}")

    print(f"Successfully parsed {len(results)}/{len(json_files)} JSON files")

    # Create CSV
    if results:
        df = pd.DataFrame(results)
        year = os.path.basename(year_dir)
        csv_path = os.path.join(year_dir, f"earthquakes_{year}_all.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"✓ Created CSV: {csv_path}")
        print(f"Total records: {len(df)}")
        return len(df)
    else:
        print("✗ No data to save")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Tạo CSV từ JSON files (dùng khi crawler bị interrupt)"
    )
    parser.add_argument("year_dir", type=str, help="Thư mục năm (ví dụ: data/1974)")
    parser.add_argument("--all", action="store_true", help="Tạo CSV cho tất cả các năm thiếu")

    args = parser.parse_args()

    if args.all:
        # Tìm tất cả các năm thiếu CSV
        data_dir = os.path.dirname(args.year_dir) if os.path.dirname(args.year_dir) else "."
        if os.path.basename(args.year_dir).isdigit():
            data_dir = os.path.dirname(os.path.dirname(args.year_dir))

        total = 0
        for item in os.listdir(data_dir):
            year_path = os.path.join(data_dir, item)
            if os.path.isdir(year_path) and item.isdigit():
                csv_files = glob.glob(os.path.join(year_path, "*.csv"))
                if not csv_files:
                    # Thử tạo CSV
                    print(f"\n{'=' * 60}")
                    print(f"Processing {item}/")
                    print("=" * 60)
                    count = create_csv_from_json(year_path)
                    if count > 0:
                        total += count

        print(f"\n{'=' * 60}")
        print(f"TOTAL: Created {total} records across all missing CSV files")
        print("=" * 60)
    else:
        create_csv_from_json(args.year_dir)


if __name__ == "__main__":
    main()
