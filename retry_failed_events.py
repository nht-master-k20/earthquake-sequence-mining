#!/usr/bin/env python3
"""
Crawl lại các event bị fail từ danh sách ID
"""

import sys
import time
import json
import argparse
import requests
import pandas as pd
from io import StringIO
from requests.exceptions import RequestException, ConnectionError, Timeout


def crawl_event(event_id, output_dir="data", max_retries=5):
    """
    Crawl chi tiết event bằng USGS API với retry

    Args:
        event_id: ID của sự kiện
        output_dir: Thư mục lưu file
        max_retries: Số lần retry (default: 5)

    Returns:
        dict: Thông tin event hoặc None nếu lỗi
    """
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    # Retry logic với exponential backoff
    for attempt in range(max_retries):
        try:
            params = {"eventid": event_id, "format": "geojson"}
            r = requests.get(url, params=params, timeout=30)

            if r.status_code != 200:
                print(f"  ✗ HTTP {r.status_code}")
                return None

            data = r.json()

            # Xử lý 2 format response
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

            # Lưu JSON - dùng event_id gốc vì result['id'] có thể None
            mag = result.get('mag', 'unknown')
            mag_str = f"{mag:.1f}" if mag is not None else "unknown"
            json_filename = f"event_{mag_str}_{event_id}.json"  # Dùng event_id tham số
            json_path = os.path.join(output_dir, json_filename)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            return result

        except (ConnectionError, Timeout) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"  ⏳ Retry {attempt + 1}/{max_retries} after {wait_time}s")
                time.sleep(wait_time)
            else:
                print(f"  ✗ Failed after {max_retries} retries")
                return None

        except RequestException as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"  ⏳ Retry {attempt + 1}/{max_retries} after {wait_time}s")
                time.sleep(wait_time)
            else:
                print(f"  ✗ Failed after {max_retries} retries: {e}")
                return None

        except Exception as e:
            print(f"  ✗ Error: {e}")
            return None


def main():
    if len(sys.argv) < 3:
        print("Usage: python retry_failed_events.py <year_dir> <event_id1> [event_id2] ...")
        print("\nExample:")
        print("  python retry_failed_events.py 1900 iscgem811607 uw10835138 iscgem811616")
        return 1

    year_dir = sys.argv[1]
    event_ids = sys.argv[2:]

    print("=" * 60)
    print(f"RETRY FAILED EVENTS")
    print("=" * 60)
    print(f"Output directory: {year_dir}/")
    print(f"Events to retry: {len(event_ids)}")
    print("=" * 60)

    results = []
    for i, event_id in enumerate(event_ids, 1):
        print(f"[{i}/{len(event_ids)}] {event_id}", end=" ")
        result = crawl_event(event_id, output_dir=year_dir, max_retries=5)
        if result:
            results.append(result)
        # Delay giữa các requests
        if i < len(event_ids):
            time.sleep(1)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"Total: {len(event_ids)}")
    print(f"Success: {len(results)}")
    print(f"Failed: {len(event_ids) - len(results)}")


if __name__ == "__main__":
    import os
    main()
