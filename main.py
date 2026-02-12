#!/usr/bin/env python3
"""
Main entry point for USGS Earthquake Crawler
"""

import sys
import subprocess

def main():
    """
    Chạy usgs_crawl.py với tham số năm

    Usage:
        # Crawl 1 năm
        python main.py 2023

        # Crawl nhiều năm
        python main.py --start-year 2020 --end-year 2023

        # Crawl tất cả các năm
        python main.py --all --start-year 2010
    """
    if len(sys.argv) < 2:
        print("USGS Earthquake Crawler - Multi-Year Support")
        print("\nUsage:")
        print("  # Crawl 1 năm (tất cả độ lớn):")
        print("  python main.py 2023")
        print("\n  # Crawl nhiều năm với độ lớn tối thiểu:")
        print("  python main.py --start-year 2020 --end-year 2023 --min-mag 5.0")
        print("\n  # Crawl tất cả các năm (từ start-year đến hiện tại):")
        print("  python main.py --all --start-year 2010")
        print("\nOptions:")
        print("  --min-mag float  Minimum magnitude (default: tất cả)")
        print("  --limit int      Limit number of events per year")
        print("\nDefault settings (không thể thay đổi):")
        print("  --save-json      Always save JSON files")
        print("  --delay          0.5s between requests")
        print("  --max-retries    3 retries for network errors")
        print("\nJSON format: event_<mag>_<id>.json (e.g., event_6.3_us70006vkq.json)")
        return 1

    # Gọi usgs_crawl.py với các tham số
    result = subprocess.run([sys.executable, "usgs_crawl.py"] + sys.argv[1:])

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
