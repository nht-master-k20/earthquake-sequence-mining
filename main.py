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
        python main.py 2023
        python main.py 2023 --min-mag 6.5
    """
    if len(sys.argv) < 2:
        print("Usage: python main.py <year> [options]")
        print("\nExamples:")
        print("  python main.py 2023")
        print("  python main.py 2023 --min-mag 6.5")
        print("  python main.py 2022 --min-mag 7.0 --limit 50")
        print("\nOptions:")
        print("  --min-mag float  Minimum magnitude (default: 6.0)")
        print("  --limit int      Limit number of events")
        print("  --no-json        Don't save JSON files")
        print("  --delay float    Delay between requests (default: 0.5s)")
        return 1

    # Gọi usgs_crawl.py với các tham số
    result = subprocess.run([sys.executable, "usgs_crawl.py"] + sys.argv[1:])

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
