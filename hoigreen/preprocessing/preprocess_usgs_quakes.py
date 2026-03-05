import json
import math
import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm


COLUMNS = [
    "id",
    "time",
    "latitude",
    "longitude",
    "depth",
    "mag",
    "magType",
    "mmi",
    "cdi",
    "felt",
    "sig",
    "tsunami",
    "gap",
    "rms",
    "nst",
    "dmin",
]


def _safe_get(d: Dict[str, Any], key: str, default=None):
    v = d.get(key, default)
    return default if v is None else v


def parse_usgs_json(obj: Any) -> List[Dict[str, Any]]:
    """
    Accept:
      - A single Feature (dict with type="Feature")
      - A FeatureCollection (dict with type="FeatureCollection" and features=[...])
      - A list of Feature objects
    Return: list of normalized rows with 16 columns.
    """
    features: List[Dict[str, Any]] = []

    if isinstance(obj, dict):
        t = obj.get("type")
        if t == "FeatureCollection" and isinstance(obj.get("features"), list):
            features = obj["features"]
        elif t == "Feature":
            features = [obj]
        else:
            # Sometimes you might have already extracted a "Feature" shape without "type"
            # Try best-effort: treat as a Feature-like dict if it has "properties" and "geometry"
            if "properties" in obj and "geometry" in obj:
                features = [obj]
            else:
                raise ValueError(
                    "Unsupported JSON dict format (not Feature/FeatureCollection).")
    elif isinstance(obj, list):
        features = obj
    else:
        raise ValueError("Unsupported JSON root type (expect dict or list).")

    rows: List[Dict[str, Any]] = []

    for f in features:
        props = f.get("properties") or {}
        geom = f.get("geometry") or {}
        coords = geom.get("coordinates") or [None, None, None]

        # USGS point is [lon, lat, depth]
        lon = coords[0] if len(coords) > 0 else None
        lat = coords[1] if len(coords) > 1 else None
        depth = coords[2] if len(coords) > 2 else None

        row = {
            "id": f.get("id") or props.get("code") or None,
            "time": props.get("time"),
            "latitude": lat,
            "longitude": lon,
            "depth": depth,
            "mag": props.get("mag"),
            "magType": props.get("magType"),
            "mmi": props.get("mmi"),
            "cdi": props.get("cdi"),
            "felt": props.get("felt"),
            "sig": props.get("sig"),
            "tsunami": props.get("tsunami"),
            "gap": props.get("gap"),
            "rms": props.get("rms"),
            "nst": props.get("nst"),
            "dmin": props.get("dmin"),
        }

        rows.append(row)

    return rows


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    # time: epoch(ms) -> datetime
    df["time"] = pd.to_datetime(
        df["time"], unit="ms", errors="coerce", utc=True)

    # Numeric conversions
    num_float_cols = ["latitude", "longitude", "depth",
                      "mag", "mmi", "cdi", "gap", "rms", "dmin"]
    for c in num_float_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    num_int_cols = ["felt", "sig", "tsunami", "nst"]
    for c in num_int_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    # magType as category-ish (string)
    df["magType"] = df["magType"].astype("string")

    # id as string
    df["id"] = df["id"].astype("string")

    return df


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    # felt: missing -> 0 (no reports)
    if "felt" in df:
        df["felt"] = df["felt"].fillna(0).astype("Int64")

    # tsunami: missing -> 0
    if "tsunami" in df:
        df["tsunami"] = df["tsunami"].fillna(0).astype("Int64")

    # robust median fill for float cols commonly missing
    for c in ["mmi", "cdi", "gap", "rms", "dmin", "depth", "mag", "latitude", "longitude"]:
        if c in df:
            med = df[c].median(skipna=True)
            # If entire column is NA, keep NA (med becomes NaN)
            if med is not None and not (isinstance(med, float) and math.isnan(med)):
                df[c] = df[c].fillna(med)

    # nst: fill median (int)
    if "nst" in df:
        med_nst = pd.to_numeric(df["nst"], errors="coerce").median(skipna=True)
        if med_nst is not None and not (isinstance(med_nst, float) and math.isnan(med_nst)):
            df["nst"] = df["nst"].fillna(int(med_nst)).astype("Int64")

    # sig: fill median (int) — hoặc bạn có thể để NA nếu muốn
    if "sig" in df:
        med_sig = pd.to_numeric(df["sig"], errors="coerce").median(skipna=True)
        if med_sig is not None and not (isinstance(med_sig, float) and math.isnan(med_sig)):
            df["sig"] = df["sig"].fillna(int(med_sig)).astype("Int64")

    # magType: missing -> "unknown"
    df["magType"] = df["magType"].fillna("unknown")

    return df


def filter_invalid(df: pd.DataFrame) -> pd.DataFrame:
    # drop rows without id or time
    df = df.dropna(subset=["id", "time"])

    # basic geo bounds
    df = df[
        (df["latitude"].between(-90, 90, inclusive="both"))
        & (df["longitude"].between(-180, 180, inclusive="both"))
    ]

    # physical sanity
    df = df[df["depth"] >= 0]
    # optionally keep negative mags if you want microquakes
    df = df[df["mag"] >= 0]

    return df


def preprocess(input_path: Path) -> pd.DataFrame:
    with input_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    rows = parse_usgs_json(obj)
    df = pd.DataFrame(rows, columns=COLUMNS)

    df = coerce_types(df)

    # de-dup by id (keep the last occurrence after type coercion)
    df = df.drop_duplicates(subset=["id"], keep="last")

    df = fill_missing(df)
    df = filter_invalid(df)

    # sort by time (ascending)
    df = df.sort_values("time").reset_index(drop=True)

    # Optional: store time as ISO string without timezone confusion
    # For ML pipelines you might prefer epoch ms; adjust as you like:
    df["time"] = df["time"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    return df


def process_year_batch(year_dir: Path, show_progress: bool = True) -> Tuple[pd.DataFrame, int]:
    """Process all JSON files from a single year directory"""
    json_files = sorted(year_dir.glob("*.json"))
    
    all_rows = []
    errors = 0
    
    iterator = tqdm(json_files, desc=f"  Loading {year_dir.name}", unit="file", leave=False) if show_progress else json_files
    
    for json_file in iterator:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                feature = json.load(f)
                rows = parse_usgs_json(feature)
                all_rows.extend(rows)
        except Exception as e:
            errors += 1
    
    if not all_rows:
        return pd.DataFrame(columns=COLUMNS), errors
    
    df = pd.DataFrame(all_rows, columns=COLUMNS)
    return df, errors


def process_batch_mode(data_root: Path, output_path: Path, show_progress: bool = True):
    """Process all years in batch mode with progress tracking"""
    start_time = time.time()
    
    print("=" * 70)
    print("🌍 USGS Earthquake Data Preprocessing Pipeline (Batch Mode)")
    print("=" * 70)
    
    if not data_root.exists():
        raise FileNotFoundError(f"Data directory not found: {data_root}")
    
    # Get all year directories
    year_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
    print(f"\n📂 Found {len(year_dirs)} year directories: {year_dirs[0].name} - {year_dirs[-1].name}")
    
    # Process year by year
    all_dfs = []
    total_files = 0
    total_errors = 0
    
    year_iterator = tqdm(year_dirs, desc="📅 Years", unit="year") if show_progress else year_dirs
    
    for year_dir in year_iterator:
        year_start = time.time()
        
        json_files = list(year_dir.glob("*.json"))
        file_count = len(json_files)
        total_files += file_count
        
        if not show_progress:
            print(f"🔄 Processing {year_dir.name}... ({file_count:,} files)")
        
        df, errors = process_year_batch(year_dir, show_progress)
        total_errors += errors
        
        if errors > 0 and not show_progress:
            print(f"   ⚠️  {errors} files had errors")
        
        if not df.empty:
            all_dfs.append(df)
            if not show_progress:
                year_elapsed = time.time() - year_start
                print(f"   ✅ Loaded {len(df):,} events in {year_elapsed:.1f}s")
    
    if show_progress:
        print()
    
    print(f"📊 Collection: {total_files:,} files processed, {total_errors} errors")
    
    # Combine all years
    print(f"🔄 Combining all years...")
    df_combined = pd.concat(all_dfs, ignore_index=True)
    print(f"✅ Combined: {len(df_combined):,} total events")
    
    # Apply preprocessing steps
    print(f"\n🔄 Applying preprocessing pipeline...")
    
    print(f"   - Converting data types...")
    df_combined = coerce_types(df_combined)
    
    print(f"   - Removing duplicates...")
    initial_count = len(df_combined)
    df_combined = df_combined.drop_duplicates(subset=["id"], keep="last")
    duplicates_removed = initial_count - len(df_combined)
    print(f"     Removed {duplicates_removed:,} duplicates")
    
    print(f"   - Filling missing values...")
    df_combined = fill_missing(df_combined)
    
    print(f"   - Filtering invalid records...")
    initial_count = len(df_combined)
    df_combined = filter_invalid(df_combined)
    invalid_removed = initial_count - len(df_combined)
    print(f"     Removed {invalid_removed:,} invalid records")
    
    print(f"   - Sorting by time...")
    df_combined = df_combined.sort_values("time").reset_index(drop=True)
    
    print(f"   - Converting time to ISO format...")
    df_combined["time"] = df_combined["time"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    
    # Save to output
    print(f"\n💾 Saving to {output_path}...")
    
    # Create parent directory if it doesn't exist
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_combined.to_csv(output_path, index=False)
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("✅ PREPROCESSING COMPLETED")
    print("=" * 70)
    print(f"📊 Results:")
    print(f"   - Input files: {total_files:,}")
    print(f"   - Output rows: {len(df_combined):,}")
    print(f"   - Duplicates removed: {duplicates_removed:,}")
    print(f"   - Invalid removed: {invalid_removed:,}")
    print(f"   - Output file: {output_path}")
    print(f"   - Columns: {len(df_combined.columns)} (expected: 16)")
    print(f"   - Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print()
    print(f"📋 Data Summary:")
    print(f"   - Time range: {df_combined['time'].iloc[0]} to {df_combined['time'].iloc[-1]}")
    print(f"   - Latitude: [{df_combined['latitude'].min():.2f}, {df_combined['latitude'].max():.2f}]")
    print(f"   - Longitude: [{df_combined['longitude'].min():.2f}, {df_combined['longitude'].max():.2f}]")
    print(f"   - Depth: [{df_combined['depth'].min():.2f}, {df_combined['depth'].max():.2f}] km")
    print(f"   - Magnitude: [{df_combined['mag'].min():.2f}, {df_combined['mag'].max():.2f}]")
    print(f"   - Unique magTypes: {df_combined['magType'].nunique()}")
    print(f"   - Tsunami events: {(df_combined['tsunami'] == 1).sum():,}")
    print(f"   - Felt reports: {(df_combined['felt'] > 0).sum():,}")
    print("=" * 70)
    
    return df_combined


def main():
    ap = argparse.ArgumentParser(
        description="Preprocess USGS earthquake JSON into clean CSV (16 columns).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single JSON file mode
  python preprocess_usgs_quakes.py -i data.json -o output.csv
  
  # Batch mode: process entire data/ directory (1980-2026)
  python preprocess_usgs_quakes.py --batch --data-dir data --output earthquake_cleaned.csv
  
  # Batch mode without progress bar
  python preprocess_usgs_quakes.py --batch --data-dir data -o output.csv --no-progress
        """)
    
    ap.add_argument("--input", "-i",
                    help="Path to input JSON (Feature or FeatureCollection). Use with single-file mode.")
    ap.add_argument("--output", "-o", required=True,
                    help="Path to output CSV.")
    ap.add_argument("--batch", action="store_true",
                    help="Batch mode: process all JSON files from data directory.")
    ap.add_argument("--data-dir", default="data",
                    help="Root directory containing year subdirectories (default: data).")
    ap.add_argument("--no-progress", action="store_true",
                    help="Disable progress bars.")
    
    args = ap.parse_args()
    
    output_path = Path(args.output)
    show_progress = not args.no_progress
    
    if args.batch:
        # Batch mode: process entire data directory
        data_root = Path(args.data_dir)
        process_batch_mode(data_root, output_path, show_progress)
    else:
        # Single file mode
        if not args.input:
            ap.error("--input/-i is required when not in batch mode")
        
        input_path = Path(args.input)
        print(f"🔄 Processing single file: {input_path}")
        
        df = preprocess(input_path)
        
        # Create parent directory if it doesn't exist
        if output_path.parent != Path("."):
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        
        print(f"✅ Wrote {len(df):,} rows to {output_path}")


if __name__ == "__main__":
    main()
