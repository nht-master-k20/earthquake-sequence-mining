import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add region_code to an earthquake CSV using the project's grid logic."
    )
    parser.add_argument("--input-csv", type=Path,
                        default=Path("data/dongdat_full.csv"))
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/dongdat_full_with_region_code.csv"),
    )
    parser.add_argument("--region-grid-size", type=float, default=2.5)
    parser.add_argument(
        "--keep-region-metadata",
        action="store_true",
        help="Also write lat_cell, lon_cell, region_lat_center, region_lon_center.",
    )
    return parser.parse_args()


def load_dataset(input_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv, low_memory=False)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    required_columns = {"latitude", "longitude"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"Input CSV missing required columns: {missing_columns}")

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    return df


def add_region_code(
    df: pd.DataFrame,
    region_grid_size: float,
    keep_region_metadata: bool,
) -> pd.DataFrame:
    out = df.copy()
    valid_mask = (
        out["latitude"].notna()
        & out["longitude"].notna()
        & out["latitude"].between(-90.0, 90.0, inclusive="both")
        & out["longitude"].between(-180.0, 180.0, inclusive="both")
    )

    out["region_code"] = pd.NA

    lat_cells = np.floor(
        (out.loc[valid_mask, "latitude"] + 90.0) / region_grid_size).astype(int)
    lon_cells = np.floor(
        (out.loc[valid_mask, "longitude"] + 180.0) / region_grid_size).astype(int)

    out.loc[valid_mask, "region_code"] = (
        "G"
        + lat_cells.astype(str).str.zfill(3)
        + "_"
        + lon_cells.astype(str).str.zfill(3)
    )

    if keep_region_metadata:
        out["lat_cell"] = pd.NA
        out["lon_cell"] = pd.NA
        out["region_lat_center"] = pd.NA
        out["region_lon_center"] = pd.NA

        out.loc[valid_mask, "lat_cell"] = lat_cells
        out.loc[valid_mask, "lon_cell"] = lon_cells
        out.loc[valid_mask, "region_lat_center"] = (
            lat_cells + 0.5) * region_grid_size - 90.0
        out.loc[valid_mask, "region_lon_center"] = (
            lon_cells + 0.5) * region_grid_size - 180.0

    return out


def main() -> None:
    args = parse_args()

    if args.region_grid_size <= 0:
        raise ValueError("--region-grid-size must be > 0")

    df = load_dataset(args.input_csv)
    out = add_region_code(
        df,
        region_grid_size=args.region_grid_size,
        keep_region_metadata=args.keep_region_metadata,
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    assigned_count = int(out["region_code"].notna().sum())
    missing_count = int(out["region_code"].isna().sum())
    distinct_regions = int(out["region_code"].nunique(dropna=True))

    print(f"Input rows: {len(out):,}")
    print(f"Rows with region_code: {assigned_count:,}")
    print(f"Rows without region_code: {missing_count:,}")
    print(f"Distinct region_code values: {distinct_regions:,}")
    print(f"Saved file: {args.output_csv}")


if __name__ == "__main__":
    main()
