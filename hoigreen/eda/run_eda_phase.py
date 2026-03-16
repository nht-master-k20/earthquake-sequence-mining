import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


RAW_NUMERIC_COLUMNS = [
    "latitude",
    "longitude",
    "depth",
    "mag",
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

EDA_NUMERIC_COLUMNS = ["mag", "depth", "sig", "gap", "rms", "nst", "dmin"]
RELATION_COLUMNS = ["mag", "depth", "sig", "gap", "rms", "nst", "dmin"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EDA phase for the raw earthquake dataset.")
    parser.add_argument("--input-csv", type=Path, default=Path("data/dongdat.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("hoigreen/eda/outputs"))
    parser.add_argument("--event-type", type=str, default="earthquake")
    parser.add_argument("--region-grid-size", type=float, default=2.5)
    parser.add_argument("--scatter-sample-size", type=int, default=25000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def load_dataset(input_csv: Path, event_type: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    raw_df = pd.read_csv(input_csv, low_memory=False)
    if "Unnamed: 0" in raw_df.columns:
        raw_df = raw_df.drop(columns=["Unnamed: 0"])

    required = {"id", "time", "latitude", "longitude", "depth", "mag", "type"}
    missing = sorted(required - set(raw_df.columns))
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    raw_df["time"] = pd.to_datetime(raw_df["time"], errors="coerce", utc=True)
    for column in RAW_NUMERIC_COLUMNS:
        if column in raw_df.columns:
            raw_df[column] = pd.to_numeric(raw_df[column], errors="coerce")

    raw_df = raw_df.dropna(subset=["id", "time", "latitude", "longitude", "depth", "mag"]).copy()
    analysis_df = raw_df[raw_df["type"].fillna("").eq(event_type)].copy()
    analysis_df["year"] = analysis_df["time"].dt.year.astype(int)
    analysis_df["month"] = analysis_df["time"].dt.month.astype(int)
    analysis_df["tsunami"] = analysis_df["tsunami"].fillna(0).clip(lower=0, upper=1).astype(int)
    return raw_df.sort_values("time").reset_index(drop=True), analysis_df.sort_values("time").reset_index(drop=True)


def add_region_ids(df: pd.DataFrame, region_grid_size: float) -> pd.DataFrame:
    out = df.copy()
    out["lat_cell"] = np.floor((out["latitude"] + 90.0) / region_grid_size).astype(int)
    out["lon_cell"] = np.floor((out["longitude"] + 180.0) / region_grid_size).astype(int)
    out["region_code"] = (
        "G"
        + out["lat_cell"].astype(str).str.zfill(3)
        + "_"
        + out["lon_cell"].astype(str).str.zfill(3)
    )
    out["region_lat_center"] = (out["lat_cell"] + 0.5) * region_grid_size - 90.0
    out["region_lon_center"] = (out["lon_cell"] + 0.5) * region_grid_size - 180.0
    return out


def build_dataset_overview(raw_df: pd.DataFrame, analysis_df: pd.DataFrame, grid_size: float) -> Dict[str, object]:
    type_counts = raw_df["type"].fillna("unknown").value_counts().to_dict()
    overview = {
        "input_rows_after_basic_cleaning": int(len(raw_df)),
        "analysis_rows_after_type_filter": int(len(analysis_df)),
        "analysis_event_type": "earthquake",
        "analysis_share": float(len(analysis_df) / max(len(raw_df), 1)),
        "time_min": str(analysis_df["time"].min()),
        "time_max": str(analysis_df["time"].max()),
        "grid_size_degree": grid_size,
        "distinct_regions": int(analysis_df["region_code"].nunique()),
        "type_counts": type_counts,
    }
    return overview


def build_numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = df[EDA_NUMERIC_COLUMNS].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).T.reset_index()
    summary = summary.rename(columns={"index": "feature"})
    summary["missing_ratio"] = df[EDA_NUMERIC_COLUMNS].isna().mean().values
    return summary


def build_yearly_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("year", as_index=False)
        .agg(
            event_count=("id", "size"),
            mag_mean=("mag", "mean"),
            mag_max=("mag", "max"),
            depth_mean=("depth", "mean"),
        )
        .sort_values("year")
        .reset_index(drop=True)
    )


def build_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["year", "month"], as_index=False)
        .agg(event_count=("id", "size"), mag_mean=("mag", "mean"))
        .sort_values(["year", "month"])
        .reset_index(drop=True)
    )


def build_region_summary(df: pd.DataFrame) -> pd.DataFrame:
    region_summary = (
        df.groupby(["region_code", "region_lat_center", "region_lon_center"], as_index=False)
        .agg(
            event_count=("id", "size"),
            mag_mean=("mag", "mean"),
            mag_max=("mag", "max"),
            depth_mean=("depth", "mean"),
            sig_mean=("sig", "mean"),
            tsunami_rate=("tsunami", "mean"),
        )
        .sort_values("event_count", ascending=False)
        .reset_index(drop=True)
    )
    region_summary["event_share"] = region_summary["event_count"] / region_summary["event_count"].sum()
    return region_summary


def plot_type_distribution(raw_df: pd.DataFrame, output_path: Path) -> None:
    type_counts = raw_df["type"].fillna("unknown").value_counts().head(10).reset_index()
    type_counts.columns = ["type", "count"]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=type_counts, x="count", y="type", ax=ax, color="#1d7874")
    ax.set_title("Raw Event Type Distribution")
    ax.set_xlabel("Count")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_missingness(df: pd.DataFrame, output_path: Path) -> None:
    missingness = df[["mag", "depth", "sig", "gap", "rms", "nst", "dmin", "mmi", "cdi", "felt"]].isna().mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=missingness.index, y=missingness.values, ax=ax, color="#6c757d")
    ax.set_title("Missing Ratio of Analysis Fields")
    ax.set_xlabel("")
    ax.set_ylabel("Missing ratio")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_numeric_distributions(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    plot_columns = EDA_NUMERIC_COLUMNS + ["tsunami"]
    for ax, column in zip(axes.flat, plot_columns):
        series = df[column].dropna()
        if series.empty:
            ax.set_visible(False)
            continue
        if column == "tsunami":
            sns.countplot(x=series, ax=ax, color="#ee6c4d")
        else:
            clipped = series.clip(upper=series.quantile(0.99))
            sns.histplot(clipped, bins=40, kde=True, ax=ax, color="#457b9d")
        ax.set_title(column)
        ax.set_xlabel("")
    fig.suptitle("Distribution of Core Analysis Variables", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    corr_df = df[RELATION_COLUMNS].copy().fillna(df[RELATION_COLUMNS].median(numeric_only=True))
    corr = corr_df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
    ax.set_title("Correlation Between Numeric Variables")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return corr


def plot_relationship_panel(
    df: pd.DataFrame,
    output_path: Path,
    sample_size: int,
    random_state: int,
) -> None:
    plot_df = df[["mag", "depth", "sig", "gap", "nst", "dmin"]].dropna()
    if len(plot_df) > sample_size:
        plot_df = plot_df.sample(n=sample_size, random_state=random_state)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    sns.scatterplot(data=plot_df, x="mag", y="sig", s=12, alpha=0.25, ax=axes[0, 0], color="#0b6e4f")
    axes[0, 0].set_title("Magnitude vs Significance")
    sns.scatterplot(data=plot_df, x="depth", y="mag", s=12, alpha=0.25, ax=axes[0, 1], color="#f4a261")
    axes[0, 1].set_title("Depth vs Magnitude")
    sns.scatterplot(data=plot_df, x="gap", y="nst", s=12, alpha=0.25, ax=axes[1, 0], color="#5a189a")
    axes[1, 0].set_title("Gap vs Number of Stations")
    sns.scatterplot(data=plot_df, x="dmin", y="mag", s=12, alpha=0.25, ax=axes[1, 1], color="#bc4749")
    axes[1, 1].set_title("dmin vs Magnitude")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_yearly_trend(yearly_summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(yearly_summary["year"], yearly_summary["event_count"], color="#0d3b66", linewidth=1.5)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Event count", color="#0d3b66")
    ax1.tick_params(axis="y", labelcolor="#0d3b66")
    ax1.grid(alpha=0.25, linestyle="--")

    ax2 = ax1.twinx()
    ax2.plot(yearly_summary["year"], yearly_summary["mag_mean"], color="#d62828", linewidth=1.5)
    ax2.set_ylabel("Mean magnitude", color="#d62828")
    ax2.tick_params(axis="y", labelcolor="#d62828")

    ax1.set_title("Yearly Earthquake Activity and Mean Magnitude")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_monthly_seasonality(monthly_summary: pd.DataFrame, output_path: Path) -> None:
    pivot = monthly_summary.pivot(index="year", columns="month", values="event_count").fillna(0)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax)
    ax.set_title("Monthly Event Count Heatmap")
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_spatial_density(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    hb = ax.hexbin(
        df["longitude"],
        df["latitude"],
        gridsize=90,
        cmap="viridis",
        bins="log",
        mincnt=1,
    )
    ax.set_title("Spatial Density of Earthquake Events")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.colorbar(hb, ax=ax, label="log10(count)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_region_activity(region_summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    scatter = ax.scatter(
        region_summary["region_lon_center"],
        region_summary["region_lat_center"],
        s=np.clip(np.log1p(region_summary["event_count"]) * 8.0, 8.0, 130.0),
        c=region_summary["mag_mean"],
        cmap="plasma",
        alpha=0.75,
    )
    ax.set_title("Region-Level Earthquake Activity")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.2, linestyle="--")
    fig.colorbar(scatter, ax=ax, label="Mean magnitude")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_top_regions(region_summary: pd.DataFrame, output_path: Path) -> None:
    top_regions = region_summary.head(15).sort_values("event_count", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=top_regions, x="event_count", y="region_code", ax=ax, color="#264653")
    ax.set_title("Top 15 Regions by Event Count")
    ax.set_xlabel("Event count")
    ax.set_ylabel("Region code")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def build_report(
    output_path: Path,
    overview: Dict[str, object],
    numeric_summary: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
    yearly_summary: pd.DataFrame,
    monthly_summary: pd.DataFrame,
    region_summary: pd.DataFrame,
) -> None:
    top_missing = (
        numeric_summary[["feature", "missing_ratio"]]
        .sort_values("missing_ratio", ascending=False)
        .head(5)
        .copy()
    )
    top_missing["missing_ratio"] = top_missing["missing_ratio"].round(4)

    top_corr = correlation_matrix.where(~np.eye(len(correlation_matrix), dtype=bool)).stack().reset_index()
    top_corr.columns = ["feature_a", "feature_b", "correlation"]
    top_corr["pair_key"] = top_corr.apply(lambda row: "::".join(sorted([row["feature_a"], row["feature_b"]])), axis=1)
    top_corr = top_corr.drop_duplicates("pair_key").drop(columns=["pair_key"])
    top_corr["abs_corr"] = top_corr["correlation"].abs()
    top_corr = top_corr.sort_values("abs_corr", ascending=False).head(5)
    top_corr["correlation"] = top_corr["correlation"].round(3)
    top_corr = top_corr[["feature_a", "feature_b", "correlation"]]

    busiest_year = yearly_summary.sort_values("event_count", ascending=False).iloc[0]
    strongest_year = yearly_summary.sort_values("mag_max", ascending=False).iloc[0]
    month_of_year = monthly_summary.groupby("month", as_index=False).agg(event_count=("event_count", "sum"))
    peak_month = month_of_year.sort_values("event_count", ascending=False).iloc[0]
    top_regions = region_summary.head(10)[["region_code", "event_count", "mag_mean", "mag_max", "event_share"]].copy()
    top_regions["mag_mean"] = top_regions["mag_mean"].round(3)
    top_regions["mag_max"] = top_regions["mag_max"].round(3)
    top_regions["event_share"] = top_regions["event_share"].round(4)

    report = f"""# EDA Report

## Scope

- Input: `data/dongdat.csv`
- Rows after basic cleaning: `{overview["input_rows_after_basic_cleaning"]:,}`
- Rows used for EDA (`earthquake` only): `{overview["analysis_rows_after_type_filter"]:,}`
- Time range: `{overview["time_min"]}` -> `{overview["time_max"]}`
- Distinct analysis regions (`{overview["grid_size_degree"]}` degree grid): `{overview["distinct_regions"]:,}`

## 1. Distribution Analysis

- Raw data contains non-earthquake event types, so the main EDA focuses on `earthquake` only.
- Core variables are strongly skewed: many small magnitudes and shallow events, with a long tail of stronger and deeper events.
- `mmi`, `cdi`, `felt` are highly incomplete and should stay descriptive rather than become core modeling features.

### Highest missing ratios

{markdown_table(top_missing)}

## 2. Relationship Analysis

- The strongest numeric relationships are below.
- `mag` and `sig` are expected to move together very strongly, so using both in downstream modeling should be done consciously.
- Observation quality variables (`gap`, `nst`, `dmin`, `rms`) carry information about data reliability and station geometry, not just earthquake physics.

### Strongest correlations

{markdown_table(top_corr)}

## 3. Temporal Analysis

- Busiest year by event count: `{int(busiest_year["year"])}` with `{int(busiest_year["event_count"]):,}` events
- Year with strongest maximum magnitude: `{int(strongest_year["year"])}` with `mag_max = {float(strongest_year["mag_max"]):.2f}`
- Peak month-of-year by total count: `{int(peak_month["month"])}` with `{int(peak_month["event_count"]):,}` events

## 4. Spatial Analysis

- Activity is highly concentrated in a limited set of spatial cells.
- Top regions below are based on the same grid logic that will be reused in Pattern Discovering.

### Top regions

{markdown_table(top_regions)}

## Recommended Follow-up for Pattern Discovering

- Keep the core stable features: `mag`, `depth`, `sig`, `gap`, `rms`, `nst`, `dmin`, `tsunami`
- Apply log transform to skewed variables before scaling
- Use the same region grid to connect EDA and cluster interpretation
- Exclude `mmi`, `cdi`, `felt` from core clustering because missingness is too high
"""
    output_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_output_dir(args.output_dir)
    sns.set_theme(style="whitegrid")

    raw_df, analysis_df = load_dataset(args.input_csv, args.event_type)
    analysis_df = add_region_ids(analysis_df, args.region_grid_size)

    overview = build_dataset_overview(raw_df, analysis_df, args.region_grid_size)
    numeric_summary = build_numeric_summary(analysis_df)
    yearly_summary = build_yearly_summary(analysis_df)
    monthly_summary = build_monthly_summary(analysis_df)
    region_summary = build_region_summary(analysis_df)

    plot_type_distribution(raw_df, args.output_dir / "07_type_distribution.png")
    plot_missingness(analysis_df, args.output_dir / "08_missingness.png")
    plot_numeric_distributions(analysis_df, args.output_dir / "09_numeric_distributions.png")
    correlation_matrix = plot_correlation_heatmap(analysis_df, args.output_dir / "10_correlation_heatmap.png")
    plot_relationship_panel(
        analysis_df,
        args.output_dir / "11_relationship_panel.png",
        sample_size=args.scatter_sample_size,
        random_state=args.random_state,
    )
    plot_yearly_trend(yearly_summary, args.output_dir / "12_yearly_trend.png")
    plot_monthly_seasonality(monthly_summary, args.output_dir / "13_monthly_seasonality.png")
    plot_spatial_density(analysis_df, args.output_dir / "14_spatial_density.png")
    plot_region_activity(region_summary, args.output_dir / "15_region_activity.png")
    plot_top_regions(region_summary, args.output_dir / "16_top_regions.png")

    (args.output_dir / "00_dataset_overview.json").write_text(json.dumps(overview, indent=2), encoding="utf-8")
    numeric_summary.to_csv(args.output_dir / "01_numeric_summary.csv", index=False)
    correlation_matrix.to_csv(args.output_dir / "02_correlation_matrix.csv")
    yearly_summary.to_csv(args.output_dir / "03_yearly_summary.csv", index=False)
    monthly_summary.to_csv(args.output_dir / "04_monthly_summary.csv", index=False)
    region_summary.to_csv(args.output_dir / "05_region_summary.csv", index=False)
    build_report(
        args.output_dir / "06_report.md",
        overview=overview,
        numeric_summary=numeric_summary,
        correlation_matrix=correlation_matrix,
        yearly_summary=yearly_summary,
        monthly_summary=monthly_summary,
        region_summary=region_summary,
    )


if __name__ == "__main__":
    main()
