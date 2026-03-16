import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import folium
from folium.plugins import HeatMapWithTime
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


RELATION_COLS = ["mag", "depth", "gap", "nst", "rms"]


def load_dataset(input_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)

    required_cols = {
        "id",
        "time",
        "latitude",
        "longitude",
        "depth",
        "mag",
        "gap",
        "nst",
        "rms",
    }
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)

    num_cols = ["latitude", "longitude", "depth", "mag", "gap", "nst", "rms"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["time", "latitude", "longitude", "depth", "mag"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def plot_time_series(df: pd.DataFrame, output_dir: Path) -> Path:
    monthly = (
        df.set_index("time")
        .resample("ME")
        .agg(event_count=("id", "count"), mean_mag=("mag", "mean"))
        .dropna()
    )

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(
        monthly.index,
        monthly["event_count"],
        color="#0d3b66",
        linewidth=1.2,
        label="Event count",
    )
    ax1.set_ylabel("Monthly earthquake count", color="#0d3b66")
    ax1.tick_params(axis="y", labelcolor="#0d3b66")
    ax1.grid(True, linestyle="--", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        monthly.index,
        monthly["mean_mag"],
        color="#f95738",
        linewidth=1.5,
        label="Mean magnitude",
    )
    ax2.set_ylabel("Monthly mean magnitude", color="#f95738")
    ax2.tick_params(axis="y", labelcolor="#f95738")

    ax1.set_title("Global Earthquake Timeline: Monthly Count and Mean Magnitude")
    ax1.set_xlabel("Time")
    fig.tight_layout()

    out_path = output_dir / "01_global_time_series.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def build_time_geo_heatmap(
    df: pd.DataFrame,
    output_dir: Path,
    max_points_per_step: int,
    random_state: int,
) -> Path:
    map_df = df[["time", "latitude", "longitude", "mag"]].copy()
    map_df["month"] = map_df["time"].dt.strftime("%Y-%m")

    time_steps: List[str] = []
    heatmap_data: List[List[List[float]]] = []

    for month, group in map_df.groupby("month", sort=True):
        group = group.dropna(subset=["latitude", "longitude", "mag"])
        if group.empty:
            continue

        if len(group) > max_points_per_step:
            group = group.sample(
                n=max_points_per_step,
                random_state=random_state,
            )

        points = [
            [float(lat), float(lon), max(0.1, min(float(mag) / 10.0, 1.0))]
            for lat, lon, mag in group[["latitude", "longitude", "mag"]].itertuples(index=False)
        ]
        if points:
            heatmap_data.append(points)
            time_steps.append(month)

    if not heatmap_data:
        raise ValueError("No valid data to build dynamic map.")

    center_lat = float(map_df["latitude"].mean())
    center_lon = float(map_df["longitude"].mean())
    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=2,
        tiles="CartoDB positron",
        control_scale=True,
        prefer_canvas=True,
    )

    HeatMapWithTime(
        heatmap_data,
        index=time_steps,
        auto_play=False,
        max_opacity=0.85,
        radius=12,
        use_local_extrema=False,
    ).add_to(fmap)

    out_path = output_dir / "02_global_time_geo_heatmap.html"
    fmap.save(out_path)
    return out_path


def plot_depth_mag_relationship(
    df: pd.DataFrame,
    output_dir: Path,
    sample_size: int,
    random_state: int,
) -> Tuple[Path, Dict[str, float]]:
    pair_df = df[["depth", "mag"]].dropna()
    if len(pair_df) > sample_size:
        pair_df = pair_df.sample(n=sample_size, random_state=random_state)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.regplot(
        data=pair_df,
        x="depth",
        y="mag",
        scatter_kws={"alpha": 0.18, "s": 16, "color": "#1f7a8c"},
        line_kws={"color": "#c1121f", "linewidth": 2},
        ax=ax,
    )
    ax.set_title("Depth vs Magnitude")
    ax.set_xlabel("Depth (km)")
    ax.set_ylabel("Magnitude")
    ax.grid(True, linestyle="--", alpha=0.25)
    fig.tight_layout()

    out_path = output_dir / "03_depth_vs_magnitude.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    corr_stats = {
        "pearson_depth_mag": float(pair_df.corr(method="pearson").iloc[0, 1]),
        "spearman_depth_mag": float(pair_df.corr(method="spearman").iloc[0, 1]),
    }
    return out_path, corr_stats


def plot_parameter_relationships(
    df: pd.DataFrame,
    output_dir: Path,
    pairplot_sample_size: int,
    random_state: int,
) -> Tuple[Path, Path, pd.DataFrame, pd.DataFrame]:
    rel_df = df[RELATION_COLS].dropna()
    if rel_df.empty:
        raise ValueError("No valid rows to analyze parameter relationships.")

    pearson_corr = rel_df.corr(method="pearson")
    spearman_corr = rel_df.corr(method="spearman")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(
        pearson_corr,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=-1,
        vmax=1,
        square=True,
        ax=axes[0],
    )
    axes[0].set_title("Pearson Correlation")

    sns.heatmap(
        spearman_corr,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=-1,
        vmax=1,
        square=True,
        ax=axes[1],
    )
    axes[1].set_title("Spearman Correlation")
    fig.tight_layout()

    corr_out = output_dir / "04_parameter_correlation_heatmaps.png"
    fig.savefig(corr_out, dpi=180)
    plt.close(fig)

    pairplot_df = rel_df
    if len(pairplot_df) > pairplot_sample_size:
        pairplot_df = pairplot_df.sample(n=pairplot_sample_size, random_state=random_state)

    pair_grid = sns.pairplot(
        pairplot_df,
        vars=RELATION_COLS,
        diag_kind="hist",
        corner=True,
        plot_kws={"alpha": 0.25, "s": 14, "color": "#0b6e4f"},
    )
    pair_grid.fig.suptitle("Pairwise Relationship: mag, depth, gap, nst, rms", y=1.02)

    pair_out = output_dir / "05_parameter_pairplot.png"
    pair_grid.savefig(pair_out, dpi=160)
    plt.close(pair_grid.fig)

    return corr_out, pair_out, pearson_corr, spearman_corr


def build_report(
    df: pd.DataFrame,
    output_dir: Path,
    corr_stats: Dict[str, float],
    pearson_corr: pd.DataFrame,
    spearman_corr: pd.DataFrame,
) -> Path:
    report_path = output_dir / "report.md"

    time_min = df["time"].min()
    time_max = df["time"].max()

    lines = [
        "# Earthquake EDA Report",
        "",
        "## Dataset Overview",
        f"- Rows: {len(df):,}",
        f"- Time range (UTC): {time_min} -> {time_max}",
        f"- Latitude range: {df['latitude'].min():.3f} -> {df['latitude'].max():.3f}",
        f"- Longitude range: {df['longitude'].min():.3f} -> {df['longitude'].max():.3f}",
        f"- Depth range (km): {df['depth'].min():.3f} -> {df['depth'].max():.3f}",
        f"- Magnitude range: {df['mag'].min():.3f} -> {df['mag'].max():.3f}",
        "",
        "## Depth vs Magnitude",
        f"- Pearson correlation: {corr_stats['pearson_depth_mag']:.4f}",
        f"- Spearman correlation: {corr_stats['spearman_depth_mag']:.4f}",
        "",
        "## Parameter Correlation (Pearson)",
        "```",
        pearson_corr.to_string(float_format=lambda v: f"{v:.4f}"),
        "```",
        "",
        "## Parameter Correlation (Spearman)",
        "```",
        spearman_corr.to_string(float_format=lambda v: f"{v:.4f}"),
        "```",
        "",
        "## Generated Files",
        "- 01_global_time_series.png",
        "- 02_global_time_geo_heatmap.html",
        "- 03_depth_vs_magnitude.png",
        "- 04_parameter_correlation_heatmaps.png",
        "- 05_parameter_pairplot.png",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_eda(
    input_csv: Path,
    output_dir: Path,
    max_points_per_step: int,
    depth_mag_sample_size: int,
    pairplot_sample_size: int,
    random_state: int,
) -> None:
    sns.set_theme(style="whitegrid")
    ensure_output_dir(output_dir)

    df = load_dataset(input_csv)

    time_series_path = plot_time_series(df, output_dir)
    map_path = build_time_geo_heatmap(
        df,
        output_dir,
        max_points_per_step=max_points_per_step,
        random_state=random_state,
    )
    depth_mag_path, corr_stats = plot_depth_mag_relationship(
        df,
        output_dir,
        sample_size=depth_mag_sample_size,
        random_state=random_state,
    )
    corr_heatmap_path, pairplot_path, pearson_corr, spearman_corr = plot_parameter_relationships(
        df,
        output_dir,
        pairplot_sample_size=pairplot_sample_size,
        random_state=random_state,
    )
    report_path = build_report(
        df,
        output_dir,
        corr_stats=corr_stats,
        pearson_corr=pearson_corr,
        spearman_corr=spearman_corr,
    )

    print("=" * 70)
    print("EDA completed successfully")
    print("=" * 70)
    print(f"Input: {input_csv}")
    print(f"Rows used: {len(df):,}")
    print(f"Output directory: {output_dir}")
    print(f"- {time_series_path.name}")
    print(f"- {map_path.name}")
    print(f"- {depth_mag_path.name}")
    print(f"- {corr_heatmap_path.name}")
    print(f"- {pairplot_path.name}")
    print(f"- {report_path.name}")
    print("=" * 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run EDA for global earthquake data with time-location visualization, "
            "depth-magnitude analysis, and parameter relationship plots."
        )
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Path to cleaned earthquake CSV (e.g., output of preprocess_usgs_quakes.py).",
    )
    parser.add_argument(
        "--output-dir",
        default="hoigreen/preprocessing/eda_outputs",
        help="Directory for generated figures and report.",
    )
    parser.add_argument(
        "--max-map-points-per-step",
        type=int,
        default=450,
        help="Maximum points per month in dynamic map to keep rendering fast.",
    )
    parser.add_argument(
        "--depth-mag-sample-size",
        type=int,
        default=80000,
        help="Sample size for depth vs magnitude scatter/regression.",
    )
    parser.add_argument(
        "--pairplot-sample-size",
        type=int,
        default=7000,
        help="Sample size for pairplot.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_eda(
        input_csv=Path(args.input_csv),
        output_dir=Path(args.output_dir),
        max_points_per_step=args.max_map_points_per_step,
        depth_mag_sample_size=args.depth_mag_sample_size,
        pairplot_sample_size=args.pairplot_sample_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
