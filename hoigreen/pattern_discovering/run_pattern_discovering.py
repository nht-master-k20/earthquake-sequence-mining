import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler


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

CLUSTER_FEATURES = [
    "mag",
    "depth_log1p",
    "sig_log1p",
    "gap",
    "rms_log1p",
    "nst_log1p",
    "dmin_log1p",
    "tsunami",
]

REGION_CLUSTER_FEATURES = [
    "event_count_log1p",
    "mag_mean",
    "mag_p90",
    "mag_max",
    "depth_mean",
    "depth_p90",
    "sig_mean",
    "major_quake_ratio",
    "shallow_ratio",
    "deep_ratio",
    "tsunami_rate",
]

PLOT_EVENT_COLUMNS = ["mag", "depth", "sig", "gap", "rms", "nst", "dmin", "tsunami"]
HIGH_MISSING_COLUMNS = ["mmi", "cdi", "felt"]
LOW_MISSING_COLUMNS = ["gap", "rms", "nst", "dmin"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone Pattern Discovering pipeline for the raw dongdat.csv dataset."
    )
    parser.add_argument("--input-csv", type=Path, default=Path("data/dongdat.csv"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("hoigreen/pattern_discovering/outputs"),
    )
    parser.add_argument("--event-type", type=str, default="earthquake")
    parser.add_argument("--region-grid-size", type=float, default=2.5)
    parser.add_argument("--event-sample-size", type=int, default=200000)
    parser.add_argument("--eval-sample-size", type=int, default=25000)
    parser.add_argument("--plot-sample-size", type=int, default=60000)
    parser.add_argument("--event-k-min", type=int, default=2)
    parser.add_argument("--event-k-max", type=int, default=8)
    parser.add_argument("--region-k-min", type=int, default=2)
    parser.add_argument("--region-k-max", type=int, default=8)
    parser.add_argument("--min-events-per-region", type=int, default=25)
    parser.add_argument("--top-regions", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def load_raw_dataset(input_csv: Path, event_type: str) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv, low_memory=False)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    required_columns = {"id", "time", "latitude", "longitude", "depth", "mag", "type"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Input CSV missing required columns: {missing_columns}")

    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    for column in RAW_NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["id", "time", "latitude", "longitude", "depth", "mag"]).copy()
    if event_type:
        df = df[df["type"].fillna("").eq(event_type)].copy()

    df["tsunami"] = df["tsunami"].fillna(0).clip(lower=0, upper=1).astype(int)
    df = df.sort_values("time").reset_index(drop=True)
    return df


def add_region_ids(df: pd.DataFrame, region_grid_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    out["lat_cell"] = np.floor((out["latitude"] + 90.0) / region_grid_size).astype(int)
    out["lon_cell"] = np.floor((out["longitude"] + 180.0) / region_grid_size).astype(int)
    out["region_key"] = out["lat_cell"].astype(str) + "_" + out["lon_cell"].astype(str)
    out["region_code"] = (
        "G"
        + out["lat_cell"].astype(str).str.zfill(3)
        + "_"
        + out["lon_cell"].astype(str).str.zfill(3)
    )
    out["region_lat_center"] = (out["lat_cell"] + 0.5) * region_grid_size - 90.0
    out["region_lon_center"] = (out["lon_cell"] + 0.5) * region_grid_size - 180.0

    region_lookup = (
        out[
            [
                "region_key",
                "region_code",
                "lat_cell",
                "lon_cell",
                "region_lat_center",
                "region_lon_center",
            ]
        ]
        .drop_duplicates()
        .sort_values(["lat_cell", "lon_cell"])
        .reset_index(drop=True)
    )
    region_lookup["region_id"] = np.arange(1, len(region_lookup) + 1, dtype=int)
    out = out.merge(region_lookup[["region_key", "region_id"]], on="region_key", how="left")
    return out, region_lookup


def add_engineered_fields(df: pd.DataFrame, region_grid_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    out["year"] = out["time"].dt.year.astype(int)
    out["month"] = out["time"].dt.month.astype(int)
    out["hour"] = out["time"].dt.hour.astype(int)
    out["month_sin"] = np.sin(2.0 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2.0 * np.pi * out["month"] / 12.0)
    out["hour_sin"] = np.sin(2.0 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * out["hour"] / 24.0)
    out["depth_log1p"] = np.log1p(out["depth"].clip(lower=0.0))
    out["sig_log1p"] = np.log1p(out["sig"].fillna(0.0).clip(lower=0.0))
    out["rms_log1p"] = np.log1p(out["rms"].fillna(0.0).clip(lower=0.0))
    out["nst_log1p"] = np.log1p(out["nst"].fillna(0.0).clip(lower=0.0))
    out["dmin_log1p"] = np.log1p(out["dmin"].fillna(0.0).clip(lower=0.0))
    out["has_mmi"] = out["mmi"].notna().astype(int)
    out["has_cdi"] = out["cdi"].notna().astype(int)
    out["has_felt"] = out["felt"].notna().astype(int)
    out["depth_band"] = pd.cut(
        out["depth"],
        bins=[-np.inf, 70.0, 300.0, np.inf],
        labels=["shallow", "intermediate", "deep"],
        include_lowest=True,
    ).astype(str)
    out["mag_band"] = pd.cut(
        out["mag"],
        bins=[-np.inf, 2.0, 4.0, 5.0, 6.0, 7.0, np.inf],
        labels=["micro_minor", "light", "moderate", "strong", "major", "great"],
        include_lowest=True,
    ).astype(str)

    for column in LOW_MISSING_COLUMNS:
        out[f"{column}_missing"] = out[column].isna().astype(int)

    out, region_lookup = add_region_ids(out, region_grid_size=region_grid_size)
    return out, region_lookup


def build_feature_overview(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for column in df.columns:
        if column == "time":
            continue
        row: Dict[str, float] = {
            "column": column,
            "missing_ratio": float(df[column].isna().mean()),
        }
        if pd.api.types.is_numeric_dtype(df[column]):
            row["mean"] = float(df[column].mean(skipna=True))
            row["std"] = float(df[column].std(skipna=True))
        else:
            row["mean"] = np.nan
            row["std"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["missing_ratio", "column"], ascending=[False, True])


def prepare_event_matrix(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, SimpleImputer, RobustScaler]:
    event_features = df[CLUSTER_FEATURES].copy()
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler(quantile_range=(10.0, 90.0))
    x_imputed = imputer.fit_transform(event_features)
    x_scaled = scaler.fit_transform(x_imputed)
    return event_features, x_scaled, imputer, scaler


def choose_sample_indices(total_size: int, sample_size: int, random_state: int) -> np.ndarray:
    if sample_size <= 0 or sample_size >= total_size:
        return np.arange(total_size)
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(total_size, size=sample_size, replace=False))


def evaluate_event_k_values(
    x_scaled: np.ndarray,
    k_min: int,
    k_max: int,
    train_sample_size: int,
    eval_sample_size: int,
    random_state: int,
) -> Tuple[pd.DataFrame, np.ndarray]:
    train_idx = choose_sample_indices(len(x_scaled), train_sample_size, random_state)
    eval_idx = choose_sample_indices(len(train_idx), min(eval_sample_size, len(train_idx)), random_state + 99)
    eval_x = x_scaled[train_idx[eval_idx]]

    valid_k_max = min(k_max, max(k_min, len(train_idx) - 1))
    records: List[Dict[str, float]] = []
    for k in range(k_min, valid_k_max + 1):
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=random_state + k,
            batch_size=4096,
            n_init="auto",
        )
        model.fit(x_scaled[train_idx])
        labels = model.predict(eval_x)
        if np.unique(labels).size < 2:
            silhouette = np.nan
        else:
            silhouette = float(silhouette_score(eval_x, labels))
        records.append(
            {
                "k": k,
                "silhouette": silhouette,
                "inertia": float(model.inertia_),
                "train_rows": int(len(train_idx)),
                "eval_rows": int(len(eval_x)),
            }
        )

    k_eval = pd.DataFrame(records).sort_values("k").reset_index(drop=True)
    return k_eval, train_idx


def fit_event_clustering(
    x_scaled: np.ndarray,
    train_idx: np.ndarray,
    best_k: int,
    random_state: int,
) -> Tuple[np.ndarray, MiniBatchKMeans]:
    model = MiniBatchKMeans(
        n_clusters=best_k,
        random_state=random_state + 1000,
        batch_size=4096,
        n_init="auto",
    )
    model.fit(x_scaled[train_idx])
    labels = model.predict(x_scaled)
    return labels.astype(int), model


def restore_event_centroids(
    model: MiniBatchKMeans,
    scaler: RobustScaler,
    cluster_features: Iterable[str],
) -> pd.DataFrame:
    centers = scaler.inverse_transform(model.cluster_centers_)
    centroid_df = pd.DataFrame(centers, columns=list(cluster_features))
    centroid_df.insert(0, "event_cluster", np.arange(len(centroid_df), dtype=int))
    centroid_df["depth_centroid_km"] = np.expm1(centroid_df["depth_log1p"])
    centroid_df["sig_centroid"] = np.expm1(centroid_df["sig_log1p"])
    centroid_df["rms_centroid"] = np.expm1(centroid_df["rms_log1p"])
    centroid_df["nst_centroid"] = np.expm1(centroid_df["nst_log1p"])
    centroid_df["dmin_centroid"] = np.expm1(centroid_df["dmin_log1p"])
    return centroid_df


def summarize_event_clusters(df: pd.DataFrame, centroid_df: pd.DataFrame) -> pd.DataFrame:
    event_profile = (
        df.groupby("event_cluster", as_index=False)
        .agg(
            event_count=("id", "size"),
            mag_mean=("mag", "mean"),
            mag_p90=("mag", lambda s: s.quantile(0.9)),
            depth_mean=("depth", "mean"),
            sig_mean=("sig", "mean"),
            tsunami_rate=("tsunami", "mean"),
            region_count=("region_id", "nunique"),
        )
        .sort_values("event_count", ascending=False)
    )

    top_mag_type = (
        df.groupby(["event_cluster", "magType"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["event_cluster", "count"], ascending=[True, False])
        .drop_duplicates(subset=["event_cluster"])
        .rename(columns={"magType": "top_mag_type"})
        [["event_cluster", "top_mag_type"]]
    )

    top_depth_band = (
        df.groupby(["event_cluster", "depth_band"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["event_cluster", "count"], ascending=[True, False])
        .drop_duplicates(subset=["event_cluster"])
        .rename(columns={"depth_band": "top_depth_band"})
        [["event_cluster", "top_depth_band"]]
    )

    event_profile = event_profile.merge(top_mag_type, on="event_cluster", how="left")
    event_profile = event_profile.merge(top_depth_band, on="event_cluster", how="left")
    event_profile = event_profile.merge(centroid_df, on="event_cluster", how="left")
    return event_profile


def build_region_summary(df: pd.DataFrame, region_lookup: pd.DataFrame) -> pd.DataFrame:
    region_summary = (
        df.groupby(
            ["region_id", "region_key", "region_code", "region_lat_center", "region_lon_center"],
            as_index=False,
        )
        .agg(
            event_count=("id", "size"),
            mag_mean=("mag", "mean"),
            mag_p90=("mag", lambda s: s.quantile(0.9)),
            mag_max=("mag", "max"),
            depth_mean=("depth", "mean"),
            depth_p90=("depth", lambda s: s.quantile(0.9)),
            sig_mean=("sig", "mean"),
            tsunami_rate=("tsunami", "mean"),
            active_years=("year", "nunique"),
            first_time=("time", "min"),
            last_time=("time", "max"),
        )
        .sort_values("event_count", ascending=False)
        .reset_index(drop=True)
    )

    band_share = (
        df.groupby(["region_id", "depth_band"])
        .size()
        .unstack(fill_value=0)
        .rename_axis(None, axis=1)
        .reset_index()
    )
    for column in ["shallow", "intermediate", "deep"]:
        if column not in band_share.columns:
            band_share[column] = 0
    band_share["depth_total"] = band_share[["shallow", "intermediate", "deep"]].sum(axis=1)
    band_share["shallow_ratio"] = band_share["shallow"] / band_share["depth_total"].clip(lower=1)
    band_share["intermediate_ratio"] = band_share["intermediate"] / band_share["depth_total"].clip(lower=1)
    band_share["deep_ratio"] = band_share["deep"] / band_share["depth_total"].clip(lower=1)
    band_share = band_share[["region_id", "shallow_ratio", "intermediate_ratio", "deep_ratio"]]

    major_quake_ratio = (
        df.assign(is_major=(df["mag"] >= 5.0).astype(float))
        .groupby("region_id", as_index=False)
        .agg(major_quake_ratio=("is_major", "mean"))
    )

    felt_share = (
        df.groupby("region_id", as_index=False)
        .agg(has_felt_ratio=("has_felt", "mean"), has_mmi_ratio=("has_mmi", "mean"))
    )

    cluster_distribution = (
        df.groupby(["region_id", "event_cluster"])
        .size()
        .reset_index(name="count")
        .sort_values(["region_id", "count"], ascending=[True, False])
    )
    dominant_cluster = cluster_distribution.drop_duplicates(subset=["region_id"]).rename(
        columns={"event_cluster": "dominant_event_cluster", "count": "dominant_cluster_count"}
    )
    cluster_count = (
        cluster_distribution.groupby("region_id", as_index=False)
        .agg(event_cluster_count=("event_cluster", "nunique"), cluster_event_total=("count", "sum"))
    )
    dominant_cluster = dominant_cluster.merge(cluster_count, on="region_id", how="left")
    dominant_cluster["dominant_cluster_share"] = (
        dominant_cluster["dominant_cluster_count"] / dominant_cluster["cluster_event_total"].clip(lower=1)
    )
    dominant_cluster = dominant_cluster[
        ["region_id", "dominant_event_cluster", "event_cluster_count", "dominant_cluster_share"]
    ]

    region_summary = region_summary.merge(band_share, on="region_id", how="left")
    region_summary = region_summary.merge(major_quake_ratio, on="region_id", how="left")
    region_summary = region_summary.merge(felt_share, on="region_id", how="left")
    region_summary = region_summary.merge(dominant_cluster, on="region_id", how="left")
    region_summary["event_count_log1p"] = np.log1p(region_summary["event_count"])
    region_summary["region_rank_by_count"] = np.arange(1, len(region_summary) + 1, dtype=int)

    region_summary = region_summary.merge(
        region_lookup[["region_id", "lat_cell", "lon_cell"]],
        on="region_id",
        how="left",
    )
    return region_summary


def evaluate_region_k_values(
    region_summary: pd.DataFrame,
    min_events_per_region: int,
    k_min: int,
    k_max: int,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, RobustScaler]:
    eligible = region_summary[region_summary["event_count"] >= min_events_per_region].copy()
    if len(eligible) < 3:
        raise ValueError("Not enough active regions for region-level clustering.")

    region_x = eligible[REGION_CLUSTER_FEATURES].fillna(0.0)
    scaler = RobustScaler(quantile_range=(10.0, 90.0))
    region_x_scaled = scaler.fit_transform(region_x)

    valid_k_max = min(k_max, max(k_min, len(eligible) - 1))
    records: List[Dict[str, float]] = []
    for k in range(k_min, valid_k_max + 1):
        model = KMeans(n_clusters=k, random_state=random_state + k, n_init="auto")
        labels = model.fit_predict(region_x_scaled)
        if np.unique(labels).size < 2:
            silhouette = np.nan
        else:
            silhouette = float(silhouette_score(region_x_scaled, labels))
        records.append(
            {
                "k": k,
                "silhouette": silhouette,
                "inertia": float(model.inertia_),
                "eligible_regions": int(len(eligible)),
            }
        )
    return pd.DataFrame(records), eligible, scaler


def fit_region_clustering(
    region_summary: pd.DataFrame,
    eligible: pd.DataFrame,
    scaler: RobustScaler,
    best_k: int,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    region_x = eligible[REGION_CLUSTER_FEATURES].fillna(0.0)
    region_x_scaled = scaler.transform(region_x)
    model = KMeans(n_clusters=best_k, random_state=random_state + 2000, n_init="auto")
    labels = model.fit_predict(region_x_scaled).astype(int)

    eligible = eligible.copy()
    eligible["region_cluster"] = labels
    region_summary = region_summary.copy()
    region_summary["region_cluster"] = -1
    region_summary.loc[eligible.index, "region_cluster"] = labels

    centroid_df = pd.DataFrame(
        scaler.inverse_transform(model.cluster_centers_),
        columns=REGION_CLUSTER_FEATURES,
    )
    centroid_df.insert(0, "region_cluster", np.arange(len(centroid_df), dtype=int))

    region_profile = (
        eligible.groupby("region_cluster", as_index=False)
        .agg(
            region_count=("region_id", "size"),
            total_events=("event_count", "sum"),
            profile_mag_mean=("mag_mean", "mean"),
            profile_mag_p90=("mag_p90", "mean"),
            profile_mag_max=("mag_max", "mean"),
            profile_depth_mean=("depth_mean", "mean"),
            profile_depth_p90=("depth_p90", "mean"),
            profile_sig_mean=("sig_mean", "mean"),
            profile_major_quake_ratio=("major_quake_ratio", "mean"),
            profile_shallow_ratio=("shallow_ratio", "mean"),
            profile_deep_ratio=("deep_ratio", "mean"),
            profile_tsunami_rate=("tsunami_rate", "mean"),
        )
        .sort_values("total_events", ascending=False)
    )
    region_profile = region_profile.merge(centroid_df, on="region_cluster", how="left")
    return region_summary, region_profile


def plot_missingness(df: pd.DataFrame, output_path: Path) -> None:
    missingness = df.isna().mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(x=missingness.values, y=missingness.index, ax=ax, color="#2b7a78")
    ax.set_title("Missing Ratio of Raw Fields")
    ax.set_xlabel("Missing ratio")
    ax.set_ylabel("")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_correlation(df: pd.DataFrame, output_path: Path) -> None:
    corr_df = df[PLOT_EVENT_COLUMNS].copy()
    corr_df = corr_df.fillna(corr_df.median(numeric_only=True))
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr_df.corr(numeric_only=True), annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
    ax.set_title("Core Feature Correlation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_k_eval(k_eval: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.lineplot(data=k_eval, x="k", y="silhouette", marker="o", ax=axes[0], color="#0b6e4f")
    axes[0].set_title(f"{title}: silhouette")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("silhouette")
    sns.lineplot(data=k_eval, x="k", y="inertia", marker="o", ax=axes[1], color="#c44536")
    axes[1].set_title(f"{title}: inertia")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("inertia")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_event_clusters_pca(
    x_scaled: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    sample_size: int,
    random_state: int,
) -> None:
    sample_idx = choose_sample_indices(len(x_scaled), sample_size, random_state + 501)
    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(x_scaled[sample_idx])
    plot_df = pd.DataFrame(
        {
            "pc1": coords[:, 0],
            "pc2": coords[:, 1],
            "event_cluster": labels[sample_idx],
        }
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        data=plot_df,
        x="pc1",
        y="pc2",
        hue="event_cluster",
        palette="tab10",
        s=12,
        linewidth=0,
        alpha=0.65,
        ax=ax,
    )
    ax.set_title("Event Clusters in PCA Space")
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_event_clusters_map(
    df: pd.DataFrame,
    output_path: Path,
    sample_size: int,
    random_state: int,
) -> None:
    sample_idx = choose_sample_indices(len(df), sample_size, random_state + 777)
    plot_df = df.iloc[sample_idx].copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    scatter = ax.scatter(
        plot_df["longitude"],
        plot_df["latitude"],
        c=plot_df["event_cluster"],
        cmap="tab10",
        s=6,
        alpha=0.55,
    )
    ax.set_title("Global Distribution of Event Clusters")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.2, linestyle="--")
    fig.colorbar(scatter, ax=ax, label="Event cluster")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_region_clusters_map(region_summary: pd.DataFrame, output_path: Path) -> None:
    plot_df = region_summary.copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    sparse = plot_df["region_cluster"] < 0
    if sparse.any():
        ax.scatter(
            plot_df.loc[sparse, "region_lon_center"],
            plot_df.loc[sparse, "region_lat_center"],
            s=np.clip(np.log1p(plot_df.loc[sparse, "event_count"]) * 5.0, 5.0, 90.0),
            color="lightgray",
            alpha=0.4,
            label="Sparse regions",
        )
    dense = ~sparse
    scatter = ax.scatter(
        plot_df.loc[dense, "region_lon_center"],
        plot_df.loc[dense, "region_lat_center"],
        s=np.clip(np.log1p(plot_df.loc[dense, "event_count"]) * 7.0, 10.0, 120.0),
        c=plot_df.loc[dense, "region_cluster"],
        cmap="tab10",
        alpha=0.75,
    )
    ax.set_title("Region Clusters by Grid Cell")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.2, linestyle="--")
    if dense.any():
        fig.colorbar(scatter, ax=ax, label="Region cluster")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_top_regions(region_summary: pd.DataFrame, top_n: int, output_path: Path) -> None:
    top_df = region_summary.head(top_n).sort_values("event_count", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    sns.barplot(data=top_df, x="event_count", y="region_code", hue="region_cluster", dodge=False, ax=ax)
    ax.set_title(f"Top {top_n} Regions by Event Count")
    ax.set_xlabel("Event count")
    ax.set_ylabel("Region code")
    ax.legend(title="Region cluster", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_region_profile_heatmap(region_profile: pd.DataFrame, output_path: Path) -> None:
    heatmap_cols = [
        "event_count_log1p",
        "profile_mag_mean",
        "profile_mag_p90",
        "profile_mag_max",
        "profile_depth_mean",
        "profile_sig_mean",
        "profile_major_quake_ratio",
        "profile_shallow_ratio",
        "profile_deep_ratio",
        "profile_tsunami_rate",
    ]
    heatmap_df = region_profile.set_index("region_cluster")[heatmap_cols]
    fig, ax = plt.subplots(figsize=(11, max(4, len(heatmap_df) * 0.65)))
    sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap="YlOrBr", ax=ax)
    ax.set_title("Region Cluster Feature Profile")
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
        cells = [str(value) for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_report(
    output_path: Path,
    raw_df: pd.DataFrame,
    event_df: pd.DataFrame,
    feature_overview: pd.DataFrame,
    event_k_eval: pd.DataFrame,
    region_k_eval: pd.DataFrame,
    event_profile: pd.DataFrame,
    region_profile: pd.DataFrame,
    region_summary: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    used_features = ", ".join(CLUSTER_FEATURES)
    excluded_features = ", ".join(HIGH_MISSING_COLUMNS)
    added_fields = ", ".join(
        [
            "year",
            "month",
            "hour",
            "month_sin",
            "month_cos",
            "hour_sin",
            "hour_cos",
            "depth_log1p",
            "sig_log1p",
            "rms_log1p",
            "nst_log1p",
            "dmin_log1p",
            "depth_band",
            "mag_band",
            "region_id",
            "region_code",
            "region_lat_center",
            "region_lon_center",
        ]
    )
    best_event_k = int(event_k_eval.sort_values(["silhouette", "inertia"], ascending=[False, True]).iloc[0]["k"])
    best_region_k = int(region_k_eval.sort_values(["silhouette", "inertia"], ascending=[False, True]).iloc[0]["k"])

    top_regions = region_summary.head(min(args.top_regions, 10))[
        ["region_id", "region_code", "event_count", "mag_mean", "mag_max", "region_cluster"]
    ].copy()
    top_regions["mag_mean"] = top_regions["mag_mean"].round(3)
    top_regions["mag_max"] = top_regions["mag_max"].round(3)

    event_cluster_table = event_profile.head(8)[
        ["event_cluster", "event_count", "mag_mean", "depth_mean", "sig_mean", "top_mag_type", "top_depth_band"]
    ].copy()
    event_cluster_table["mag_mean"] = event_cluster_table["mag_mean"].round(3)
    event_cluster_table["depth_mean"] = event_cluster_table["depth_mean"].round(3)
    event_cluster_table["sig_mean"] = event_cluster_table["sig_mean"].round(3)

    region_cluster_table = region_profile[
        [
            "region_cluster",
            "region_count",
            "total_events",
            "profile_mag_mean",
            "profile_depth_mean",
            "profile_major_quake_ratio",
            "profile_tsunami_rate",
        ]
    ].copy()
    region_cluster_table = region_cluster_table.rename(
        columns={
            "profile_mag_mean": "mag_mean",
            "profile_depth_mean": "depth_mean",
            "profile_major_quake_ratio": "major_quake_ratio",
            "profile_tsunami_rate": "tsunami_rate",
        }
    )
    region_cluster_table["mag_mean"] = region_cluster_table["mag_mean"].round(3)
    region_cluster_table["depth_mean"] = region_cluster_table["depth_mean"].round(3)
    region_cluster_table["major_quake_ratio"] = region_cluster_table["major_quake_ratio"].round(4)
    region_cluster_table["tsunami_rate"] = region_cluster_table["tsunami_rate"].round(4)

    low_missing = feature_overview[feature_overview["column"].isin(LOW_MISSING_COLUMNS + ["mag", "depth", "sig"])]
    low_missing = low_missing[["column", "missing_ratio"]].copy()
    low_missing["missing_ratio"] = low_missing["missing_ratio"].round(4)

    report = f"""# Pattern Discovering Report

## Scope

- Input file: `{args.input_csv}`
- Filtered event type: `{args.event_type}`
- Rows after filtering: `{len(event_df):,}`
- Distinct regions with grid size `{args.region_grid_size}` degree: `{region_summary["region_id"].nunique():,}`

## Feature Strategy

- Event clustering uses 8 core features: `{used_features}`
- Added analytical fields: `{added_fields}`
- Excluded from clustering because missing ratio is too high: `{excluded_features}`
- Scaling strategy: `log1p` for skewed numeric features, median imputation for `{", ".join(LOW_MISSING_COLUMNS)}`, then `RobustScaler(10, 90)`
- Region strategy: split latitude/longitude into fixed grid cells and assign `region_id` plus `region_code`

### Missingness of retained numeric fields

{markdown_table(low_missing)}

## Clustering Decision

- Best event-level `k`: `{best_event_k}`
- Best region-level `k`: `{best_region_k}`
- Sparse regions below `{args.min_events_per_region}` events keep `region_cluster = -1`

### Event cluster summary

{markdown_table(event_cluster_table)}

### Region cluster summary

{markdown_table(region_cluster_table)}

## Top Regions

{markdown_table(top_regions)}

## Files Generated

- `00_feature_overview.csv`
- `01_event_cluster_assignments.csv`
- `02_event_cluster_k_eval.csv`
- `03_event_cluster_centroids.csv`
- `04_event_cluster_profile.csv`
- `05_region_lookup.csv`
- `06_region_summary.csv`
- `07_region_cluster_k_eval.csv`
- `08_region_cluster_profile.csv`
- `09_pipeline_metadata.json`
- `10_report.md`
- `11_missingness.png`
- `12_core_feature_correlation.png`
- `13_event_k_selection.png`
- `14_event_clusters_pca.png`
- `15_event_clusters_map.png`
- `16_region_k_selection.png`
- `17_region_clusters_map.png`
- `18_top_regions.png`
- `19_region_cluster_profile_heatmap.png`

## Notes

- `mmi`, `cdi`, `felt` are still preserved in the raw dataframe for descriptive analysis, but not used for clustering because their missing ratio is above 96%.
- The clustering task is separated from preprocessing, so this pipeline reads `dongdat.csv` directly and creates only analysis-ready outputs.
"""
    output_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_output_dir(args.output_dir)
    sns.set_theme(style="whitegrid")

    raw_df = load_raw_dataset(args.input_csv, event_type=args.event_type)
    event_df, region_lookup = add_engineered_fields(raw_df, region_grid_size=args.region_grid_size)
    feature_overview = build_feature_overview(event_df)

    event_features, event_x_scaled, imputer, event_scaler = prepare_event_matrix(event_df)
    event_k_eval, train_idx = evaluate_event_k_values(
        event_x_scaled,
        k_min=args.event_k_min,
        k_max=args.event_k_max,
        train_sample_size=args.event_sample_size,
        eval_sample_size=args.eval_sample_size,
        random_state=args.random_state,
    )
    best_event_k = int(
        event_k_eval.sort_values(["silhouette", "inertia"], ascending=[False, True]).iloc[0]["k"]
    )
    event_labels, event_model = fit_event_clustering(
        event_x_scaled,
        train_idx=train_idx,
        best_k=best_event_k,
        random_state=args.random_state,
    )
    event_df["event_cluster"] = event_labels
    event_centroids = restore_event_centroids(event_model, event_scaler, CLUSTER_FEATURES)
    event_profile = summarize_event_clusters(event_df, event_centroids)

    region_summary = build_region_summary(event_df, region_lookup)
    region_k_eval, eligible_regions, region_scaler = evaluate_region_k_values(
        region_summary,
        min_events_per_region=args.min_events_per_region,
        k_min=args.region_k_min,
        k_max=args.region_k_max,
        random_state=args.random_state,
    )
    best_region_k = int(
        region_k_eval.sort_values(["silhouette", "inertia"], ascending=[False, True]).iloc[0]["k"]
    )
    region_summary, region_profile = fit_region_clustering(
        region_summary,
        eligible=eligible_regions,
        scaler=region_scaler,
        best_k=best_region_k,
        random_state=args.random_state,
    )
    event_df = event_df.merge(region_summary[["region_id", "region_cluster"]], on="region_id", how="left")

    metadata = {
        "input_csv": str(args.input_csv),
        "output_dir": str(args.output_dir),
        "rows_after_filtering": int(len(event_df)),
        "event_type": args.event_type,
        "region_grid_size": args.region_grid_size,
        "core_event_cluster_features": CLUSTER_FEATURES,
        "region_cluster_features": REGION_CLUSTER_FEATURES,
        "excluded_high_missing_features": HIGH_MISSING_COLUMNS,
        "median_imputation_features": LOW_MISSING_COLUMNS,
        "best_event_k": best_event_k,
        "best_region_k": best_region_k,
        "region_count": int(region_summary["region_id"].nunique()),
        "eligible_region_count": int((region_summary["region_cluster"] >= 0).sum()),
        "sparse_region_count": int((region_summary["region_cluster"] < 0).sum()),
        "event_feature_fill_values": {
            column: float(value)
            for column, value in zip(CLUSTER_FEATURES, imputer.statistics_)
        },
    }

    assignments = event_df[
        [
            "id",
            "time",
            "latitude",
            "longitude",
            "depth",
            "mag",
            "sig",
            "tsunami",
            "magType",
            "place",
            "depth_band",
            "mag_band",
            "region_id",
            "region_code",
            "region_lat_center",
            "region_lon_center",
            "event_cluster",
            "region_cluster",
        ]
    ].copy()

    feature_overview.to_csv(args.output_dir / "00_feature_overview.csv", index=False)
    assignments.to_csv(args.output_dir / "01_event_cluster_assignments.csv", index=False)
    event_k_eval.to_csv(args.output_dir / "02_event_cluster_k_eval.csv", index=False)
    event_centroids.to_csv(args.output_dir / "03_event_cluster_centroids.csv", index=False)
    event_profile.to_csv(args.output_dir / "04_event_cluster_profile.csv", index=False)
    region_lookup.to_csv(args.output_dir / "05_region_lookup.csv", index=False)
    region_summary.to_csv(args.output_dir / "06_region_summary.csv", index=False)
    region_k_eval.to_csv(args.output_dir / "07_region_cluster_k_eval.csv", index=False)
    region_profile.to_csv(args.output_dir / "08_region_cluster_profile.csv", index=False)
    (args.output_dir / "09_pipeline_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    plot_missingness(raw_df, args.output_dir / "11_missingness.png")
    plot_correlation(event_df, args.output_dir / "12_core_feature_correlation.png")
    plot_k_eval(event_k_eval, args.output_dir / "13_event_k_selection.png", "Event clustering")
    plot_event_clusters_pca(
        event_x_scaled,
        event_labels,
        args.output_dir / "14_event_clusters_pca.png",
        sample_size=args.plot_sample_size,
        random_state=args.random_state,
    )
    plot_event_clusters_map(
        event_df,
        args.output_dir / "15_event_clusters_map.png",
        sample_size=args.plot_sample_size,
        random_state=args.random_state,
    )
    plot_k_eval(region_k_eval, args.output_dir / "16_region_k_selection.png", "Region clustering")
    plot_region_clusters_map(region_summary, args.output_dir / "17_region_clusters_map.png")
    plot_top_regions(region_summary, args.top_regions, args.output_dir / "18_top_regions.png")
    plot_region_profile_heatmap(region_profile, args.output_dir / "19_region_cluster_profile_heatmap.png")
    build_report(
        args.output_dir / "10_report.md",
        raw_df=raw_df,
        event_df=event_df,
        feature_overview=feature_overview,
        event_k_eval=event_k_eval,
        region_k_eval=region_k_eval,
        event_profile=event_profile,
        region_profile=region_profile,
        region_summary=region_summary,
        args=args,
    )


if __name__ == "__main__":
    main()
