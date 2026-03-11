import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import folium
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PHYSICAL_FEATURES = ["mag", "depth", "gap", "nst", "rms"]
LOCATION_FEATURES = ["latitude", "longitude"]
ALL_CLUSTER_FEATURES = PHYSICAL_FEATURES + LOCATION_FEATURES
EPS = 1e-12


def load_dataset(input_csv: Path, max_rows: int, random_state: int) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)

    required = {"id", "time", "latitude", "longitude", "depth", "mag", "gap", "nst", "rms"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    for col in ["latitude", "longitude", "depth", "mag", "gap", "nst", "rms"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["id", "time", "latitude", "longitude", "depth", "mag"]).copy()
    df = df.sort_values("time").reset_index(drop=True)

    if max_rows > 0 and len(df) > max_rows:
        df = (
            df.sample(n=max_rows, random_state=random_state)
            .sort_values("time")
            .reset_index(drop=True)
        )

    return df


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def zscore_standardize(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std < EPS, 1.0, std)
    scaled = (values - mean) / std
    return scaled, mean, std


def kmeans_numpy(
    x: np.ndarray,
    k: int,
    random_state: int,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, float]:
    if k <= 1:
        raise ValueError("k must be >= 2")
    if k > len(x):
        raise ValueError("k cannot exceed number of rows")

    rng = np.random.default_rng(random_state)
    init_idx = rng.choice(len(x), size=k, replace=False)
    centroids = x[init_idx].copy()
    labels = np.zeros(len(x), dtype=np.int32)

    for _ in range(max_iter):
        distances = np.linalg.norm(x[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = distances.argmin(axis=1)

        new_centroids = np.empty_like(centroids)
        for cluster_id in range(k):
            member_mask = new_labels == cluster_id
            if np.any(member_mask):
                new_centroids[cluster_id] = x[member_mask].mean(axis=0)
            else:
                new_centroids[cluster_id] = x[rng.integers(0, len(x))]

        shift = np.linalg.norm(new_centroids - centroids)
        labels = new_labels
        centroids = new_centroids
        if shift <= tol:
            break

    inertia = float(np.sum((x - centroids[labels]) ** 2))
    return labels, centroids, inertia


def evaluate_k_values(
    x: np.ndarray,
    k_min: int,
    k_max: int,
    random_state: int,
) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    overall_mean = x.mean(axis=0)

    for k in range(k_min, k_max + 1):
        labels, centroids, inertia = kmeans_numpy(x, k=k, random_state=random_state + k)
        between_ss = 0.0
        for cluster_id in range(k):
            mask = labels == cluster_id
            n_cluster = int(mask.sum())
            if n_cluster == 0:
                continue
            diff = centroids[cluster_id] - overall_mean
            between_ss += float(n_cluster * np.dot(diff, diff))

        score = between_ss / (inertia + EPS)
        records.append(
            {
                "k": k,
                "between_ss": between_ss,
                "within_ss": inertia,
                "separation_score": score,
            }
        )

    return pd.DataFrame(records).sort_values("k").reset_index(drop=True)


def run_physical_clustering(
    df: pd.DataFrame,
    output_dir: Path,
    k_min: int,
    k_max: int,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feature_df = df[PHYSICAL_FEATURES].dropna().copy()
    x_scaled, _, _ = zscore_standardize(feature_df.to_numpy(dtype=float))

    k_eval = evaluate_k_values(x_scaled, k_min=k_min, k_max=k_max, random_state=random_state)
    best_row = k_eval.sort_values("separation_score", ascending=False).iloc[0]
    best_k = int(best_row["k"])

    labels, centroids, inertia = kmeans_numpy(
        x_scaled, k=best_k, random_state=random_state + 100
    )

    feature_df["cluster_physical"] = labels
    cluster_profile = (
        feature_df.groupby("cluster_physical")
        .agg(
            count=("mag", "size"),
            mag_mean=("mag", "mean"),
            depth_mean=("depth", "mean"),
            gap_mean=("gap", "mean"),
            nst_mean=("nst", "mean"),
            rms_mean=("rms", "mean"),
        )
        .sort_values("count", ascending=False)
        .reset_index()
    )

    meta = {
        "best_k": best_k,
        "best_within_ss": inertia,
        "centroids_scaled": centroids.tolist(),
    }
    (output_dir / "physical_cluster_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    k_eval.to_csv(output_dir / "physical_cluster_k_eval.csv", index=False)
    cluster_profile.to_csv(output_dir / "physical_cluster_profile.csv", index=False)

    # Map cluster labels back to original rows.
    clustered = df.copy()
    clustered["cluster_physical"] = -1
    clustered.loc[feature_df.index, "cluster_physical"] = labels

    return clustered, cluster_profile


def run_spatial_physical_clustering(
    df: pd.DataFrame,
    output_dir: Path,
    k_min: int,
    k_max: int,
    random_state: int,
    plot_sample_size: int,
) -> pd.DataFrame:
    feature_df = df[ALL_CLUSTER_FEATURES].dropna().copy()
    x_scaled, _, _ = zscore_standardize(feature_df.to_numpy(dtype=float))

    k_eval = evaluate_k_values(
        x_scaled, k_min=k_min, k_max=k_max, random_state=random_state + 500
    )
    best_k = int(k_eval.sort_values("separation_score", ascending=False).iloc[0]["k"])
    labels, _, _ = kmeans_numpy(x_scaled, k=best_k, random_state=random_state + 700)

    feature_df["cluster_spatial_physical"] = labels
    out_df = df.copy()
    out_df["cluster_spatial_physical"] = -1
    out_df.loc[feature_df.index, "cluster_spatial_physical"] = labels

    k_eval.to_csv(output_dir / "spatial_physical_cluster_k_eval.csv", index=False)
    (
        feature_df.groupby("cluster_spatial_physical")
        .agg(
            count=("mag", "size"),
            mag_mean=("mag", "mean"),
            depth_mean=("depth", "mean"),
            lat_mean=("latitude", "mean"),
            lon_mean=("longitude", "mean"),
        )
        .sort_values("count", ascending=False)
        .reset_index()
        .to_csv(output_dir / "spatial_physical_cluster_profile.csv", index=False)
    )

    plot_df = out_df[out_df["cluster_spatial_physical"] >= 0]
    if len(plot_df) > plot_sample_size:
        plot_df = plot_df.sample(n=plot_sample_size, random_state=random_state)

    fig, ax = plt.subplots(figsize=(12, 6))
    scatter = ax.scatter(
        plot_df["longitude"],
        plot_df["latitude"],
        c=plot_df["cluster_spatial_physical"],
        cmap="tab20",
        s=8,
        alpha=0.7,
    )
    ax.set_title("Spatial-Physical Clusters (sampled)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.2, linestyle="--")
    fig.colorbar(scatter, ax=ax, label="Cluster ID")
    fig.tight_layout()
    fig.savefig(output_dir / "01_spatial_physical_clusters.png", dpi=180)
    plt.close(fig)

    return out_df


def add_grid_cells(df: pd.DataFrame, grid_size_deg: float) -> pd.DataFrame:
    grid_df = df.copy()
    grid_df["lat_cell"] = np.floor((grid_df["latitude"] + 90.0) / grid_size_deg).astype(int)
    grid_df["lon_cell"] = np.floor((grid_df["longitude"] + 180.0) / grid_size_deg).astype(int)
    grid_df["grid_id"] = grid_df["lat_cell"].astype(str) + "_" + grid_df["lon_cell"].astype(str)
    grid_df["grid_lat_center"] = (grid_df["lat_cell"] + 0.5) * grid_size_deg - 90.0
    grid_df["grid_lon_center"] = (grid_df["lon_cell"] + 0.5) * grid_size_deg - 180.0
    return grid_df


def detect_hotspots(
    df: pd.DataFrame,
    output_dir: Path,
    grid_size_deg: float,
    hotspot_quantile: float,
) -> Tuple[pd.DataFrame, Set[str]]:
    grid_df = add_grid_cells(df, grid_size_deg=grid_size_deg)

    hotspot_stats = (
        grid_df.groupby(["grid_id", "grid_lat_center", "grid_lon_center"], as_index=False)
        .agg(
            event_count=("id", "count"),
            mean_mag=("mag", "mean"),
            max_mag=("mag", "max"),
            mean_depth=("depth", "mean"),
        )
        .sort_values("event_count", ascending=False)
    )
    threshold = float(hotspot_stats["event_count"].quantile(hotspot_quantile))
    hotspot_stats["is_hotspot"] = hotspot_stats["event_count"] >= threshold
    hotspot_stats.to_csv(output_dir / "02_hotspots.csv", index=False)

    hotspot_cells = set(hotspot_stats.loc[hotspot_stats["is_hotspot"], "grid_id"].tolist())

    center_lat = float(df["latitude"].mean())
    center_lon = float(df["longitude"].mean())
    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=2,
        tiles="CartoDB positron",
        control_scale=True,
    )

    heat_points = hotspot_stats[["grid_lat_center", "grid_lon_center", "event_count"]].values.tolist()
    HeatMap(heat_points, radius=18, blur=12, max_zoom=4, min_opacity=0.35).add_to(fmap)

    top_hotspots = hotspot_stats.head(50)
    for row in top_hotspots.itertuples(index=False):
        folium.CircleMarker(
            location=[row.grid_lat_center, row.grid_lon_center],
            radius=4,
            color="#b10026",
            fill=True,
            fill_opacity=0.8,
            popup=(
                f"grid_id={row.grid_id}<br>"
                f"count={row.event_count}<br>"
                f"mean_mag={row.mean_mag:.2f}<br>"
                f"max_mag={row.max_mag:.2f}"
            ),
        ).add_to(fmap)

    fmap.save(output_dir / "02_hotspots_map.html")
    return hotspot_stats, hotspot_cells


def assign_event_token(df: pd.DataFrame) -> pd.Series:
    mag_bins = [0, 3.0, 4.5, 6.0, 10.5]
    mag_labels = ["M0_3", "M3_4_5", "M4_5_6", "M6_plus"]
    depth_bins = [0, 70, 300, 800]
    depth_labels = ["shallow", "intermediate", "deep"]

    mag_cat = pd.cut(df["mag"], bins=mag_bins, labels=mag_labels, right=False, include_lowest=True)
    depth_cat = pd.cut(
        df["depth"],
        bins=depth_bins,
        labels=depth_labels,
        right=False,
        include_lowest=True,
    )
    token = mag_cat.astype("string").fillna("M_unknown") + "_" + depth_cat.astype("string").fillna(
        "depth_unknown"
    )
    return token


def mine_temporal_patterns(df: pd.DataFrame, output_dir: Path, top_n: int) -> pd.DataFrame:
    seq_df = df[["time", "mag", "depth"]].copy()
    seq_df["day"] = seq_df["time"].dt.floor("D")
    seq_df["token"] = assign_event_token(seq_df)

    day_token = (
        seq_df.groupby(["day", "token"])
        .size()
        .reset_index(name="count")
        .sort_values(["day", "count"], ascending=[True, False])
        .drop_duplicates("day")
        .sort_values("day")
    )
    sequence = day_token["token"].tolist()

    if len(sequence) < 4:
        raise ValueError("Not enough temporal sequence to mine patterns.")

    records: List[Dict[str, object]] = []
    n_days = len(sequence)

    for length in [2, 3]:
        if n_days < length:
            continue
        counter: Counter = Counter()
        for i in range(n_days - length + 1):
            key = tuple(sequence[i : i + length])
            counter[key] += 1

        for pattern, cnt in counter.most_common(top_n):
            records.append(
                {
                    "pattern_len": length,
                    "pattern": " -> ".join(pattern),
                    "count": cnt,
                    "support": cnt / max(1, n_days - length + 1),
                }
            )

    pattern_df = (
        pd.DataFrame(records)
        .sort_values(["pattern_len", "count"], ascending=[True, False])
        .reset_index(drop=True)
    )
    pattern_df.to_csv(output_dir / "03_temporal_patterns.csv", index=False)

    # Transition heatmap for 1-step transitions.
    states = sorted(day_token["token"].unique().tolist())
    transition = pd.DataFrame(0, index=states, columns=states, dtype=float)
    for i in range(len(sequence) - 1):
        transition.loc[sequence[i], sequence[i + 1]] += 1.0

    row_sums = transition.sum(axis=1).replace(0, 1.0)
    transition_prob = transition.div(row_sums, axis=0)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        transition_prob,
        cmap="YlGnBu",
        annot=True,
        fmt=".2f",
        cbar=True,
        ax=ax,
    )
    ax.set_title("Daily Dominant Token Transition Probabilities")
    ax.set_xlabel("Next day token")
    ax.set_ylabel("Current day token")
    fig.tight_layout()
    fig.savefig(output_dir / "03_temporal_transition_heatmap.png", dpi=180)
    plt.close(fig)

    return pattern_df


def haversine_distance_km(
    lat1: float,
    lon1: float,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    r = 6371.0
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a)))
    return r * c


def classify_shock_roles(
    df: pd.DataFrame,
    output_dir: Path,
    mainshock_threshold: float,
    shock_radius_km: float,
    foreshock_days: int,
    aftershock_days: int,
) -> pd.DataFrame:
    work_df = df.sort_values("time").reset_index(drop=True).copy()
    times_ns = work_df["time"].astype("int64").to_numpy()
    lat = work_df["latitude"].to_numpy(dtype=float)
    lon = work_df["longitude"].to_numpy(dtype=float)
    mag = work_df["mag"].to_numpy(dtype=float)
    ids = work_df["id"].astype("string").to_numpy()

    role = np.full(len(work_df), "unclassified", dtype=object)
    related_mainshock = np.full(len(work_df), "", dtype=object)
    related_time_diff_hours = np.full(len(work_df), np.inf, dtype=float)

    mainshock_idx = np.where(mag >= mainshock_threshold)[0]
    foreshock_ns = np.int64(foreshock_days) * 24 * 3600 * 1_000_000_000
    aftershock_ns = np.int64(aftershock_days) * 24 * 3600 * 1_000_000_000

    for idx in mainshock_idx:
        role[idx] = "mainshock"
        related_mainshock[idx] = str(ids[idx])
        related_time_diff_hours[idx] = 0.0

    for idx in mainshock_idx:
        t0 = times_ns[idx]
        left = int(np.searchsorted(times_ns, t0 - foreshock_ns, side="left"))
        right = int(np.searchsorted(times_ns, t0 + aftershock_ns, side="right"))
        if right - left <= 1:
            continue

        window_indices = np.arange(left, right)
        distances = haversine_distance_km(lat[idx], lon[idx], lat[window_indices], lon[window_indices])
        within_radius = distances <= shock_radius_km
        smaller_mag = mag[window_indices] <= mag[idx] + EPS
        valid = within_radius & smaller_mag
        candidate_idx = window_indices[valid]

        for target in candidate_idx:
            if target == idx:
                continue
            if role[target] == "mainshock":
                continue

            dt_hours = abs(float(times_ns[target] - t0)) / (3600 * 1e9)
            if dt_hours >= related_time_diff_hours[target]:
                continue

            if times_ns[target] < t0:
                role[target] = "foreshock"
                related_mainshock[target] = str(ids[idx])
                related_time_diff_hours[target] = dt_hours
            elif times_ns[target] > t0:
                role[target] = "aftershock"
                related_mainshock[target] = str(ids[idx])
                related_time_diff_hours[target] = dt_hours

    out = work_df.copy()
    out["shock_role"] = role
    out["related_mainshock_id"] = related_mainshock
    out["time_diff_to_mainshock_hours"] = np.where(
        np.isfinite(related_time_diff_hours), related_time_diff_hours, np.nan
    )
    out.to_csv(output_dir / "04_shock_classification.csv", index=False)
    return out


def robust_zscore(values: np.ndarray) -> np.ndarray:
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    if mad < EPS:
        return np.zeros_like(values)
    return 0.6745 * (values - med) / mad


def detect_outliers(
    df: pd.DataFrame,
    output_dir: Path,
    grid_size_deg: float,
    outlier_quantile: float,
    plot_sample_size: int,
    random_state: int,
) -> pd.DataFrame:
    out_df = add_grid_cells(df, grid_size_deg=grid_size_deg).copy()
    out_df = out_df.sort_values("time").reset_index(drop=True)

    feature_score = np.zeros(len(out_df), dtype=float)
    for col in PHYSICAL_FEATURES:
        z = robust_zscore(out_df[col].to_numpy(dtype=float))
        feature_score += np.abs(z)

    grid_counts = out_df["grid_id"].value_counts()
    count_val = out_df["grid_id"].map(grid_counts).to_numpy(dtype=float)
    rarity = 1.0 - (count_val - count_val.min()) / (count_val.max() - count_val.min() + EPS)

    delta_hours = (
        out_df["time"].diff().dt.total_seconds().fillna(0.0).clip(lower=0.0).to_numpy() / 3600.0
    )
    temporal_z = np.abs(robust_zscore(np.log1p(delta_hours)))

    outlier_score = feature_score + 2.0 * rarity + temporal_z
    threshold = float(np.quantile(outlier_score, outlier_quantile))
    out_df["outlier_score"] = outlier_score
    out_df["is_outlier"] = outlier_score >= threshold

    outlier_rows = out_df[out_df["is_outlier"]].sort_values("outlier_score", ascending=False)
    outlier_rows.to_csv(output_dir / "05_outliers.csv", index=False)

    plot_df = out_df
    if len(plot_df) > plot_sample_size:
        plot_df = plot_df.sample(n=plot_sample_size, random_state=random_state)
    plot_outlier = plot_df[plot_df["is_outlier"]]
    plot_normal = plot_df[~plot_df["is_outlier"]]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(plot_normal["longitude"], plot_normal["latitude"], s=6, alpha=0.2, c="#457b9d", label="Normal")
    ax.scatter(plot_outlier["longitude"], plot_outlier["latitude"], s=16, alpha=0.8, c="#d00000", label="Outlier")
    ax.set_title("Outlier Earthquakes by Location (sampled)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "05_outliers_map.png", dpi=180)
    plt.close(fig)

    return out_df


def qcut_label(values: pd.Series, q: int, prefix: str) -> pd.Series:
    try:
        labels = [f"{prefix}_q{i + 1}" for i in range(q)]
        return pd.qcut(values, q=q, labels=labels, duplicates="drop").astype("string")
    except ValueError:
        return pd.Series([f"{prefix}_q1"] * len(values), index=values.index, dtype="string")


def build_transactions(df: pd.DataFrame, hotspot_cells: Set[str]) -> List[Set[str]]:
    tx_df = add_grid_cells(df, grid_size_deg=5.0).copy()

    tx_df["mag_bin"] = pd.cut(
        tx_df["mag"],
        bins=[0, 3.0, 4.5, 6.0, 10.5],
        labels=["mag_low", "mag_moderate", "mag_strong", "mag_major"],
        include_lowest=True,
        right=False,
    ).astype("string")
    tx_df["depth_bin"] = pd.cut(
        tx_df["depth"],
        bins=[0, 70, 300, 800],
        labels=["depth_shallow", "depth_intermediate", "depth_deep"],
        include_lowest=True,
        right=False,
    ).astype("string")

    tx_df["gap_bin"] = qcut_label(tx_df["gap"], q=3, prefix="gap")
    tx_df["nst_bin"] = qcut_label(tx_df["nst"], q=3, prefix="nst")
    tx_df["rms_bin"] = qcut_label(tx_df["rms"], q=3, prefix="rms")

    tx_df["lat_zone"] = pd.cut(
        tx_df["latitude"],
        bins=[-90, -60, -30, 0, 30, 60, 90],
        labels=["lat_-90_-60", "lat_-60_-30", "lat_-30_0", "lat_0_30", "lat_30_60", "lat_60_90"],
        include_lowest=True,
    ).astype("string")
    tx_df["lon_zone"] = pd.cut(
        tx_df["longitude"],
        bins=[-180, -120, -60, 0, 60, 120, 180],
        labels=[
            "lon_-180_-120",
            "lon_-120_-60",
            "lon_-60_0",
            "lon_0_60",
            "lon_60_120",
            "lon_120_180",
        ],
        include_lowest=True,
    ).astype("string")
    tx_df["hotspot_flag"] = np.where(tx_df["grid_id"].isin(hotspot_cells), "hotspot_yes", "hotspot_no")

    transactions: List[Set[str]] = []
    for row in tx_df.itertuples(index=False):
        items = {
            f"mag:{row.mag_bin}",
            f"depth:{row.depth_bin}",
            f"gap:{row.gap_bin}",
            f"nst:{row.nst_bin}",
            f"rms:{row.rms_bin}",
            f"lat:{row.lat_zone}",
            f"lon:{row.lon_zone}",
            f"hotspot:{row.hotspot_flag}",
        }
        transactions.append(items)
    return transactions


def mine_frequent_itemsets(
    transactions: Sequence[Set[str]],
    max_len: int,
    min_support: float,
) -> Dict[Tuple[str, ...], float]:
    counter: Counter = Counter()
    n_tx = len(transactions)
    if n_tx == 0:
        return {}

    for tx in transactions:
        sorted_items = sorted(tx)
        for size in range(1, max_len + 1):
            for comb in combinations(sorted_items, size):
                counter[comb] += 1

    supports: Dict[Tuple[str, ...], float] = {}
    for itemset, cnt in counter.items():
        support = cnt / n_tx
        if support >= min_support:
            supports[itemset] = support
    return supports


def generate_location_rules(
    supports: Dict[Tuple[str, ...], float],
    min_confidence: float,
    min_lift: float,
) -> pd.DataFrame:
    location_prefixes = ("lat:", "lon:", "hotspot:")
    rules: List[Dict[str, object]] = []

    for itemset, sup_xy in supports.items():
        if len(itemset) < 2:
            continue
        itemset_set = set(itemset)

        for y in itemset:
            if not y.startswith(location_prefixes):
                continue
            x = tuple(sorted(itemset_set - {y}))
            if not x:
                continue
            if any(item.startswith(location_prefixes) for item in x):
                continue
            sup_x = supports.get(x)
            sup_y = supports.get((y,))
            if sup_x is None or sup_y is None or sup_x <= 0 or sup_y <= 0:
                continue

            confidence = sup_xy / sup_x
            lift = confidence / sup_y
            if confidence >= min_confidence and lift >= min_lift:
                rules.append(
                    {
                        "antecedent": " + ".join(x),
                        "consequent": y,
                        "support": sup_xy,
                        "confidence": confidence,
                        "lift": lift,
                    }
                )

    if not rules:
        return pd.DataFrame(columns=["antecedent", "consequent", "support", "confidence", "lift"])
    return pd.DataFrame(rules).sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)


def run_association_mining(
    df: pd.DataFrame,
    output_dir: Path,
    hotspot_cells: Set[str],
    association_max_rows: int,
    min_support: float,
    min_confidence: float,
    min_lift: float,
    random_state: int,
) -> pd.DataFrame:
    tx_df = df.copy()
    if association_max_rows > 0 and len(tx_df) > association_max_rows:
        tx_df = tx_df.sample(n=association_max_rows, random_state=random_state).reset_index(drop=True)

    transactions = build_transactions(tx_df, hotspot_cells=hotspot_cells)
    supports = mine_frequent_itemsets(transactions, max_len=3, min_support=min_support)
    rules = generate_location_rules(
        supports=supports,
        min_confidence=min_confidence,
        min_lift=min_lift,
    )
    rules.to_csv(output_dir / "06_association_rules.csv", index=False)

    itemset_records = [
        {"itemset": " + ".join(itemset), "support": support, "size": len(itemset)}
        for itemset, support in supports.items()
    ]
    pd.DataFrame(itemset_records).sort_values(
        ["size", "support"], ascending=[True, False]
    ).to_csv(output_dir / "06_frequent_itemsets.csv", index=False)
    return rules


def build_report(
    df: pd.DataFrame,
    output_dir: Path,
    cluster_profile: pd.DataFrame,
    hotspots: pd.DataFrame,
    patterns: pd.DataFrame,
    shock_df: pd.DataFrame,
    outlier_df: pd.DataFrame,
    rules: pd.DataFrame,
) -> Path:
    report_path = output_dir / "report.md"

    role_counts = shock_df["shock_role"].value_counts().to_dict()
    outlier_count = int(outlier_df["is_outlier"].sum())
    hotspot_count = int(hotspots["is_hotspot"].sum())

    lines = [
        "# Clustering & Pattern Mining Report",
        "",
        "## Dataset",
        f"- Rows analyzed: {len(df):,}",
        f"- Time range: {df['time'].min()} -> {df['time'].max()}",
        "",
        "## 1) Earthquake Clustering (Physical + Geospatial)",
        f"- Number of physical clusters: {cluster_profile['cluster_physical'].nunique()}",
        "- Cluster profile file: `physical_cluster_profile.csv`",
        "",
        "## 2) Seismic Hotspots",
        f"- Hotspot grid cells: {hotspot_count}",
        "- Hotspot file: `02_hotspots.csv`",
        "- Hotspot map: `02_hotspots_map.html`",
        "",
        "## 3) Temporal Sequence Patterns",
        f"- Pattern rows (top n-grams): {len(patterns):,}",
        "- Pattern file: `03_temporal_patterns.csv`",
        "",
        "## 4) Mainshock / Foreshock / Aftershock Classification",
        f"- mainshock: {role_counts.get('mainshock', 0):,}",
        f"- foreshock: {role_counts.get('foreshock', 0):,}",
        f"- aftershock: {role_counts.get('aftershock', 0):,}",
        f"- unclassified: {role_counts.get('unclassified', 0):,}",
        "- Classification file: `04_shock_classification.csv`",
        "",
        "## 5) Outlier Detection",
        f"- Outlier events: {outlier_count:,}",
        "- Outlier file: `05_outliers.csv`",
        "",
        "## 6) Association Rules (Characteristics -> Location)",
        f"- Rules discovered: {len(rules):,}",
        "- Rules file: `06_association_rules.csv`",
        "",
        "## Output Files",
        "- 01_spatial_physical_clusters.png",
        "- 02_hotspots.csv",
        "- 02_hotspots_map.html",
        "- 03_temporal_patterns.csv",
        "- 03_temporal_transition_heatmap.png",
        "- 04_shock_classification.csv",
        "- 05_outliers.csv",
        "- 05_outliers_map.png",
        "- 06_association_rules.csv",
        "- 06_frequent_itemsets.csv",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run clustering and pattern mining pipeline for earthquake data."
    )
    parser.add_argument("--input-csv", required=True, help="Path to cleaned earthquake CSV.")
    parser.add_argument(
        "--output-dir",
        default="hoigreen/clustering_pattern_mining/outputs",
        help="Directory for outputs.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=220000,
        help="Maximum rows for analysis. Use <=0 for full dataset.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--k-min", type=int, default=3)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--plot-sample-size", type=int, default=30000)
    parser.add_argument("--grid-size-deg", type=float, default=2.5)
    parser.add_argument("--hotspot-quantile", type=float, default=0.95)
    parser.add_argument("--temporal-top-n", type=int, default=30)
    parser.add_argument("--mainshock-threshold", type=float, default=5.5)
    parser.add_argument("--shock-radius-km", type=float, default=120.0)
    parser.add_argument("--foreshock-days", type=int, default=7)
    parser.add_argument("--aftershock-days", type=int, default=30)
    parser.add_argument("--outlier-quantile", type=float, default=0.995)
    parser.add_argument("--association-max-rows", type=int, default=120000)
    parser.add_argument("--min-support", type=float, default=0.01)
    parser.add_argument("--min-confidence", type=float, default=0.35)
    parser.add_argument("--min-lift", type=float, default=1.05)
    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> None:
    sns.set_theme(style="whitegrid")
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)

    df = load_dataset(Path(args.input_csv), max_rows=args.max_rows, random_state=args.random_state)

    physical_clustered, cluster_profile = run_physical_clustering(
        df,
        output_dir=output_dir,
        k_min=args.k_min,
        k_max=args.k_max,
        random_state=args.random_state,
    )
    clustered = run_spatial_physical_clustering(
        physical_clustered,
        output_dir=output_dir,
        k_min=args.k_min,
        k_max=args.k_max,
        random_state=args.random_state,
        plot_sample_size=args.plot_sample_size,
    )
    clustered.to_csv(output_dir / "01_clustered_events.csv", index=False)

    hotspots, hotspot_cells = detect_hotspots(
        clustered,
        output_dir=output_dir,
        grid_size_deg=args.grid_size_deg,
        hotspot_quantile=args.hotspot_quantile,
    )

    patterns = mine_temporal_patterns(
        clustered,
        output_dir=output_dir,
        top_n=args.temporal_top_n,
    )

    shock_df = classify_shock_roles(
        clustered,
        output_dir=output_dir,
        mainshock_threshold=args.mainshock_threshold,
        shock_radius_km=args.shock_radius_km,
        foreshock_days=args.foreshock_days,
        aftershock_days=args.aftershock_days,
    )

    outlier_df = detect_outliers(
        clustered,
        output_dir=output_dir,
        grid_size_deg=args.grid_size_deg,
        outlier_quantile=args.outlier_quantile,
        plot_sample_size=args.plot_sample_size,
        random_state=args.random_state,
    )

    rules = run_association_mining(
        clustered,
        output_dir=output_dir,
        hotspot_cells=hotspot_cells,
        association_max_rows=args.association_max_rows,
        min_support=args.min_support,
        min_confidence=args.min_confidence,
        min_lift=args.min_lift,
        random_state=args.random_state,
    )

    report_path = build_report(
        clustered,
        output_dir=output_dir,
        cluster_profile=cluster_profile,
        hotspots=hotspots,
        patterns=patterns,
        shock_df=shock_df,
        outlier_df=outlier_df,
        rules=rules,
    )

    print("=" * 72)
    print("Clustering & Pattern Mining completed")
    print("=" * 72)
    print(f"Input file: {args.input_csv}")
    print(f"Rows analyzed: {len(clustered):,}")
    print(f"Output directory: {output_dir}")
    print(f"Report: {report_path.name}")
    print("=" * 72)


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
