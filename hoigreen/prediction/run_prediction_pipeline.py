import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from xgboost import XGBRegressor


EPS = 1e-12
EVENT_FEATURE_COLS = ["mag", "depth", "gap", "nst", "rms"]


def format_elapsed(seconds: float) -> str:
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    remain = seconds - 60 * minutes
    return f"{minutes}m {remain:.1f}s"


def print_progress(step: int, total: int, message: str) -> None:
    print(f"[{step}/{total}] {message}")


@dataclass
class StandardScalerNumpy:
    mean_: np.ndarray
    std_: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "StandardScalerNumpy":
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std < EPS, 1.0, std)
        return cls(mean_=mean, std_=std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.std_

    def to_dict(self) -> Dict[str, List[float]]:
        return {
            "mean": self.mean_.tolist(),
            "std": self.std_.tolist(),
        }


@dataclass
class LogisticRegressorGD:
    l2: float
    lr: float
    max_iter: int
    tol: float
    coef_: np.ndarray
    intercept_: float

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        l2: float,
        lr: float,
        max_iter: int,
        tol: float,
    ) -> "LogisticRegressorGD":
        n, d = x.shape
        w = np.zeros(d, dtype=float)
        b = 0.0
        prev_loss = np.inf

        for _ in range(max_iter):
            z = x @ w + b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))

            grad_w = (x.T @ (p - y)) / n + l2 * w
            grad_b = float(np.mean(p - y))

            w -= lr * grad_w
            b -= lr * grad_b

            p_safe = np.clip(p, EPS, 1.0 - EPS)
            loss = (
                -float(np.mean(y * np.log(p_safe) + (1.0 - y) * np.log(1.0 - p_safe)))
                + 0.5 * l2 * float(np.dot(w, w))
            )
            if abs(prev_loss - loss) < tol:
                break
            prev_loss = loss

        return cls(
            l2=l2,
            lr=lr,
            max_iter=max_iter,
            tol=tol,
            coef_=w,
            intercept_=b,
        )

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z = x @ self.coef_ + self.intercept_
        return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))

    def to_dict(self) -> Dict[str, object]:
        return {
            "l2": float(self.l2),
            "lr": float(self.lr),
            "max_iter": int(self.max_iter),
            "tol": float(self.tol),
            "intercept": float(self.intercept_),
            "coef": self.coef_.tolist(),
        }


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), EPS)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    f1 = 2.0 * precision * recall / max(EPS, precision + recall)
    accuracy = (tp + tn) / max(1, len(y_true))
    brier = float(np.mean((y_score - y_true) ** 2))

    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "brier": brier,
        "auc": float(binary_auc(y_true, y_score)),
    }


def binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(int)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = pd.Series(y_score).rank(method="average").to_numpy()
    sum_ranks_pos = float(np.sum(ranks[y_true == 1]))
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def best_threshold_by_f1(y_true: np.ndarray, y_score: np.ndarray) -> float:
    candidates = np.linspace(0.1, 0.9, 81)
    best_t = 0.5
    best_f1 = -1.0
    for t in candidates:
        m = classification_metrics(y_true, y_score, threshold=float(t))
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_t = float(t)
    return best_t


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


def ensure_output_dirs(output_dir: Path) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    figure_dir = output_dir / "figures"
    model_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    return model_dir, figure_dir


def make_event_sequence_dataset(df: pd.DataFrame, lookback: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(df) <= lookback + 1:
        raise ValueError("Not enough rows for event-sequence prediction.")

    mag = df["mag"].to_numpy(dtype=float)
    depth = df["depth"].to_numpy(dtype=float)
    gap = df["gap"].to_numpy(dtype=float)
    nst = df["nst"].to_numpy(dtype=float)
    rms = df["rms"].to_numpy(dtype=float)

    delta_hours_prev = (
        df["time"].diff().dt.total_seconds().fillna(0.0).clip(lower=0.0).to_numpy() / 3600.0
    )
    delta_hours_next = (
        df["time"].shift(-1).sub(df["time"]).dt.total_seconds().fillna(0.0).clip(lower=0.0).to_numpy()
        / 3600.0
    )

    def windows(arr: np.ndarray) -> np.ndarray:
        return np.lib.stride_tricks.sliding_window_view(arr, window_shape=lookback)[:-1]

    w_mag = windows(mag)
    w_depth = windows(depth)
    w_gap = windows(gap)
    w_nst = windows(nst)
    w_rms = windows(rms)
    w_prev_dt = windows(delta_hours_prev)

    mag_mean = w_mag.mean(axis=1, keepdims=True)
    mag_std = w_mag.std(axis=1, keepdims=True)
    depth_mean = w_depth.mean(axis=1, keepdims=True)
    prev_dt_mean = w_prev_dt.mean(axis=1, keepdims=True)
    prev_dt_std = w_prev_dt.std(axis=1, keepdims=True)

    x = np.hstack(
        [
            w_mag,
            w_depth,
            w_gap,
            w_nst,
            w_rms,
            w_prev_dt,
            mag_mean,
            mag_std,
            depth_mean,
            prev_dt_mean,
            prev_dt_std,
        ]
    )

    y_next_mag = mag[lookback:]
    y_next_dt_hours = delta_hours_next[lookback - 1 : -1]
    target_time = df["time"].to_numpy()[lookback:]

    return x, y_next_mag, y_next_dt_hours, target_time


def train_test_split_time(x: np.ndarray, y: np.ndarray, train_ratio: float) -> Tuple[np.ndarray, ...]:
    n = len(x)
    if n < 20:
        raise ValueError("Dataset too small for train/test split.")
    split = int(n * train_ratio)
    split = min(max(split, 10), n - 5)
    return x[:split], x[split:], y[:split], y[split:], np.arange(split), np.arange(split, n)


def train_xgb_regressor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    subsample: float,
    colsample_bytree: float,
    min_child_weight: float,
    reg_lambda: float,
    progress_label: str = "",
    xgb_verbose_every: int = 0,
) -> XGBRegressor:
    start = time.time()
    if progress_label:
        print(f"    -> Training {progress_label} ...")

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        reg_lambda=reg_lambda,
        random_state=random_state,
        eval_metric="rmse",
        n_jobs=-1,
        tree_method="hist",
    )
    if xgb_verbose_every > 0:
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_train, y_train)],
            verbose=xgb_verbose_every,
        )
    else:
        model.fit(x_train, y_train)

    if progress_label:
        print(f"    -> Done {progress_label} in {format_elapsed(time.time() - start)}")
    return model


def run_next_magnitude_forecast(
    df: pd.DataFrame,
    output_dir: Path,
    model_dir: Path,
    figure_dir: Path,
    lookback: int,
    train_ratio: float,
    random_state: int,
    xgb_n_estimators: int,
    xgb_max_depth: int,
    xgb_learning_rate: float,
    xgb_subsample: float,
    xgb_colsample_bytree: float,
    xgb_min_child_weight: float,
    xgb_reg_lambda: float,
    show_progress: bool,
    xgb_verbose_every: int,
) -> Dict[str, float]:
    x, y_mag, _, target_time = make_event_sequence_dataset(df, lookback=lookback)
    x_train, x_test, y_train, y_test, _, test_idx = train_test_split_time(x, y_mag, train_ratio=train_ratio)
    x_train = x_train.astype(np.float32, copy=False)
    x_test = x_test.astype(np.float32, copy=False)
    y_train = y_train.astype(np.float32, copy=False)
    model = train_xgb_regressor(
        x_train=x_train,
        y_train=y_train,
        random_state=random_state + 1001,
        n_estimators=xgb_n_estimators,
        max_depth=xgb_max_depth,
        learning_rate=xgb_learning_rate,
        subsample=xgb_subsample,
        colsample_bytree=xgb_colsample_bytree,
        min_child_weight=xgb_min_child_weight,
        reg_lambda=xgb_reg_lambda,
        progress_label="XGBoost for next-magnitude" if show_progress else "",
        xgb_verbose_every=xgb_verbose_every if show_progress else 0,
    )
    pred_test = model.predict(x_test)

    metrics = {
        "task": "next_magnitude",
        "mae": mae(y_test, pred_test),
        "rmse": rmse(y_test, pred_test),
        "mape": mape(y_test, pred_test),
    }
    model_params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "eval_metric": "rmse",
        "n_estimators": xgb_n_estimators,
        "max_depth": xgb_max_depth,
        "learning_rate": xgb_learning_rate,
        "subsample": xgb_subsample,
        "colsample_bytree": xgb_colsample_bytree,
        "min_child_weight": xgb_min_child_weight,
        "reg_lambda": xgb_reg_lambda,
        "random_state": random_state + 1001,
    }

    pred_df = pd.DataFrame(
        {
            "time": target_time[test_idx],
            "actual_mag": y_test,
            "pred_mag": pred_test,
            "abs_error": np.abs(y_test - pred_test),
        }
    )
    pred_df.to_csv(output_dir / "01_next_magnitude_predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    sample = pred_df.tail(min(800, len(pred_df)))
    ax.plot(sample["time"], sample["actual_mag"], label="Actual", color="#003049", linewidth=1.0)
    ax.plot(sample["time"], sample["pred_mag"], label="Predicted", color="#d62828", linewidth=1.0, alpha=0.9)
    ax.set_title("Next Earthquake Magnitude Forecast (test period)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Magnitude")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "01_next_magnitude_forecast.png", dpi=180)
    plt.close(fig)

    (model_dir / "01_next_magnitude_model.json").write_text(
        json.dumps(
            {
                "model_type": "xgboost_regressor",
                "lookback": lookback,
                "xgb_params": model_params,
                "feature_importance": model.feature_importances_.astype(float).tolist(),
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    model.save_model(str(model_dir / "01_next_magnitude_model.ubj"))
    return metrics


def run_next_time_prediction(
    df: pd.DataFrame,
    output_dir: Path,
    model_dir: Path,
    figure_dir: Path,
    lookback: int,
    train_ratio: float,
    random_state: int,
    xgb_n_estimators: int,
    xgb_max_depth: int,
    xgb_learning_rate: float,
    xgb_subsample: float,
    xgb_colsample_bytree: float,
    xgb_min_child_weight: float,
    xgb_reg_lambda: float,
    show_progress: bool,
    xgb_verbose_every: int,
) -> Dict[str, float]:
    x, _, y_dt_hours, target_time = make_event_sequence_dataset(df, lookback=lookback)
    y_log = np.log1p(np.maximum(y_dt_hours, 0.0))
    x_train, x_test, y_train, y_test, _, test_idx = train_test_split_time(x, y_log, train_ratio=train_ratio)
    x_train = x_train.astype(np.float32, copy=False)
    x_test = x_test.astype(np.float32, copy=False)
    y_train = y_train.astype(np.float32, copy=False)
    model = train_xgb_regressor(
        x_train=x_train,
        y_train=y_train,
        random_state=random_state + 2001,
        n_estimators=xgb_n_estimators,
        max_depth=xgb_max_depth,
        learning_rate=xgb_learning_rate,
        subsample=xgb_subsample,
        colsample_bytree=xgb_colsample_bytree,
        min_child_weight=xgb_min_child_weight,
        reg_lambda=xgb_reg_lambda,
        progress_label="XGBoost for next-time" if show_progress else "",
        xgb_verbose_every=xgb_verbose_every if show_progress else 0,
    )
    pred_log = model.predict(x_test)

    y_test_hours = np.expm1(y_test)
    pred_hours = np.maximum(0.0, np.expm1(pred_log))

    metrics = {
        "task": "next_time_hours",
        "mae_hours": mae(y_test_hours, pred_hours),
        "rmse_hours": rmse(y_test_hours, pred_hours),
        "mape_hours": mape(y_test_hours, pred_hours),
    }
    model_params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "eval_metric": "rmse",
        "n_estimators": xgb_n_estimators,
        "max_depth": xgb_max_depth,
        "learning_rate": xgb_learning_rate,
        "subsample": xgb_subsample,
        "colsample_bytree": xgb_colsample_bytree,
        "min_child_weight": xgb_min_child_weight,
        "reg_lambda": xgb_reg_lambda,
        "random_state": random_state + 2001,
    }

    pred_df = pd.DataFrame(
        {
            "time": target_time[test_idx],
            "actual_next_hours": y_test_hours,
            "pred_next_hours": pred_hours,
            "abs_error_hours": np.abs(y_test_hours - pred_hours),
        }
    )
    pred_df.to_csv(output_dir / "02_next_time_predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    sample = pred_df.sample(n=min(5000, len(pred_df)), random_state=42)
    ax.scatter(sample["actual_next_hours"], sample["pred_next_hours"], s=10, alpha=0.2, color="#2a9d8f")
    lim = float(np.quantile(np.r_[sample["actual_next_hours"], sample["pred_next_hours"]], 0.98))
    lim = max(1.0, lim)
    ax.plot([0, lim], [0, lim], color="#e63946", linewidth=1.5)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_title("Next-event Time Prediction (hours)")
    ax.set_xlabel("Actual next-event time (hours)")
    ax.set_ylabel("Predicted next-event time (hours)")
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(figure_dir / "02_next_time_prediction.png", dpi=180)
    plt.close(fig)

    (model_dir / "02_next_time_model.json").write_text(
        json.dumps(
            {
                "model_type": "xgboost_regressor",
                "lookback": lookback,
                "target_transform": "log1p(hours)",
                "xgb_params": model_params,
                "feature_importance": model.feature_importances_.astype(float).tolist(),
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    model.save_model(str(model_dir / "02_next_time_model.ubj"))
    return metrics


def build_daily_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["day"] = d["time"].dt.floor("D")
    daily = d.groupby("day").agg(
        event_count=("id", "size"),
        mean_mag=("mag", "mean"),
        max_mag=("mag", "max"),
        std_mag=("mag", "std"),
        mean_depth=("depth", "mean"),
    )

    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D", tz="UTC")
    daily = daily.reindex(full_index)
    daily["event_count"] = daily["event_count"].fillna(0.0)
    daily["mean_mag"] = daily["mean_mag"].fillna(0.0)
    daily["max_mag"] = daily["max_mag"].fillna(0.0)
    daily["std_mag"] = daily["std_mag"].fillna(0.0)
    daily["mean_depth"] = daily["mean_depth"].fillna(0.0)
    daily.index.name = "day"
    return daily.reset_index()


def build_daily_features(daily: pd.DataFrame) -> pd.DataFrame:
    feat = daily.copy()
    c = feat["event_count"]
    feat["count_3d"] = c.rolling(3, min_periods=1).sum()
    feat["count_7d"] = c.rolling(7, min_periods=1).sum()
    feat["count_14d"] = c.rolling(14, min_periods=1).sum()
    feat["count_30d"] = c.rolling(30, min_periods=1).sum()
    feat["count_60d"] = c.rolling(60, min_periods=1).sum()

    feat["mean_mag_7d"] = feat["mean_mag"].rolling(7, min_periods=1).mean()
    feat["max_mag_7d"] = feat["max_mag"].rolling(7, min_periods=1).max()
    feat["std_mag_7d"] = feat["mean_mag"].rolling(7, min_periods=1).std().fillna(0.0)
    feat["mean_depth_14d"] = feat["mean_depth"].rolling(14, min_periods=1).mean()

    rate_7 = feat["count_7d"] / 7.0
    rate_30 = feat["count_30d"] / 30.0
    feat["rate_ratio_7_30"] = rate_7 / (rate_30 + EPS)
    feat["silence_score"] = rate_30 - rate_7
    feat["activity_change_3_14"] = (feat["count_3d"] / 3.0) - (feat["count_14d"] / 14.0)
    feat["energy_proxy_7d"] = (feat["mean_mag_7d"] ** 2) * (feat["count_7d"] + 1.0)
    return feat


def label_future_event(
    daily: pd.DataFrame,
    mag_threshold: float,
    horizon_days: int,
) -> np.ndarray:
    major_today = (daily["max_mag"] >= mag_threshold).astype(int).to_numpy()
    y = np.zeros(len(daily), dtype=int)
    for i in range(len(daily)):
        start = i + 1
        end = min(len(daily), i + 1 + horizon_days)
        if start < end and np.any(major_today[start:end] == 1):
            y[i] = 1
    return y


def run_quiet_period_detection(
    df: pd.DataFrame,
    output_dir: Path,
    model_dir: Path,
    figure_dir: Path,
    train_ratio: float,
    major_mag_threshold: float,
    major_horizon_days: int,
    quiet_window_days: int,
    baseline_window_days: int,
    quiet_ratio_threshold: float,
    logreg_l2: float,
    logreg_lr: float,
    logreg_iter: int,
) -> Dict[str, float]:
    daily = build_daily_aggregation(df)
    feat = build_daily_features(daily)
    y = label_future_event(feat, mag_threshold=major_mag_threshold, horizon_days=major_horizon_days)

    feature_cols = [
        "event_count",
        "count_3d",
        "count_7d",
        "count_14d",
        "count_30d",
        "count_60d",
        "mean_mag_7d",
        "max_mag_7d",
        "std_mag_7d",
        "mean_depth_14d",
        "rate_ratio_7_30",
        "silence_score",
        "activity_change_3_14",
        "energy_proxy_7d",
    ]
    x = feat[feature_cols].to_numpy(dtype=float)

    x_train, x_test, y_train, y_test, _, test_idx = train_test_split_time(x, y.astype(float), train_ratio=train_ratio)
    y_train_int = y_train.astype(int)
    y_test_int = y_test.astype(int)

    scaler = StandardScalerNumpy.fit(x_train)
    x_train_s = scaler.transform(x_train)
    x_test_s = scaler.transform(x_test)

    model = LogisticRegressorGD.fit(
        x_train_s,
        y_train_int,
        l2=logreg_l2,
        lr=logreg_lr,
        max_iter=logreg_iter,
        tol=1e-7,
    )
    train_proba = model.predict_proba(x_train_s)
    best_t = best_threshold_by_f1(y_train_int, train_proba)
    test_proba = model.predict_proba(x_test_s)
    metrics = classification_metrics(y_test_int, test_proba, threshold=best_t)
    metrics["task"] = "quiet_period_before_major"
    metrics["positive_rate_test"] = float(np.mean(y_test_int))

    score_df = pd.DataFrame(
        {
            "day": feat["day"].to_numpy()[test_idx],
            "target_major_next_horizon": y_test_int,
            "pred_proba_major_next_horizon": test_proba,
            "pred_label": (test_proba >= best_t).astype(int),
            "event_count": feat["event_count"].to_numpy()[test_idx],
            "silence_score": feat["silence_score"].to_numpy()[test_idx],
            "rate_ratio_7_30": feat["rate_ratio_7_30"].to_numpy()[test_idx],
        }
    )
    score_df.to_csv(output_dir / "03_quiet_period_model_scores.csv", index=False)

    major_today = (feat["max_mag"] >= major_mag_threshold).astype(int).to_numpy()
    quiet_records: List[Dict[str, object]] = []
    for idx in np.where(major_today == 1)[0]:
        q_start = idx - quiet_window_days
        q_end = idx - 1
        b_start = q_start - baseline_window_days
        b_end = q_start - 1
        if b_start < 0 or q_start < 0:
            continue

        quiet_rate = float(np.mean(feat["event_count"].iloc[q_start : q_end + 1]))
        baseline_rate = float(np.mean(feat["event_count"].iloc[b_start : b_end + 1]))
        ratio = quiet_rate / (baseline_rate + EPS)
        is_quiet_period = ratio <= quiet_ratio_threshold

        quiet_records.append(
            {
                "major_day": feat["day"].iloc[idx],
                "major_mag": float(feat["max_mag"].iloc[idx]),
                "quiet_start": feat["day"].iloc[q_start],
                "quiet_end": feat["day"].iloc[q_end],
                "quiet_rate": quiet_rate,
                "baseline_rate": baseline_rate,
                "quiet_to_baseline_ratio": ratio,
                "is_quiet_period": bool(is_quiet_period),
            }
        )

    quiet_df = pd.DataFrame(quiet_records).sort_values("major_day")
    quiet_df.to_csv(output_dir / "03_quiet_periods_detected.csv", index=False)

    fig, ax = plt.subplots(figsize=(13, 5))
    plot_df = score_df.tail(min(len(score_df), 900))
    ax.plot(plot_df["day"], plot_df["pred_proba_major_next_horizon"], color="#6a4c93", linewidth=1.2, label="Pred prob")
    ax.plot(plot_df["day"], plot_df["target_major_next_horizon"], color="#d00000", linewidth=1.0, alpha=0.7, label="Actual label")
    ax.axhline(best_t, color="#0f4c5c", linestyle="--", linewidth=1.1, label=f"Threshold={best_t:.2f}")
    ax.set_title("Quiet-period Model: Probability of Major Earthquake in Next Horizon")
    ax.set_xlabel("Day")
    ax.set_ylabel("Probability / Label")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "03_quiet_period_model_timeline.png", dpi=180)
    plt.close(fig)

    (model_dir / "03_quiet_period_model.json").write_text(
        json.dumps(
            {
                "major_mag_threshold": major_mag_threshold,
                "horizon_days": major_horizon_days,
                "feature_cols": feature_cols,
                "threshold": best_t,
                "scaler": scaler.to_dict(),
                "model": model.to_dict(),
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return metrics


def run_early_warning_model(
    df: pd.DataFrame,
    output_dir: Path,
    model_dir: Path,
    figure_dir: Path,
    train_ratio: float,
    warning_mag_threshold: float,
    warning_horizon_days: int,
    logreg_l2: float,
    logreg_lr: float,
    logreg_iter: int,
) -> Dict[str, float]:
    daily = build_daily_aggregation(df)
    feat = build_daily_features(daily)
    y = label_future_event(feat, mag_threshold=warning_mag_threshold, horizon_days=warning_horizon_days)

    feature_cols = [
        "event_count",
        "count_3d",
        "count_7d",
        "count_14d",
        "mean_mag_7d",
        "max_mag_7d",
        "std_mag_7d",
        "rate_ratio_7_30",
        "activity_change_3_14",
        "energy_proxy_7d",
    ]
    x = feat[feature_cols].to_numpy(dtype=float)

    x_train, x_test, y_train, y_test, _, test_idx = train_test_split_time(x, y.astype(float), train_ratio=train_ratio)
    y_train_int = y_train.astype(int)
    y_test_int = y_test.astype(int)

    scaler = StandardScalerNumpy.fit(x_train)
    x_train_s = scaler.transform(x_train)
    x_test_s = scaler.transform(x_test)

    model = LogisticRegressorGD.fit(
        x_train_s,
        y_train_int,
        l2=logreg_l2,
        lr=logreg_lr,
        max_iter=logreg_iter,
        tol=1e-7,
    )

    train_proba = model.predict_proba(x_train_s)
    best_t = best_threshold_by_f1(y_train_int, train_proba)
    high_t = min(0.95, best_t + 0.18)

    test_proba = model.predict_proba(x_test_s)
    metrics = classification_metrics(y_test_int, test_proba, threshold=best_t)
    metrics["task"] = "early_warning"
    metrics["positive_rate_test"] = float(np.mean(y_test_int))

    risk_level = np.where(
        test_proba >= high_t,
        "high",
        np.where(test_proba >= best_t, "medium", "low"),
    )
    pred_df = pd.DataFrame(
        {
            "day": feat["day"].to_numpy()[test_idx],
            "target_event_next_horizon": y_test_int,
            "warning_probability": test_proba,
            "alert_level": risk_level,
            "event_count": feat["event_count"].to_numpy()[test_idx],
            "max_mag_7d": feat["max_mag_7d"].to_numpy()[test_idx],
            "energy_proxy_7d": feat["energy_proxy_7d"].to_numpy()[test_idx],
        }
    )
    pred_df.to_csv(output_dir / "04_early_warning_predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(13, 5))
    plot_df = pred_df.tail(min(len(pred_df), 900))
    ax.plot(plot_df["day"], plot_df["warning_probability"], color="#005f73", linewidth=1.2, label="Warning probability")
    ax.plot(plot_df["day"], plot_df["target_event_next_horizon"], color="#9b2226", linewidth=1.0, alpha=0.7, label="Actual label")
    ax.axhline(best_t, color="#ee9b00", linestyle="--", linewidth=1.0, label=f"Medium threshold={best_t:.2f}")
    ax.axhline(high_t, color="#ca6702", linestyle="--", linewidth=1.0, label=f"High threshold={high_t:.2f}")
    ax.set_title("Early Warning Model Timeline")
    ax.set_xlabel("Day")
    ax.set_ylabel("Probability / Label")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "04_early_warning_timeline.png", dpi=180)
    plt.close(fig)

    (model_dir / "04_early_warning_model.json").write_text(
        json.dumps(
            {
                "warning_mag_threshold": warning_mag_threshold,
                "horizon_days": warning_horizon_days,
                "feature_cols": feature_cols,
                "medium_threshold": best_t,
                "high_threshold": high_t,
                "scaler": scaler.to_dict(),
                "model": model.to_dict(),
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return metrics


def build_report(
    output_dir: Path,
    mag_metrics: Dict[str, float],
    time_metrics: Dict[str, float],
    quiet_metrics: Dict[str, float],
    warning_metrics: Dict[str, float],
) -> Path:
    report_path = output_dir / "report.md"
    lines = [
        "# Prediction Stage Report",
        "",
        "## 1) Forecast Next Earthquake Magnitude",
        f"- MAE: {mag_metrics['mae']:.4f}",
        f"- RMSE: {mag_metrics['rmse']:.4f}",
        f"- MAPE: {mag_metrics['mape']:.4f}",
        "",
        "## 2) Predict Time Until Next Earthquake",
        f"- MAE (hours): {time_metrics['mae_hours']:.4f}",
        f"- RMSE (hours): {time_metrics['rmse_hours']:.4f}",
        f"- MAPE (hours): {time_metrics['mape_hours']:.4f}",
        "",
        "## 3) Detect Quiet Period Before Major Earthquakes",
        f"- AUC: {quiet_metrics['auc']:.4f}",
        f"- Precision: {quiet_metrics['precision']:.4f}",
        f"- Recall: {quiet_metrics['recall']:.4f}",
        f"- F1: {quiet_metrics['f1']:.4f}",
        "",
        "## 4) Early Warning Model",
        f"- AUC: {warning_metrics['auc']:.4f}",
        f"- Precision: {warning_metrics['precision']:.4f}",
        f"- Recall: {warning_metrics['recall']:.4f}",
        f"- F1: {warning_metrics['f1']:.4f}",
        "",
        "## Generated Files",
        "- 01_next_magnitude_predictions.csv",
        "- 02_next_time_predictions.csv",
        "- 03_quiet_periods_detected.csv",
        "- 03_quiet_period_model_scores.csv",
        "- 04_early_warning_predictions.csv",
        "- figures/01_next_magnitude_forecast.png",
        "- figures/02_next_time_prediction.png",
        "- figures/03_quiet_period_model_timeline.png",
        "- figures/04_early_warning_timeline.png",
        "- models/*.json",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and run earthquake prediction pipeline.")
    parser.add_argument(
        "--input-csv",
        default="hoigreen/preprocessing/outputs/earthquake_cleaned.csv",
        help="Path to cleaned earthquake CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="hoigreen/prediction/outputs",
        help="Directory for predictions, models, and report.",
    )
    parser.add_argument("--max-rows", type=int, default=350000, help="Max rows to use. <=0 for full dataset.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--lookback", type=int, default=8, help="Number of previous events used for next-step prediction.")
    parser.add_argument("--xgb-n-estimators", type=int, default=600)
    parser.add_argument("--xgb-max-depth", type=int, default=6)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-subsample", type=float, default=0.85)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.85)
    parser.add_argument("--xgb-min-child-weight", type=float, default=5.0)
    parser.add_argument("--xgb-reg-lambda", type=float, default=1.0)
    parser.add_argument(
        "--xgb-verbose-every",
        type=int,
        default=100,
        help="Print XGBoost training metric every N boosting rounds. <=0 disables.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress logs during pipeline execution.",
    )

    parser.add_argument("--major-mag-threshold", type=float, default=6.5)
    parser.add_argument("--major-horizon-days", type=int, default=7)
    parser.add_argument("--quiet-window-days", type=int, default=14)
    parser.add_argument("--baseline-window-days", type=int, default=45)
    parser.add_argument("--quiet-ratio-threshold", type=float, default=0.7)

    parser.add_argument("--warning-mag-threshold", type=float, default=5.8)
    parser.add_argument("--warning-horizon-days", type=int, default=3)

    parser.add_argument("--logreg-l2", type=float, default=0.02)
    parser.add_argument("--logreg-lr", type=float, default=0.05)
    parser.add_argument("--logreg-iter", type=int, default=2500)
    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> None:
    sns.set_theme(style="whitegrid")
    total_steps = 4
    pipeline_start = time.time()
    show_progress = not args.no_progress

    output_dir = Path(args.output_dir)
    model_dir, figure_dir = ensure_output_dirs(output_dir)

    df = load_dataset(Path(args.input_csv), max_rows=args.max_rows, random_state=args.random_state)

    if show_progress:
        print_progress(1, total_steps, "Forecast next magnitude")
    step_start = time.time()
    mag_metrics = run_next_magnitude_forecast(
        df,
        output_dir=output_dir,
        model_dir=model_dir,
        figure_dir=figure_dir,
        lookback=args.lookback,
        train_ratio=args.train_ratio,
        random_state=args.random_state,
        xgb_n_estimators=args.xgb_n_estimators,
        xgb_max_depth=args.xgb_max_depth,
        xgb_learning_rate=args.xgb_learning_rate,
        xgb_subsample=args.xgb_subsample,
        xgb_colsample_bytree=args.xgb_colsample_bytree,
        xgb_min_child_weight=args.xgb_min_child_weight,
        xgb_reg_lambda=args.xgb_reg_lambda,
        show_progress=show_progress,
        xgb_verbose_every=args.xgb_verbose_every,
    )
    if show_progress:
        print(f"    Completed step 1 in {format_elapsed(time.time() - step_start)}")

    if show_progress:
        print_progress(2, total_steps, "Predict time until next earthquake")
    step_start = time.time()
    time_metrics = run_next_time_prediction(
        df,
        output_dir=output_dir,
        model_dir=model_dir,
        figure_dir=figure_dir,
        lookback=args.lookback,
        train_ratio=args.train_ratio,
        random_state=args.random_state,
        xgb_n_estimators=args.xgb_n_estimators,
        xgb_max_depth=args.xgb_max_depth,
        xgb_learning_rate=args.xgb_learning_rate,
        xgb_subsample=args.xgb_subsample,
        xgb_colsample_bytree=args.xgb_colsample_bytree,
        xgb_min_child_weight=args.xgb_min_child_weight,
        xgb_reg_lambda=args.xgb_reg_lambda,
        show_progress=show_progress,
        xgb_verbose_every=args.xgb_verbose_every,
    )
    if show_progress:
        print(f"    Completed step 2 in {format_elapsed(time.time() - step_start)}")

    if show_progress:
        print_progress(3, total_steps, "Detect quiet periods")
    step_start = time.time()
    quiet_metrics = run_quiet_period_detection(
        df,
        output_dir=output_dir,
        model_dir=model_dir,
        figure_dir=figure_dir,
        train_ratio=args.train_ratio,
        major_mag_threshold=args.major_mag_threshold,
        major_horizon_days=args.major_horizon_days,
        quiet_window_days=args.quiet_window_days,
        baseline_window_days=args.baseline_window_days,
        quiet_ratio_threshold=args.quiet_ratio_threshold,
        logreg_l2=args.logreg_l2,
        logreg_lr=args.logreg_lr,
        logreg_iter=args.logreg_iter,
    )
    if show_progress:
        print(f"    Completed step 3 in {format_elapsed(time.time() - step_start)}")

    if show_progress:
        print_progress(4, total_steps, "Run early warning model")
    step_start = time.time()
    warning_metrics = run_early_warning_model(
        df,
        output_dir=output_dir,
        model_dir=model_dir,
        figure_dir=figure_dir,
        train_ratio=args.train_ratio,
        warning_mag_threshold=args.warning_mag_threshold,
        warning_horizon_days=args.warning_horizon_days,
        logreg_l2=args.logreg_l2,
        logreg_lr=args.logreg_lr,
        logreg_iter=args.logreg_iter,
    )
    if show_progress:
        print(f"    Completed step 4 in {format_elapsed(time.time() - step_start)}")

    report_path = build_report(
        output_dir=output_dir,
        mag_metrics=mag_metrics,
        time_metrics=time_metrics,
        quiet_metrics=quiet_metrics,
        warning_metrics=warning_metrics,
    )

    print("=" * 72)
    print("Prediction pipeline completed")
    print("=" * 72)
    print(f"Input file: {args.input_csv}")
    print(f"Rows analyzed: {len(df):,}")
    print(f"Output directory: {output_dir}")
    print(f"Report: {report_path.name}")
    print(f"Total runtime: {format_elapsed(time.time() - pipeline_start)}")
    print("=" * 72)


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
