import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pattern Discovering phase wrapper for the raw earthquake clustering pipeline."
    )
    parser.add_argument("--input-csv", type=Path, default=Path("data/dongdat.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("hoigreen/pattern_discovering/outputs"))
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


def main() -> None:
    args = parse_args()
    target_script = (
        Path(__file__).resolve().parents[1]
        / "clustering_pattern_mining"
        / "run_raw_visualization_clustering.py"
    )
    cmd = [
        sys.executable,
        str(target_script),
        "--input-csv",
        str(args.input_csv),
        "--output-dir",
        str(args.output_dir),
        "--event-type",
        args.event_type,
        "--region-grid-size",
        str(args.region_grid_size),
        "--event-sample-size",
        str(args.event_sample_size),
        "--eval-sample-size",
        str(args.eval_sample_size),
        "--plot-sample-size",
        str(args.plot_sample_size),
        "--event-k-min",
        str(args.event_k_min),
        "--event-k-max",
        str(args.event_k_max),
        "--region-k-min",
        str(args.region_k_min),
        "--region-k-max",
        str(args.region_k_max),
        "--min-events-per-region",
        str(args.min_events_per_region),
        "--top-regions",
        str(args.top_regions),
        "--random-state",
        str(args.random_state),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
