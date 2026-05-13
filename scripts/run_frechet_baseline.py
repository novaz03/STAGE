#!/usr/bin/env python3
"""Run the Frechet-distance + DBSCAN trajectory clustering baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd

from evacmob.simulation_clustering import search_best_frechet_dbscan
from evacmob.simulation_results import format_simulation_results


def _parse_percentiles(raw: str) -> list[int]:
    values = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    if not values:
        raise ValueError("At least one eps percentile must be provided.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Frechet+DBSCAN baseline on a point-level simulation parquet."
    )
    parser.add_argument(
        "--input-parquet",
        type=Path,
        default=Path("ref_simulation_point_gdf.parquet"),
        help="Point-level parquet containing one row per trajectory point.",
    )
    parser.add_argument("--label-column", default="reference_lab")
    parser.add_argument("--id-column", default="traj_id")
    parser.add_argument("--order-column", default=None)
    parser.add_argument("--n-resample", type=int, default=50)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--eps-percentiles", default="50,60,70,75,80,85,90,95")
    parser.add_argument(
        "--score",
        choices=["accuracy", "balanced_accuracy"],
        default="accuracy",
        help="Metric used to choose the best DBSCAN run across eps percentiles.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    parser.add_argument("--topn", type=int, default=10, help="Number of leaderboard rows to print.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eps_percentiles = _parse_percentiles(args.eps_percentiles)
    points = gpd.read_parquet(args.input_parquet)

    result = search_best_frechet_dbscan(
        points,
        label_col=args.label_column,
        id_col=args.id_column,
        order_col=args.order_column,
        n_resample=args.n_resample,
        min_samples=args.min_samples,
        eps_percentiles=eps_percentiles,
        score_name=args.score,
        show_progress=not args.no_progress,
    )

    print(f"Input parquet: {args.input_parquet}")
    print(f"Best {result.score_name}: {result.score_value:.4f}")
    print(
        f"Best run: eps_percentile={result.eps_percentile}, "
        f"eps={result.eps:.3f}, min_samples={result.min_samples}"
    )
    print(
        f"ARI={result.ari:.4f}, NMI={result.nmi:.4f}, "
        f"mapped_accuracy_all={result.mapped_accuracy_all:.4f}, "
        f"mapped_accuracy_non_noise={result.mapped_accuracy_non_noise:.4f}, "
        f"n_clusters={result.n_clusters}, noise_fraction={result.noise_fraction:.4f}"
    )
    print()
    print(format_simulation_results(result.view))
    print("=== Top Frechet+DBSCAN Runs ===")
    print(result.leaderboard.head(args.topn).to_string(index=False))


if __name__ == "__main__":
    main()
