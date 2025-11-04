#!/usr/bin/env python
"""Generate Hex_bound_POI.csv and Hex_tesse_raw.parquet."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from evacmob.pipeline import (
    FilteredPOIConfig,
    HexTessellationConfig,
    generate_hex_bound_poi,
    generate_hex_tessellation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate filtered POI CSV and hex tessellation.")
    parser.add_argument("--raw-poi", type=Path, required=True, help="Raw US_POI.csv file path")
    parser.add_argument("--filtered-poi", type=Path, default=Path("Hex_bound_POI.csv"))
    parser.add_argument("--hex-parquet", type=Path, default=Path("Hex_tesse_raw.parquet"))
    parser.add_argument("--min-lon", type=float, default=-88.57)
    parser.add_argument("--max-lon", type=float, default=-79.95)
    parser.add_argument("--min-lat", type=float, default=24.45)
    parser.add_argument("--max-lat", type=float, default=32.35)
    parser.add_argument("--hex-radius-m", type=float, default=8000.0)
    parser.add_argument("--concave-ratio", type=float, default=0.05)
    parser.add_argument(
        "--geometry-column",
        default=None,
        help="Name of geometry (WKT) column in the raw POI CSV; when provided, the bounding box is ignored",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    poi_cfg = FilteredPOIConfig(
        input_csv=args.raw_poi,
        output_csv=args.filtered_poi,
        bounding_box=(args.min_lon, args.max_lon, args.min_lat, args.max_lat),
        geometry_column=args.geometry_column,
    )
    generate_hex_bound_poi(poi_cfg)

    hex_cfg = HexTessellationConfig(
        poi_csv=args.filtered_poi,
        output_parquet=args.hex_parquet,
        concave_ratio=args.concave_ratio,
        hex_radius_m=args.hex_radius_m,
    )
    generate_hex_tessellation(hex_cfg)


if __name__ == "__main__":
    main()
