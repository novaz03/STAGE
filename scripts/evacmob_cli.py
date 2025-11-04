#!/usr/bin/env python
"""Command-line interface for evacmob."""
import argparse
from pathlib import Path
from evacmob.visualize import copy_static_report
from evacmob.simulate import (
    SimulationConfig,
    build_demo_simulation_inputs,
    run_simulation,
)

def main():
    parser = argparse.ArgumentParser(prog="evacmob", description="Hurricane mobility toolkit")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sim = sub.add_parser("simulate", help="Run a placeholder simulation")
    sim.add_argument("--out", type=Path, default=Path("outputs/sim.txt"))

    rep = sub.add_parser("copy-html", help="Copy an existing HTML report into docs/reports")
    rep.add_argument("--src", type=Path, required=True)
    rep.add_argument("--dest", type=Path, default=Path("docs/reports/simulated_trajectories.html"))

    args = parser.parse_args()

    if args.cmd == "simulate":
        config = SimulationConfig()
        people_df, pois = build_demo_simulation_inputs(config=config)
        trajectories = run_simulation(people_df, pois, config=config)

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        lines = ["person_id,day,latitude,longitude"]
        for person_id in sorted(trajectories):
            for day, poi_idx in trajectories[person_id]:
                geom = pois.at[poi_idx, "geometry"]
                lines.append(f"{person_id},{day},{geom.y:.6f},{geom.x:.6f}")

        out_path.write_text("\n".join(lines))
        print(f"Wrote {out_path}")

    elif args.cmd == "copy-html":
        out = copy_static_report(args.src, args.dest)
        print(f"Copied to {out}")

if __name__ == "__main__":
    main()
