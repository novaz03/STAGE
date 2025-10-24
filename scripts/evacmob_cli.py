#!/usr/bin/env python
"""Command-line interface for evacmob."""
import argparse
from pathlib import Path
from evacmob.visualize import copy_static_report
from evacmob.simulate import run_simulation
from evacmob.report import write_text_report

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
        pts = run_simulation({})
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("\n".join(f"{lat},{lon}" for lat, lon in pts))
        print(f"Wrote {args.out}")

    elif args.cmd == "copy-html":
        out = copy_static_report(args.src, args.dest)
        print(f"Copied to {out}")

if __name__ == "__main__":
    main()
