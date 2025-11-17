#!/usr/bin/env python
"""Command-line interface for evacmob."""

import argparse
from pathlib import Path
from evacmob.visualize import copy_static_report


def main():
    parser = argparse.ArgumentParser(prog="evacmob", description="Hurricane mobility toolkit")
    sub = parser.add_subparsers(dest="cmd", required=True)

    rep = sub.add_parser("copy-html", help="Copy an existing HTML report into docs/reports")
    rep.add_argument("--src", type=Path, required=True)
    rep.add_argument("--dest", type=Path, default=Path("docs/reports/trajectory_report.html"))

    args = parser.parse_args()

    if args.cmd == "copy-html":
        out = copy_static_report(args.src, args.dest)
        print(f"Copied to {out}")


if __name__ == "__main__":
    main()
