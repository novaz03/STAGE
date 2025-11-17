#!/usr/bin/env python
"""Example wrapper to copy the uploaded folium HTML into docs/reports."""

from pathlib import Path


def main() -> None:
    src = Path("docs/reports/trajectory_report.html")
    if not src.exists():
        print(
            "No HTML found at docs/reports/trajectory_report.html. "
            "Use evacmob copy-html --src <path> first."
        )
        return
    print(f"HTML report present at: {src.resolve()}")


if __name__ == "__main__":
    main()
