#!/usr/bin/env python
"""Example wrapper to confirm the copied folium HTML is present."""

from pathlib import Path


def main() -> None:
    src = Path("outputs/trajectory_report.html")
    if not src.exists():
        print(
            "No HTML found at outputs/trajectory_report.html. "
            "Use evacmob copy-html --src <path> first."
        )
        return
    print(f"HTML report present at: {src.resolve()}")


if __name__ == "__main__":
    main()
