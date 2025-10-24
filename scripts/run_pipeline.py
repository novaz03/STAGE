#!/usr/bin/env python
"""Example pipeline orchestration script."""
from pathlib import Path
from evacmob.report import write_text_report

def main():
    out = write_text_report("Pipeline ran (placeholder).", Path("outputs/pipeline.txt"))
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
