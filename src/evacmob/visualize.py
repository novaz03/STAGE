"""Visualization helpers (folium, matplotlib)."""
from pathlib import Path

def copy_static_report(src_html: str | Path, dest: str | Path) -> Path:
    src = Path(src_html)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(src.read_bytes())
    return dest
