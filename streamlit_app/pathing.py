from __future__ import annotations

from pathlib import Path

# streamlit_app/pathing.py
# app.py is inside streamlit_app/, so project root is one level above it.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
DATA_DIR = PROJECT_ROOT / "data"


def resolve_first_existing(rel_paths: list[str]) -> Path | None:
    """
    Try multiple relative paths (relative to PROJECT_ROOT) and return the first that exists.
    """
    for rel in rel_paths:
        p = PROJECT_ROOT / rel
        if p.exists():
            return p
    return None
