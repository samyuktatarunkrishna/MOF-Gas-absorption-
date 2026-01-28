from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IN_CSV = PROJECT_ROOT / "outputs" / "candidates" / "topk_candidates.csv"
OUT_FIG_DIR = PROJECT_ROOT / "outputs" / "figures" / "06_summary"
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_FIG_DIR / "topk_candidates_top20_v1.png"


def main() -> None:
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Missing: {IN_CSV}. Run rank_candidates.py first.")

    df = pd.read_csv(IN_CSV).head(20).copy()

    # uncertainty band: use graph_neighbor_std if present else RF std
    df["uncert"] = df["graph_neighbor_std"].fillna(df["pred_uptake_rf_std"]).fillna(0.0)

    # reverse for nicer horizontal ordering
    df = df.iloc[::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(df["refcode"], df["screen_score"], xerr=df["uncert"])
    plt.xlabel("Screening score (higher = prioritize)")
    plt.ylabel("MOF refcode")
    plt.title("Top-20 MOF candidates to evaluate next (with uncertainty band)")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200)
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
