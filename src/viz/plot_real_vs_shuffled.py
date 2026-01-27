from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------- INPUTS (edit only if your paths differ) ----------
BH_PER_RUN = Path("outputs/baselines_bh_features/per_run_metrics.csv")
DESC_BASELINES = Path("outputs/baselines/metrics.json")  # ridge/knn/rf
OUT_DIR = Path("outputs/figures/05_gnn_vs_shuffled")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# We will auto-detect these by searching for summary.json files
SEARCH_ROOT = Path("outputs")


def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text())


def _find_gnn_summaries() -> Tuple[Optional[Path], Optional[Path]]:
    """
    Try to find:
      - train_gnn_gcn.py summary.json (real graph)
      - train_gnn_gcn_shuffled.py summary.json (shuffled graph)

    We match by file content if possible, else by parent folder naming.
    """
    cands = list(SEARCH_ROOT.rglob("summary.json"))
    if not cands:
        return None, None

    real = None
    shuf = None

    # Heuristic 1: folder name contains "shuff"
    for p in cands:
        low = str(p).lower()
        if "shuff" in low:
            shuf = p
        elif "gnn" in low or "gcn" in low:
            # might be real
            real = p

    # Heuristic 2: if still ambiguous, pick two newest
    if real is None or shuf is None:
        cands_sorted = sorted(cands, key=lambda x: x.stat().st_mtime, reverse=True)
        # try to identify using content keys (both have same structure)
        # so we just assign newest as shuffled if folder contains 'shuff'
        for p in cands_sorted:
            low = str(p).lower()
            if shuf is None and "shuff" in low:
                shuf = p
            if real is None and "shuff" not in low:
                real = p
            if real is not None and shuf is not None:
                break

    return real, shuf


def _parse_gnn_summary(p: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Expected format:
    {
      "threshold": "...",
      "val": {"rmse": {"mean":..., "std":...}, ...},
      "test": {...}
    }
    """
    d = _load_json(p)
    out = {}
    for split in ["val", "test"]:
        out[split] = {}
        for metric in ["rmse", "mae", "r2"]:
            out[split][metric] = {
                "mean": float(d[split][metric]["mean"]),
                "std": float(d[split][metric]["std"]),
            }
    return out


def _summarize_bh_features(per_run_csv: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    per_run_metrics.csv columns you showed:
    ['threshold','run','matched_labeled','train_n','val_n','test_n','n_features',
     'val_rmse','val_mae','val_r2','test_rmse','test_mae','test_r2']
    """
    df = pd.read_csv(per_run_csv)

    out = {}
    for split in ["val", "test"]:
        out[split] = {}
        for metric in ["rmse", "mae", "r2"]:
            col = f"{split}_{metric}"
            out[split][metric] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std(ddof=0)),
            }
    return out


def _pick_descriptor_baseline(metrics_json: Path, split: str, metric: str) -> Optional[float]:
    """
    metrics.json from train_baselines.py likely stores something like:
    {
      "RIDGE": {"val": {...}, "test": {...}},
      "KNN": {...},
      "RF": {...}
    }
    We'll pick the BEST (lowest rmse/mae, highest r2) on VAL and then report TEST,
    OR if you prefer: pick best on TEST. Here: pick best on VAL to be fair.
    """
    if not metrics_json.exists():
        return None

    d = _load_json(metrics_json)

    # normalize keys
    models = []
    for k, v in d.items():
        # accept both "RIDGE" and "ridge"
        if isinstance(v, dict) and ("val" in v and "test" in v):
            models.append((k, v))

    if not models:
        return None

    # choose best on val
    def score(val_dict):
        x = float(val_dict.get(metric, np.nan))
        if metric in ["rmse", "mae"]:
            return x  # lower is better
        else:
            return -x  # higher is better -> minimize negative

    best_name, best_block = min(models, key=lambda kv: score(kv[1]["val"]))
    return float(best_block[split][metric])


def _barplot_compare(title: str, labels, means, stds, ylabel: str, outpath: Path):
    x = np.arange(len(labels))
    plt.figure(figsize=(8.5, 4.8))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, labels, rotation=10)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()
    print("Saved:", outpath)


def main():
    # --- Load BH-feature baseline ---
    assert BH_PER_RUN.exists(), f"Missing {BH_PER_RUN} (BH baseline per-run metrics)."
    bh = _summarize_bh_features(BH_PER_RUN)

    # --- Find and load GNN summaries ---
    real_path, shuf_path = _find_gnn_summaries()
    if real_path is None or shuf_path is None:
        print("[ERROR] Could not auto-find both real + shuffled summary.json.")
        print("Found summary.json files:")
        for p in SEARCH_ROOT.rglob("summary.json"):
            print(" -", p)
        raise SystemExit(1)

    gnn_real = _parse_gnn_summary(real_path)
    gnn_shuf = _parse_gnn_summary(shuf_path)

    # --- Create plots for VAL and TEST ---
    for split in ["val", "test"]:
        # RMSE plot
        labels = ["BH-features (baseline)", "GNN (real graph)", "GNN (shuffled)"]
        means = [
            bh[split]["rmse"]["mean"],
            gnn_real[split]["rmse"]["mean"],
            gnn_shuf[split]["rmse"]["mean"],
        ]
        stds = [
            bh[split]["rmse"]["std"],
            gnn_real[split]["rmse"]["std"],
            gnn_shuf[split]["rmse"]["std"],
        ]
        _barplot_compare(
            title=f"RMSE comparison on {split.upper()} (mean ± std across runs)",
            labels=labels,
            means=means,
            stds=stds,
            ylabel="RMSE (lower is better)",
            outpath=OUT_DIR / f"real_vs_shuffled_rmse_{split}.png",
        )

        # R2 plot
        means_r2 = [
            bh[split]["r2"]["mean"],
            gnn_real[split]["r2"]["mean"],
            gnn_shuf[split]["r2"]["mean"],
        ]
        stds_r2 = [
            bh[split]["r2"]["std"],
            gnn_real[split]["r2"]["std"],
            gnn_shuf[split]["r2"]["std"],
        ]
        _barplot_compare(
            title=f"R² comparison on {split.upper()} (mean ± std across runs)",
            labels=labels,
            means=means_r2,
            stds=stds_r2,
            ylabel="R² (higher is better)",
            outpath=OUT_DIR / f"real_vs_shuffled_r2_{split}.png",
        )

    # --- Save a small CSV summary too (useful for thesis tables) ---
    rows = []
    for model_name, block in [
        ("BH-features", bh),
        ("GNN_real", gnn_real),
        ("GNN_shuffled", gnn_shuf),
    ]:
        for split in ["val", "test"]:
            rows.append({
                "model": model_name,
                "split": split,
                "rmse_mean": block[split]["rmse"]["mean"],
                "rmse_std": block[split]["rmse"]["std"],
                "mae_mean": block[split]["mae"]["mean"],
                "mae_std": block[split]["mae"]["std"],
                "r2_mean": block[split]["r2"]["mean"],
                "r2_std": block[split]["r2"]["std"],
            })
    out_csv = OUT_DIR / "model_comparison_real_vs_shuffled.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("Saved:", out_csv)

    print("\nUsing summaries:")
    print(" - GNN real   :", real_path)
    print(" - GNN shuffled:", shuf_path)


if __name__ == "__main__":
    main()