# src/viz/plot_threshold_comparison.py
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_COV = Path("data/processed/labeled_coverage_by_threshold.csv")
DATA_CONN = Path("outputs/split_connectivity_audit.csv")  # created by audit_split_connectivity.py

OUT_DIR = Path("outputs/figures/03_graph_threshold_diagnostics")


def thr_to_float(thr: str) -> float:
    """
    Convert strings like 'threshold_0.10' -> 0.10
    """
    m = re.search(r"threshold_(\d+\.\d+|\d+)", str(thr))
    return float(m.group(1)) if m else np.nan


def summarize_by_threshold(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Group by threshold, compute mean/std across runs, and sort by numeric threshold.
    """
    df = df.copy()
    df["thr"] = df["threshold"].apply(thr_to_float)
    g = df.groupby("thr")[cols].agg(["mean", "std"]).reset_index()
    # flatten MultiIndex columns like ('graph_nodes','mean') -> 'graph_nodes_mean'
    g.columns = ["thr"] + [f"{c}_{stat}" for c, stat in g.columns[1:]]
    g = g.sort_values("thr", ascending=True).reset_index(drop=True)
    return g


def savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_COV.exists():
        raise FileNotFoundError(f"Missing {DATA_COV}. Run labeled_coverage_by_threshold.py first.")

    cov = pd.read_csv(DATA_COV)

    # ---- FIGURE 1: Label coverage vs threshold ----
    # We want the figure to visually show "coverage increases as threshold decreases".
    # To do that clearly, we plot threshold on x but invert x-axis (0.90 -> left, 0.00 -> right).
    cov_sum = summarize_by_threshold(cov, cols=["labeled_in_graph"])

    plt.figure()
    plt.errorbar(
        cov_sum["thr"],
        cov_sum["labeled_in_graph_mean"],
        yerr=cov_sum["labeled_in_graph_std"],
        fmt="-o",
        capsize=3,
    )
    plt.gca().invert_xaxis()
    plt.xlabel("BlackHole sparsification threshold (higher = stricter)")
    plt.ylabel("Labeled MOFs present in graph (mean ± std across runs)")
    plt.title("Label coverage increases as threshold decreases")
    savefig(OUT_DIR / "fig1_label_coverage_vs_threshold.png")

    # ---- FIGURE 2: Graph size vs threshold (nodes + edges) ----
    cov_sum2 = summarize_by_threshold(cov, cols=["graph_nodes", "graph_edges"])

    fig, ax1 = plt.subplots()
    ax1.errorbar(
        cov_sum2["thr"],
        cov_sum2["graph_nodes_mean"],
        yerr=cov_sum2["graph_nodes_std"],
        fmt="-o",
        capsize=3,
        label="Nodes (mean ± std)",
    )
    ax1.set_xlabel("BlackHole sparsification threshold (higher = stricter)")
    ax1.set_ylabel("Graph nodes")
    ax1.invert_xaxis()

    ax2 = ax1.twinx()
    ax2.errorbar(
        cov_sum2["thr"],
        cov_sum2["graph_edges_mean"],
        yerr=cov_sum2["graph_edges_std"],
        fmt="--s",
        capsize=3,
        label="Edges (mean ± std)",
    )
    ax2.set_ylabel("Graph edges")

    # Combine legends from both axes
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")

    plt.title("Graph size grows rapidly as threshold decreases")
    savefig(OUT_DIR / "fig2_graph_size_vs_threshold.png")

    # ---- FIGURE 3: Train-neighbor exposure in TEST vs threshold (connectivity/leakage risk proxy) ----
    if DATA_CONN.exists():
        conn = pd.read_csv(DATA_CONN)

        # conn already has: ['threshold','run',...,'pct_test_with_train_neighbor', 'mean_max_weight_test_to_train']
        conn_sum = summarize_by_threshold(
            conn,
            cols=["pct_test_with_train_neighbor", "mean_max_weight_test_to_train"],
        )

        plt.figure()
        plt.errorbar(
            conn_sum["thr"],
            conn_sum["pct_test_with_train_neighbor_mean"],
            yerr=conn_sum["pct_test_with_train_neighbor_std"],
            fmt="-o",
            capsize=3,
        )
        plt.gca().invert_xaxis()
        plt.xlabel("BlackHole sparsification threshold (higher = stricter)")
        plt.ylabel("% TEST nodes with at least one TRAIN neighbor (mean ± std)")
        plt.title("TEST→TRAIN neighbor exposure rises when the graph becomes denser")
        savefig(OUT_DIR / "fig3_test_train_exposure_vs_threshold.png")

        plt.figure()
        plt.errorbar(
            conn_sum["thr"],
            conn_sum["mean_max_weight_test_to_train_mean"],
            yerr=conn_sum["mean_max_weight_test_to_train_std"],
            fmt="-o",
            capsize=3,
        )
        plt.gca().invert_xaxis()
        plt.xlabel("BlackHole sparsification threshold (higher = stricter)")
        plt.ylabel("Mean max edge-weight from TEST→TRAIN (mean ± std)")
        plt.title("Strength of nearest TEST→TRAIN connections increases at lower thresholds")
        savefig(OUT_DIR / "fig4_test_train_weight_vs_threshold.png")

    else:
        print(f"[WARN] {DATA_CONN} not found. Skipping connectivity/leakage plots.")

    print(f"Saved figures to: {OUT_DIR.resolve()}")
    print("Generated:")
    print(" - fig1_label_coverage_vs_threshold.png")
    print(" - fig2_graph_size_vs_threshold.png")
    if DATA_CONN.exists():
        print(" - fig3_test_train_exposure_vs_threshold.png")
        print(" - fig4_test_train_weight_vs_threshold.png")


if __name__ == "__main__":
    main()
