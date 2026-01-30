import pandas as pd
import numpy as np
from pathlib import Path

TOPK_PATH = Path("outputs/candidates/topk_candidates.csv")
GRAPH_PATH = Path("graphs/threshold_0.10/run_0/edges.csv")

def smooth_scores(df, edges):
    ref_to_score = df.set_index("refcode")["screen_score"]
    edges = edges[edges["source"].isin(ref_to_score.index) & edges["target"].isin(ref_to_score.index)]

    scores = {}
    for ref in ref_to_score.index:
        nbrs = edges[edges["source"] == ref]["target"]
        if nbrs.empty:
            scores[ref] = ref_to_score[ref]
            continue
        nbr_vals = ref_to_score.loc[nbrs]
        scores[ref] = 0.7 * ref_to_score[ref] + 0.3 * nbr_vals.mean()
    return pd.Series(scores)

def main():
    df = pd.read_csv(TOPK_PATH)
    edges = pd.read_csv(GRAPH_PATH)
    smoothed = smooth_scores(df, edges)
    df["screen_score_smooth"] = df["refcode"].map(smoothed)
    df = df.sort_values("screen_score_smooth", ascending=False)
    df.to_csv(TOPK_PATH, index=False)
    print("[smooth_neighbors] Updated Top-K with smoothed screen_score")

if __name__ == "__main__":
    main()
