from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
GRAPHS_DIR = PROJECT_ROOT / "graphs"
OUT_DIR = PROJECT_ROOT / "outputs" / "candidates"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["Largest Cavity Diameter", "Pore Limiting Diameter", "Largest Free Sphere"]


def detect_label_column(df: pd.DataFrame) -> str:
    """
    Robust label column detection:
    - prefer columns containing 'co2' and ('298' or '1bar' or 'uptake')
    - otherwise fallback to any numeric column containing 'co2'
    """
    cols = list(df.columns)
    lowered = {c: c.lower() for c in cols}

    preferred = []
    for c in cols:
        s = lowered[c]
        if "co2" in s and ("uptake" in s or "1bar" in s or "298" in s):
            preferred.append(c)
    if preferred:
        # pick the most "specific"
        preferred.sort(key=lambda x: (("1bar" in lowered[x]) + ("298" in lowered[x]) + ("uptake" in lowered[x])), reverse=True)
        return preferred[0]

    fallback = [c for c in cols if "co2" in lowered[c] and pd.api.types.is_numeric_dtype(df[c])]
    if fallback:
        return fallback[0]

    raise ValueError(
        "Could not detect CO2 label column. "
        "Open data/processed/MOFCSD_with_co2_labels.csv and check column names."
    )


def load_edges(edge_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(edge_csv)
    # try common schemas
    # expected either: src, dst, weight  OR  u, v, w
    cols = [c.lower() for c in df.columns]
    if {"src", "dst"}.issubset(cols):
        src = df.columns[cols.index("src")]
        dst = df.columns[cols.index("dst")]
        w = df.columns[cols.index("weight")] if "weight" in cols else None
    elif {"u", "v"}.issubset(cols):
        src = df.columns[cols.index("u")]
        dst = df.columns[cols.index("v")]
        w = df.columns[cols.index("w")] if "w" in cols else None
    else:
        # if unknown, assume first two columns are nodes
        src, dst = df.columns[:2]
        w = df.columns[2] if df.shape[1] >= 3 else None

    out = df[[src, dst]].copy()
    out.columns = ["src", "dst"]
    if w is not None:
        out["weight"] = df[w].astype(float)
    else:
        out["weight"] = 1.0
    return out


def graph_neighbor_prediction(
    edges: pd.DataFrame,
    labeled_values: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """
    For each node, compute weighted neighbor mean using only labeled neighbors.

    Returns:
      pred: neighbor-mean prediction for nodes that have labeled neighbors
      support: neighbor support (sum of weights from labeled neighbors)
    """
    # labeled_values: index=refcode, value=label
    labeled_set = set(labeled_values.index)

    # keep edges where dst is labeled (so src can borrow label) and vice versa
    e1 = edges[edges["dst"].isin(labeled_set)].copy()
    e1["nbr_label"] = e1["dst"].map(labeled_values)

    e2 = edges[edges["src"].isin(labeled_set)].copy()
    e2 = e2.rename(columns={"src": "dst", "dst": "src"})  # swap to reuse same logic
    e2["nbr_label"] = e2["dst"].map(labeled_values)

    e = pd.concat([e1, e2], ignore_index=True)
    if e.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    # weighted mean per src
    e["wlabel"] = e["weight"] * e["nbr_label"]
    grp = e.groupby("src", as_index=True)
    support = grp["weight"].sum()
    pred = grp["wlabel"].sum() / support.replace(0, np.nan)

    return pred, support


def main(top_k: int = 50) -> None:
    mof_path = PROCESSED / "MOFCSD_with_co2_labels.csv"
    if not mof_path.exists():
        raise FileNotFoundError(f"Missing: {mof_path}")

    df = pd.read_csv(mof_path)

    if "refcode" not in df.columns:
        raise ValueError("Expected 'refcode' column in MOFCSD_with_co2_labels.csv")

    label_col = detect_label_column(df)
    print(f"[rank_candidates] Using label column: {label_col}")

    # basic feature cleaning
    for c in FEATURES:
        if c not in df.columns:
            raise ValueError(f"Missing feature column: {c}")
    df_feat = df[["refcode", label_col] + FEATURES].copy()

    # Split labeled vs unlabeled
    labeled = df_feat.dropna(subset=[label_col]).copy()
    unlabeled = df_feat[df_feat[label_col].isna()].copy()

    print(f"[rank_candidates] labeled={len(labeled)} unlabeled={len(unlabeled)}")

    # Train a stable descriptor baseline (RF) for initial ranking
    X = labeled[FEATURES].astype(float).values
    y = labeled[label_col].astype(float).values

    # simple robustness split (not your thesis eval split; this is ranking utility)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    rf.fit(Xtr, ytr)

    # Predict on all MOFs (including unlabeled)
    X_all = df_feat[FEATURES].astype(float).values
    pred_rf = rf.predict(X_all)

    # RF uncertainty proxy = std across trees
    tree_preds = np.stack([t.predict(X_all) for t in rf.estimators_], axis=0)  # [trees, n]
    pred_std = tree_preds.std(axis=0)

    df_all = pd.DataFrame(
        {
            "refcode": df_feat["refcode"].values,
            "pred_uptake_rf": pred_rf,
            "pred_uptake_rf_std": pred_std,
        }
    )

    # Graph neighbor agreement across thresholds/runs
    # We use a few representative graphs you already have.
    graph_candidates = []
    for thr in ["threshold_0.90", "threshold_0.10", "threshold_0.00"]:
        thr_dir = GRAPHS_DIR / thr
        if not thr_dir.exists():
            continue
        for run_dir in sorted(thr_dir.glob("run_*/edges.csv")):
            graph_candidates.append(run_dir)

    labeled_values = labeled.set_index("refcode")[label_col].astype(float)

    per_graph_preds = []
    per_graph_support = []
    used_graphs = []

    for edge_csv in graph_candidates:
        edges = load_edges(edge_csv)
        # node ids in edges might not be refcodes → if mismatch, this graph contributes nothing
        pred, support = graph_neighbor_prediction(edges, labeled_values)
        if len(pred) == 0:
            continue

        used_graphs.append(str(edge_csv.relative_to(PROJECT_ROOT)))
        per_graph_preds.append(pred.rename(str(edge_csv.parent)))
        per_graph_support.append(support.rename(str(edge_csv.parent)))

    if per_graph_preds:
        preds_mat = pd.concat(per_graph_preds, axis=1)  # index=node, cols=graphs
        supp_mat = pd.concat(per_graph_support, axis=1)

        # aggregate across graphs
        nbr_mean = preds_mat.mean(axis=1)
        nbr_std = preds_mat.std(axis=1)
        nbr_support = supp_mat.sum(axis=1)

        df_all = df_all.merge(nbr_mean.rename("graph_neighbor_mean"), left_on="refcode", right_index=True, how="left")
        df_all = df_all.merge(nbr_std.rename("graph_neighbor_std"), left_on="refcode", right_index=True, how="left")
        df_all = df_all.merge(nbr_support.rename("graph_neighbor_support"), left_on="refcode", right_index=True, how="left")
    else:
        df_all["graph_neighbor_mean"] = np.nan
        df_all["graph_neighbor_std"] = np.nan
        df_all["graph_neighbor_support"] = np.nan

    # Compose a single screening score
    # - primary: RF prediction
    # - bonus: neighbor mean (if available)
    # - confidence: high support + low std
    df_all["screen_score"] = df_all["pred_uptake_rf"]
    df_all["screen_score"] = df_all["screen_score"].where(df_all["graph_neighbor_mean"].isna(), 0.6 * df_all["pred_uptake_rf"] + 0.4 * df_all["graph_neighbor_mean"])

    # confidence score
    # higher support increases confidence, higher neighbor std decreases confidence
    supp = df_all["graph_neighbor_support"].fillna(0.0)
    gstd = df_all["graph_neighbor_std"].fillna(df_all["pred_uptake_rf_std"])
    df_all["confidence"] = np.log1p(supp) / (1.0 + gstd)

    # Keep only unlabeled candidates for shortlist
    unlabeled_set = set(unlabeled["refcode"].values)
    cand = df_all[df_all["refcode"].isin(unlabeled_set)].copy()

    cand = cand.sort_values(["screen_score", "confidence"], ascending=[False, False]).head(top_k)

    out_csv = OUT_DIR / "topk_candidates.csv"
    cand.to_csv(out_csv, index=False)
    print(f"[rank_candidates] Saved: {out_csv}")

    out_meta = OUT_DIR / "topk_candidates_meta.json"
    meta = {
        "label_col_used": label_col,
        "features": FEATURES,
        "graphs_used": used_graphs,
        "top_k": top_k,
    }
    out_meta.write_text(json.dumps(meta, indent=2))
    print(f"[rank_candidates] Saved: {out_meta}")


if __name__ == "__main__":
    main(top_k=50)
