from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
GRAPHS_DIR = PROJECT_ROOT / "graphs"
GNN_DIR = PROJECT_ROOT / "outputs" / "gnn_gcn" / "threshold_0.10"
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "pyg" / "threshold_0.10"
OUT_DIR = PROJECT_ROOT / "outputs" / "candidates"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["Largest Cavity Diameter", "Pore Limiting Diameter", "Largest Free Sphere"]

def detect_label_column(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lowered = {c: c.lower() for c in cols}
    preferred = [c for c in cols if "co2" in lowered[c] and any(w in lowered[c] for w in ["uptake", "298", "1bar"])]
    if preferred:
        return sorted(preferred, key=lambda x: sum(w in lowered[x] for w in ["1bar", "298", "uptake"]), reverse=True)[0]
    fallback = [c for c in cols if "co2" in lowered[c] and pd.api.types.is_numeric_dtype(df[c])]
    if fallback:
        return fallback[0]
    raise ValueError("Could not detect CO₂ label column.")

def main(top_k: int = 50):
    df = pd.read_csv(PROCESSED / "MOFCSD_with_co2_labels.csv")
    label_col = detect_label_column(df)
    print(f"[rank_candidates] Using label column: {label_col}")

    df_feat = df[["refcode", label_col] + FEATURES].copy()
    labeled = df_feat.dropna(subset=[label_col])
    unlabeled = df_feat[df_feat[label_col].isna()]
    print(f"[rank_candidates] labeled={len(labeled)} unlabeled={len(unlabeled)}")

    # --- Train RF ---
    X = labeled[FEATURES].astype(float).values
    y = labeled[label_col].astype(float).values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=42)
    rf.fit(Xtr, ytr)

    X_all = df_feat[FEATURES].astype(float).values
    pred_rf = rf.predict(X_all)
    pred_rf_std = np.stack([t.predict(X_all) for t in rf.estimators_], axis=0).std(axis=0)

    df_all = pd.DataFrame({
        "refcode": df_feat["refcode"].values,
        "pred_uptake_rf": pred_rf,
        "pred_uptake_rf_std": pred_rf_std,
    })

    # --- GNN predictions ---
    agg_path = GNN_DIR / "aggregated_predictions.npz"
    if agg_path.exists():
        agg = np.load(agg_path)
        gnn_mean = agg["mean"]
        gnn_std = agg["std"]

        # match with refcodes from run_0.pt
        run0_path = DATA_DIR / "run_0.pt"
        data = torch.load(run0_path, map_location="cpu")
        if hasattr(data, "refcode"):
            df_gnn = pd.DataFrame({
                "refcode": data.refcode,
                "pred_uptake_gnn": gnn_mean,
                "pred_uptake_gnn_std": gnn_std,
            })
            df_all = df_all.merge(df_gnn, on="refcode", how="left")
        else:
            print("Warning: run_0.pt missing refcodes. Skipping GNN merge.")
            df_all["pred_uptake_gnn"] = np.nan
            df_all["pred_uptake_gnn_std"] = np.nan
    else:
        print("⚠️ GNN aggregated_predictions.npz not found")
        df_all["pred_uptake_gnn"] = np.nan
        df_all["pred_uptake_gnn_std"] = np.nan

    # --- Compose screen score ---
    df_all["screen_score"] = 0.5 * df_all["pred_uptake_rf"] + 0.5 * df_all["pred_uptake_gnn"]
    df_all["screen_score"] = df_all["screen_score"].fillna(df_all["pred_uptake_rf"])

    # --- Confidence ---
    std_combined = df_all["pred_uptake_gnn_std"].fillna(df_all["pred_uptake_rf_std"] + 1e-4)
    df_all["confidence"] = (1.0 / (1.0 + std_combined)).clip(lower=0.01)

    # --- Rationale ---
    df_all["rationale"] = np.where(
        df_all["pred_uptake_gnn"].notna(),
        "RF+GNN hybrid",
        "RF-only (no GNN prediction)"
    )

    # --- Add feature columns back for easy inspection ---
    desc_df = df[["refcode"] + FEATURES].drop_duplicates()
    df_all = df_all.merge(desc_df, on="refcode", how="left")

    # --- Save full base prediction table ---
    df_all.to_csv(OUT_DIR / "top_candidates.csv", index=False)

    # --- Filter and rank ---
    cand = df_all[df_all["refcode"].isin(unlabeled["refcode"])]
    cand = cand.sort_values(["screen_score", "confidence"], ascending=[False, False])
    cand["rank"] = np.arange(1, len(cand) + 1)

    topk = cand.head(top_k)
    topk.to_csv(OUT_DIR / "topk_candidates.csv", index=False)
    print("[rank_candidates] Saved: topk_candidates.csv")

    meta = {
        "label_col_used": label_col,
        "features": FEATURES,
        "uses_gnn": agg_path.exists(),
        "top_k": top_k,
    }
    Path(OUT_DIR / "topk_candidates_meta.json").write_text(json.dumps(meta, indent=2))
    print("[rank_candidates] Saved: topk_candidates_meta.json")

if __name__ == "__main__":
    main()