import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ID_COL = "refcode"
Y_COL = "co2_uptake"

LABELS_PATH = Path("data/processed/MOFCSD_with_co2_labels.csv")
SPLITS_PATH = Path("data/processed/splits_seed42.json")

BH_BASE = Path("data/external/repos/BlackHole/sparsified_graphs")
THRESHOLD = "threshold_0.10"
METHOD = "method_blackhole"
RUNS = ["run_0", "run_1", "run_2", "run_3"]

OUT_DIR = Path("outputs/baselines_bh_features")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }

def normalize_ids(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()

def load_run_features(run: str) -> pd.DataFrame:
    # remaining_node_features_t0.10_rX.csv
    r = run.split("_")[1]  # "0"
    p = BH_BASE / THRESHOLD / METHOD / run / f"remaining_node_features_t0.10_r{r}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing node features: {p}")
    X = pd.read_csv(p)
    # Find likely ID column
    if ID_COL not in X.columns:
        # try common alternatives
        candidates = [c for c in X.columns if c.lower() in {"refcode", "mof_id", "mof", "id", "name", "node"}]
        if candidates:
            X = X.rename(columns={candidates[0]: ID_COL})
        else:
            # fall back: first column as ID
            X = X.rename(columns={X.columns[0]: ID_COL})
    X[ID_COL] = normalize_ids(X[ID_COL])
    return X

def main():
    lab = pd.read_csv(LABELS_PATH)[[ID_COL, Y_COL]].copy()
    lab[ID_COL] = normalize_ids(lab[ID_COL])
    lab = lab[lab[Y_COL].notna()].copy()

    splits = json.loads(SPLITS_PATH.read_text())
    train_ids = set(map(str.upper, splits["train"]))
    val_ids   = set(map(str.upper, splits["val"]))
    test_ids  = set(map(str.upper, splits["test"]))

    run_results = []
    per_run_summary = []

    for run in RUNS:
        X = load_run_features(run)
        df = X.merge(lab, on=ID_COL, how="inner")
        df = df[df[Y_COL].notna()].copy()

        # numeric feature columns
        feat_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if Y_COL in feat_cols:
            feat_cols.remove(Y_COL)

        tr = df[df[ID_COL].isin(train_ids)]
        va = df[df[ID_COL].isin(val_ids)]
        te = df[df[ID_COL].isin(test_ids)]

        model = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("rf", RandomForestRegressor(
                n_estimators=600,
                random_state=42,
                n_jobs=-1
            )),
        ])

        model.fit(tr[feat_cols], tr[Y_COL])

        p_va = model.predict(va[feat_cols])
        p_te = model.predict(te[feat_cols])

        va_m = metrics(va[Y_COL], p_va)
        te_m = metrics(te[Y_COL], p_te)

        per_run_summary.append({
            "threshold": THRESHOLD,
            "run": run,
            "matched_labeled": len(df),
            "train_n": len(tr),
            "val_n": len(va),
            "test_n": len(te),
            "n_features": len(feat_cols),
            **{f"val_{k}": v for k, v in va_m.items()},
            **{f"test_{k}": v for k, v in te_m.items()},
        })

        print(f"\n{run} matched={len(df)} features={len(feat_cols)}")
        print("VAL :", va_m)
        print("TEST:", te_m)

        run_results.append((va_m, te_m))

    summ = pd.DataFrame(per_run_summary)
    summ.to_csv(OUT_DIR / "per_run_metrics.csv", index=False)

    # mean ± std across runs (test only, val only)
    for split in ["val", "test"]:
        for key in ["rmse", "mae", "r2"]:
            vals = summ[f"{split}_{key}"].values
            print(f"\n{split.upper()} {key}: mean={vals.mean():.4f} std={vals.std(ddof=1):.4f}")

    print("\nSaved:", OUT_DIR / "per_run_metrics.csv")

if __name__ == "__main__":
    main()