import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ID_COL = "refcode"
Y_COL = "co2_uptake"

DATA_PATH = Path("data/processed/MOFCSD_with_co2_labels.csv")
SPLITS_PATH = Path("data/processed/splits_seed42.json")

OUT_DIR = Path("outputs/baselines")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def m(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}

def main():
    df = pd.read_csv(DATA_PATH)
    df = df[df[Y_COL].notna()].copy()
    df[ID_COL] = df[ID_COL].astype(str)

    splits = json.loads(SPLITS_PATH.read_text())
    train_ids, val_ids, test_ids = set(splits["train"]), set(splits["val"]), set(splits["test"])

    # numeric features only
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if Y_COL in num_cols:
        num_cols.remove(Y_COL)

    X = df[num_cols]
    y = df[Y_COL].astype(float).values
    ids = df[ID_COL].values

    idx = {rid: i for i, rid in enumerate(ids)}
    i_tr = np.array([idx[r] for r in train_ids if r in idx])
    i_va = np.array([idx[r] for r in val_ids if r in idx])
    i_te = np.array([idx[r] for r in test_ids if r in idx])

    X_tr, y_tr = X.iloc[i_tr], y[i_tr]
    X_va, y_va = X.iloc[i_va], y[i_va]
    X_te, y_te = X.iloc[i_te], y[i_te]

    print("Train/Val/Test:", len(i_tr), len(i_va), len(i_te))
    print("Numeric features used:", len(num_cols))
    print("Feature columns:", num_cols)

    models = {
        "ridge": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=42)),
        ]),
        "knn": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", KNeighborsRegressor(n_neighbors=7)),
        ]),
        "rf": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=400, random_state=42, n_jobs=-1
            )),
        ]),
    }

    results = {}
    preds = []

    for name, model in models.items():
        model.fit(X_tr, y_tr)

        p_va = model.predict(X_va)
        p_te = model.predict(X_te)

        results[name] = {"val": m(y_va, p_va), "test": m(y_te, p_te)}

        print(f"\n{name.upper()} VAL :", results[name]["val"])
        print(f"{name.upper()} TEST:", results[name]["test"])

        for rid, yt, yp in zip(ids[i_te], y_te, p_te):
            preds.append({"model": name, "refcode": rid, "y_true": float(yt), "y_pred": float(yp)})

    (OUT_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame(preds).to_csv(OUT_DIR / "test_predictions.csv", index=False)

    print("\nSaved:", OUT_DIR / "metrics.json")
    print("Saved:", OUT_DIR / "test_predictions.csv")

if __name__ == "__main__":
    main()