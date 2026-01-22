import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# paths
NODE_FEATS = Path(
    "data/external/repos/BlackHole/sparsified_graphs/"
    "threshold_0.10/method_blackhole/run_0/"
    "remaining_node_features_t0.10_r0.csv"
)
LABELS = Path("data/processed/MOFCSD_with_co2_labels.csv")
SPLITS = Path("data/processed/splits_seed42.json")

ID_COL = "refcode"
Y_COL = "co2_uptake"

def metrics(y, p):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "mae": float(mean_absolute_error(y, p)),
        "r2": float(r2_score(y, p)),
    }

def main():
    X = pd.read_csv(NODE_FEATS)
    y = pd.read_csv(LABELS)[[ID_COL, Y_COL]]

    # normalize IDs
    X[ID_COL] = X[ID_COL].astype(str).str.upper()
    y[ID_COL] = y[ID_COL].astype(str).str.upper()

    df = X.merge(y, on=ID_COL, how="inner")
    df = df[df[Y_COL].notna()].copy()

    print("Matched labeled nodes:", len(df))

    splits = json.loads(SPLITS.read_text())
    tr, va, te = map(set, (splits["train"], splits["val"], splits["test"]))

    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols.remove(Y_COL)

    def sel(ids):
        return df[df[ID_COL].isin(ids)]

    tr_df, va_df, te_df = sel(tr), sel(va), sel(te)

    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("rf", RandomForestRegressor(
            n_estimators=500,
            random_state=42,
            n_jobs=-1
        )),
    ])

    model.fit(tr_df[feature_cols], tr_df[Y_COL])

    print("VAL:", metrics(va_df[Y_COL], model.predict(va_df[feature_cols])))
    print("TEST:", metrics(te_df[Y_COL], model.predict(te_df[feature_cols])))

if __name__ == "__main__":
    main()
