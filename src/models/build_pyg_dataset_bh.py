import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

ID_COL = "refcode"
Y_COL = "co2_uptake"

SPLITS_PATH = Path("data/processed/splits_seed42.json")
LABELS_PATH = Path("data/processed/MOFCSD_with_co2_labels.csv")

BH_BASE = Path("data/external/repos/BlackHole/sparsified_graphs")
GRAPHS_BASE = Path("graphs")  # exported edges.csv live here

def norm_ids(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()

def load_splits():
    s = json.loads(SPLITS_PATH.read_text())
    return set(map(str.upper, s["train"])), set(map(str.upper, s["val"])), set(map(str.upper, s["test"]))

def load_labels():
    lab = pd.read_csv(LABELS_PATH)[[ID_COL, Y_COL]].copy()
    lab[ID_COL] = norm_ids(lab[ID_COL])
    lab = lab[lab[Y_COL].notna()].copy()
    return lab

def load_node_features(threshold: str, run: str) -> pd.DataFrame:
    r = run.split("_")[1]
    p = BH_BASE / threshold / "method_blackhole" / run / f"remaining_node_features_t{threshold.split('_')[1]}_r{r}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing node features: {p}")
    X = pd.read_csv(p)

    # Ensure ID col exists
    if ID_COL not in X.columns:
        # try common names
        candidates = [c for c in X.columns if c.lower() in {"refcode", "mof_id", "mof", "id", "name", "node"}]
        if candidates:
            X = X.rename(columns={candidates[0]: ID_COL})
        else:
            X = X.rename(columns={X.columns[0]: ID_COL})

    X[ID_COL] = norm_ids(X[ID_COL])
    return X

def load_edges(threshold: str, run: str) -> pd.DataFrame:
    p = GRAPHS_BASE / threshold / run / "edges.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing edges.csv: {p}")
    e = pd.read_csv(p)
    e["source"] = norm_ids(e["source"])
    e["target"] = norm_ids(e["target"])
    if "weight" in e.columns:
        e["weight"] = pd.to_numeric(e["weight"], errors="coerce").fillna(1.0)
    else:
        e["weight"] = 1.0
    return e

def build_data(threshold: str, run: str, out_path: Path):
    train_ids, val_ids, test_ids = load_splits()
    labels = load_labels()
    X = load_node_features(threshold, run)
    E = load_edges(threshold, run)

    # Merge features + labels
    df = X.merge(labels, on=ID_COL, how="left")

    # Choose numeric feature columns (exclude target)
    feat_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if Y_COL in feat_cols:
        feat_cols.remove(Y_COL)

    # Keep only nodes that are in the graph (appear in edges)
    graph_nodes = pd.Index(pd.unique(pd.concat([E["source"], E["target"]], ignore_index=True)))
    df = df[df[ID_COL].isin(graph_nodes)].copy()

    # Node index mapping
    nodes = df[ID_COL].tolist()
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    # Build edge_index (undirected)
    src = E["source"].map(node_to_idx)
    dst = E["target"].map(node_to_idx)
    m = src.notna() & dst.notna()
    src = src[m].astype(int).to_numpy()
    dst = dst[m].astype(int).to_numpy()
    w = E.loc[m, "weight"].to_numpy(dtype=np.float32)

    # Add reverse direction
    edge_index = np.vstack([np.concatenate([src, dst]), np.concatenate([dst, src])])
    edge_weight = np.concatenate([w, w]).astype(np.float32)

    # Node features
    X_mat = df[feat_cols].to_numpy(dtype=np.float32)
    # Simple median impute for any NaNs
    if np.isnan(X_mat).any():
        med = np.nanmedian(X_mat, axis=0)
        inds = np.where(np.isnan(X_mat))
        X_mat[inds] = np.take(med, inds[1])

    # Labels and masks
    y = df[Y_COL].to_numpy(dtype=np.float32)
    y_mask = ~np.isnan(y)  # labeled nodes

    train_mask = df[ID_COL].isin(train_ids).to_numpy() & y_mask
    val_mask   = df[ID_COL].isin(val_ids).to_numpy() & y_mask
    test_mask  = df[ID_COL].isin(test_ids).to_numpy() & y_mask

    data = Data(
        x=torch.tensor(X_mat),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_weight=torch.tensor(edge_weight),
        y=torch.tensor(np.nan_to_num(y, nan=0.0)),
        train_mask=torch.tensor(train_mask),
        val_mask=torch.tensor(val_mask),
        test_mask=torch.tensor(test_mask),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_path)

    print(f"{threshold}/{run}: nodes={data.num_nodes} edges={data.num_edges//2} "
          f"labeled={int(y_mask.sum())} train={int(train_mask.sum())} val={int(val_mask.sum())} test={int(test_mask.sum())} "
          f"features={data.num_node_features} -> {out_path}")

def main():
    threshold = "threshold_0.10"
    runs = ["run_0", "run_1", "run_2", "run_3"]
    for run in runs:
        out = Path("data/processed/pyg") / threshold / f"{run}.pt"
        build_data(threshold, run, out)

if __name__ == "__main__":
    main()

