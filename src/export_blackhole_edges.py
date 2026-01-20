import pandas as pd
from pathlib import Path

ID_COL = "refcode"

BLACKHOLE_BASE = Path("data/external/repos/BlackHole/sparsified_graphs/threshold_0.90/method_blackhole")
OUT_BASE = Path("graphs")
MOF_PATH = Path("data/raw/MOFCSD.csv")

SRC_COL, DST_COL, W_COL = "source", "target", "weight"

def main():
    mof = pd.read_csv(MOF_PATH)
    valid_ids = set(mof[ID_COL].astype(str).tolist())

    run_dirs = sorted([p for p in BLACKHOLE_BASE.glob("run_*") if p.is_dir()])
    print("Detected runs:", [p.name for p in run_dirs])

    for run_dir in run_dirs:
        edge_path = next(run_dir.glob("BH_edges*.csv"))
        edges = pd.read_csv(edge_path)

        edges[SRC_COL] = edges[SRC_COL].astype(str)
        edges[DST_COL] = edges[DST_COL].astype(str)

        kept = edges[edges[SRC_COL].isin(valid_ids) & edges[DST_COL].isin(valid_ids)].copy()

        out_dir = OUT_BASE / run_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "edges.csv"
        kept[[SRC_COL, DST_COL, W_COL]].to_csv(out_path, index=False)

        unique_nodes = pd.unique(pd.concat([kept[SRC_COL], kept[DST_COL]], ignore_index=True))
        print(f"{run_dir.name}: raw_edges={len(edges)} kept_edges={len(kept)} unique_nodes={len(unique_nodes)} saved={out_path}")

if __name__ == "__main__":
    main()