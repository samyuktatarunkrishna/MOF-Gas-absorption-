import pandas as pd
from pathlib import Path

ID_COL = "refcode"
MOF_PATH = Path("data/raw/MOFCSD.csv")

BLACKHOLE_BASE = Path("data/external/repos/BlackHole/sparsified_graphs")
OUT_BASE = Path("graphs")

SRC_COL, DST_COL, W_COL = "source", "target", "weight"

def export_threshold(threshold: str):
    src_root = BLACKHOLE_BASE / f"threshold_{threshold}" / "method_blackhole"
    if not src_root.exists():
        print(f"[threshold {threshold}] missing folder: {src_root}")
        return

    mof = pd.read_csv(MOF_PATH)
    valid_ids = set(mof[ID_COL].astype(str).tolist())

    run_dirs = sorted([p for p in src_root.glob("run_*") if p.is_dir()])
    print(f"\n=== threshold_{threshold} ===")
    print("Detected runs:", [p.name for p in run_dirs])

    for run_dir in run_dirs:
        edge_files = list(run_dir.glob("BH_edges*.csv")) + list(run_dir.glob("*edges*.csv"))
        if not edge_files:
            print(run_dir.name, "no edge csv -> skipped")
            continue

        edge_path = edge_files[0]
        edges = pd.read_csv(edge_path)

        if SRC_COL not in edges.columns or DST_COL not in edges.columns:
            print(run_dir.name, "unexpected columns:", list(edges.columns), "-> skipped")
            continue

        edges[SRC_COL] = edges[SRC_COL].astype(str)
        edges[DST_COL] = edges[DST_COL].astype(str)

        kept = edges[edges[SRC_COL].isin(valid_ids) & edges[DST_COL].isin(valid_ids)].copy()

        out_dir = OUT_BASE / f"threshold_{threshold}" / run_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / "edges.csv"
        cols = [SRC_COL, DST_COL] + ([W_COL] if W_COL in kept.columns else [])
        kept[cols].to_csv(out_path, index=False)

        unique_nodes = pd.unique(pd.concat([kept[SRC_COL], kept[DST_COL]], ignore_index=True))
        print(f"{run_dir.name}: raw_edges={len(edges)} kept_edges={len(kept)} unique_nodes={len(unique_nodes)} saved={out_path}")

def main():
    # start with the two that matter most for your thesis
    for t in ["0.00", "0.10", "0.90"]:
        export_threshold(t)

if __name__ == "__main__":
    main()