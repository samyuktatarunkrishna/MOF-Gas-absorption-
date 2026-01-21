import pandas as pd
from pathlib import Path
import json

ID_COL = "refcode"

LABELED_TABLE = Path("data/processed/MOFCSD_with_co2_labels.csv")
SPLITS_PATH = Path("data/processed/splits_seed42.json")
GRAPHS_DIR = Path("graphs")
OUT_TABLE = Path("data/processed/run_labeled_subgraph_stats.csv")

def main():
    df = pd.read_csv(LABELED_TABLE)
    labeled_ids = set(df.loc[df["co2_uptake"].notna(), ID_COL].astype(str))

    splits = json.loads(SPLITS_PATH.read_text())
    split_ids = set(splits["train"] + splits["val"] + splits["test"])
    # sanity: labeled_ids and split_ids should match size (both 363)
    print("Labeled IDs:", len(labeled_ids))
    print("Split IDs:", len(split_ids))

    stats = []

    run_dirs = sorted([p for p in GRAPHS_DIR.glob("run_*") if p.is_dir()])
    for run_dir in run_dirs:
        edges_path = run_dir / "edges.csv"
        if not edges_path.exists():
            print(run_dir.name, "missing edges.csv -> skipped")
            continue

        edges = pd.read_csv(edges_path)
        edges["source"] = edges["source"].astype(str)
        edges["target"] = edges["target"].astype(str)

        nodes_in_run = set(pd.unique(pd.concat([edges["source"], edges["target"]], ignore_index=True)))

        labeled_in_run = sorted(nodes_in_run & labeled_ids)
        labeled_in_run_set = set(labeled_in_run)

        # edges where both endpoints are labeled
        edges_labeled = edges[edges["source"].isin(labeled_in_run_set) & edges["target"].isin(labeled_in_run_set)].copy()

        # save artifacts
        (run_dir / "labeled_nodes.txt").write_text("\n".join(labeled_in_run))
        edges_labeled.to_csv(run_dir / "edges_labeled.csv", index=False)

        stats.append({
            "run": run_dir.name,
            "unique_nodes_in_run": len(nodes_in_run),
            "labeled_nodes_in_run": len(labeled_in_run),
            "edges_total": len(edges),
            "edges_labeled": len(edges_labeled),
        })

        print(f"{run_dir.name}: nodes={len(nodes_in_run)} labeled_nodes={len(labeled_in_run)} edges={len(edges)} labeled_edges={len(edges_labeled)}")

    out = pd.DataFrame(stats)
    out.to_csv(OUT_TABLE, index=False)
    print("Saved stats table:", OUT_TABLE)

if __name__ == "__main__":
    main()