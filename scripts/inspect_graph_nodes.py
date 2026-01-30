from pathlib import Path
import pandas as pd

GRAPHS_DIR = Path("graphs/threshold_0.10")
all_nodes = set()

for run_dir in GRAPHS_DIR.glob("run_*/edges.csv"):
    df = pd.read_csv(run_dir)
    for col in ["source", "target", "src", "dst", "u", "v"]:
        if col in df.columns:
            all_nodes.update(df[col].astype(str).str.strip().str.upper().unique())

print(f"Total unique nodes across graphs: {len(all_nodes)}")

# Save for checking
pd.Series(sorted(all_nodes)).to_csv("outputs/debug/all_graph_nodes.csv", index=False)
