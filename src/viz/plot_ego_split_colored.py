from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import pandas as pd

THRESHOLD = "threshold_0.10"
RUN = "run_0"
EDGES_CSV = Path(f"graphs/{THRESHOLD}/{RUN}/edges.csv")
SPLITS_JSON = Path("data/processed/splits_seed42.json")

OUT_DIR = Path("outputs/figures/06_graph_visuals")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / f"B2_ego_splitcolored_{THRESHOLD}_{RUN}.png"

# ---- pick a center node for the ego-graph (change this to any node you want) ----
CENTER_NODE = "PIFPIE"   # <- your test labeled example; replace if needed
HOPS = 1                 # 1-hop ego graph is usually enough (2-hop gets very dense)
MAX_NODES = 350          # safety cap to keep plots readable

def load_splits(path: Path):
    s = json.loads(path.read_text())
    train = set(s["train"])
    val = set(s["val"])
    test = set(s["test"])
    return train, val, test

def node_color(n, train_ids, val_ids, test_ids):
    if n in train_ids:
        return "tab:green"
    if n in val_ids:
        return "tab:orange"
    if n in test_ids:
        return "tab:red"
    return "tab:blue"  # unlabeled

def main():
    assert EDGES_CSV.exists(), f"Missing {EDGES_CSV}"
    assert SPLITS_JSON.exists(), f"Missing {SPLITS_JSON}"

    train_ids, val_ids, test_ids = load_splits(SPLITS_JSON)

    # Load edges
    e = pd.read_csv(EDGES_CSV)
    # expected columns: source, target, weight (weight optional)
    if "weight" not in e.columns:
        e["weight"] = 1.0

    # Build graph
    G = nx.from_pandas_edgelist(e, "source", "target", edge_attr="weight", create_using=nx.Graph())

    if CENTER_NODE not in G:
        raise ValueError(
            f"CENTER_NODE={CENTER_NODE} not in graph. "
            f"Pick a node that exists in {EDGES_CSV}."
        )

    # Ego graph
    H = nx.ego_graph(G, CENTER_NODE, radius=HOPS)

    # cap size if too large
    if H.number_of_nodes() > MAX_NODES:
        # keep center + top-degree neighbors (more stable than random)
        deg = dict(H.degree())
        keep = sorted(deg, key=lambda n: deg[n], reverse=True)[:MAX_NODES-1]
        keep = [CENTER_NODE] + [n for n in keep if n != CENTER_NODE]
        H = H.subgraph(keep).copy()

    # Layout on FINAL subgraph (critical)
    pos = nx.spring_layout(H, seed=42)

    # Colors per node (split-aware)
    colors = [node_color(n, train_ids, val_ids, test_ids) for n in H.nodes()]

    plt.figure(figsize=(10, 7))

    # Draw edges first
    nx.draw_networkx_edges(H, pos, alpha=0.18, width=0.6)

    # Draw nodes
    nx.draw_networkx_nodes(H, pos, node_color=colors, node_size=70, alpha=0.92)

    # Emphasize center node
    nx.draw_networkx_nodes(
        H, pos, nodelist=[CENTER_NODE],
        node_color="yellow", edgecolors="black", linewidths=1.2,
        node_size=220, alpha=1.0
    )

    # Real legend (matches actual encoding)
    legend_handles = [
        mpatches.Patch(color="tab:green", label="Train"),
        mpatches.Patch(color="tab:orange", label="Validation"),
        mpatches.Patch(color="tab:red", label="Test"),
        mpatches.Patch(color="tab:blue", label="Unlabeled"),
        mpatches.Patch(color="yellow", label="Center node"),
    ]
    plt.legend(handles=legend_handles, loc="best", fontsize=9)

    plt.title(f"Ego-graph around {CENTER_NODE} ({THRESHOLD}/{RUN})\nColors show split membership")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=250)
    plt.close()

    print("Saved:", OUT_PATH)

if __name__ == "__main__":
    main()
