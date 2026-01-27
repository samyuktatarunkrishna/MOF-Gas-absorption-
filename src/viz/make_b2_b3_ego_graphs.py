from __future__ import annotations

import json
from pathlib import Path
import random

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import pandas as pd

# -----------------------------
# CONFIG: choose threshold/run
# -----------------------------
THRESHOLD = "threshold_0.10"
RUN = "run_0"
HOPS = 1
MAX_NODES = 350  # cap to keep plot readable

EDGE_PATH = Path(f"graphs/{THRESHOLD}/{RUN}/edges.csv")
SPLIT_PATH = Path("data/processed/splits_seed42.json")
LABELS_PATH = Path("data/processed/MOFCSD_with_co2_labels.csv")  # optional filter
AUDIT_PATH = Path("outputs/split_connectivity_audit.csv")        # used to pick B2/B3

OUT_DIR = Path("outputs/figures/06_graph_visuals")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# helpers
# -----------------------------
def load_splits(p: Path):
    s = json.loads(p.read_text())
    return set(s["train"]), set(s["val"]), set(s["test"])


def node_color(n: str, train: set, val: set, test: set) -> str:
    if n in train:
        return "tab:green"
    if n in val:
        return "tab:orange"
    if n in test:
        return "tab:red"
    return "tab:blue"  # unlabeled


def draw_ego(G: nx.Graph, center: str, train: set, val: set, test: set, title: str, outpath: Path):
    H = nx.ego_graph(G, center, radius=HOPS)

    # size safety cap (keep center + highest-degree nodes)
    if H.number_of_nodes() > MAX_NODES:
        deg = dict(H.degree())
        keep = sorted(deg, key=lambda n: deg[n], reverse=True)
        keep = [center] + [n for n in keep if n != center][: MAX_NODES - 1]
        H = H.subgraph(keep).copy()

    pos = nx.spring_layout(H, seed=42)

    colors = [node_color(n, train, val, test) for n in H.nodes()]

    plt.figure(figsize=(10, 7))
    nx.draw_networkx_edges(H, pos, alpha=0.18, width=0.6)
    nx.draw_networkx_nodes(H, pos, node_color=colors, node_size=70, alpha=0.92)

    # highlight center
    nx.draw_networkx_nodes(
        H,
        pos,
        nodelist=[center],
        node_color="yellow",
        edgecolors="black",
        linewidths=1.2,
        node_size=240,
        alpha=1.0,
    )

    legend_handles = [
        mpatches.Patch(color="tab:green", label="Train"),
        mpatches.Patch(color="tab:orange", label="Validation"),
        mpatches.Patch(color="tab:red", label="Test"),
        mpatches.Patch(color="tab:blue", label="Unlabeled"),
        mpatches.Patch(color="yellow", label="Center node"),
    ]
    plt.legend(handles=legend_handles, loc="best", fontsize=9)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outpath, dpi=250)
    plt.close()
    print("Saved:", outpath)


def pick_b2_b3_nodes(train: set, val: set, test: set, labeled: set[str] | None = None):
    """
    Choose:
      - B2: test node with MANY train neighbors (strongly supported)
      - B3: test node with ZERO train neighbors (weakly supported)
    Uses audit file if available, otherwise computes from graph locally later.
    """
    if AUDIT_PATH.exists():
        a = pd.read_csv(AUDIT_PATH)

        # filter to chosen threshold/run
        # your audit columns are: ['threshold','run',...]
        a = a[(a["threshold"] == THRESHOLD) & (a["run"] == RUN)].copy()
        if a.empty:
            raise ValueError(f"No rows for {THRESHOLD}/{RUN} in {AUDIT_PATH}")

        # audit is aggregated, not per-node. So we cannot pick node ids from it directly.
        # We'll fall back to local computation in main().
        return None, None

    return None, None


def main():
    assert EDGE_PATH.exists(), f"Missing edges: {EDGE_PATH}"
    assert SPLIT_PATH.exists(), f"Missing splits: {SPLIT_PATH}"

    train, val, test = load_splits(SPLIT_PATH)

    # optional: labeled set to ensure center nodes have CO2 labels (useful for explanation)
    labeled = None
    if LABELS_PATH.exists():
        df = pd.read_csv(LABELS_PATH)
        labeled = set(df.loc[df["co2_uptake"].notna(), "refcode"].astype(str))

    # Load graph
    e = pd.read_csv(EDGE_PATH)
    if "weight" not in e.columns:
        e["weight"] = 1.0
    G = nx.from_pandas_edgelist(e, "source", "target", edge_attr="weight", create_using=nx.Graph())

    # Candidate test nodes present in graph
    test_nodes = list(set(G.nodes()).intersection(test))
    if labeled is not None:
        test_nodes = list(set(test_nodes).intersection(labeled))  # test + labeled in dataset

    if not test_nodes:
        raise ValueError("No TEST nodes found in this graph (after optional label filter).")

    # For each test node, count how many TRAIN neighbors it has
    train_set = set(train)
    support = []
    for t in test_nodes:
        nbs = set(G.neighbors(t))
        n_train = len(nbs.intersection(train_set))
        support.append((t, n_train))

    # Pick B2: highest train-neighbor count
    support_sorted = sorted(support, key=lambda x: x[1], reverse=True)
    b2_node, b2_k = support_sorted[0]

    # Pick B3: zero-train-neighbor if possible; else pick smallest
    zero = [x for x in support_sorted if x[1] == 0]
    if zero:
        b3_node, b3_k = random.choice(zero)
    else:
        b3_node, b3_k = support_sorted[-1]

    # Create B2
    draw_ego(
        G,
        b2_node,
        train,
        val,
        test,
        title=f"B2 (well-supported): TEST node {b2_node} with {b2_k} TRAIN neighbors\n({THRESHOLD}/{RUN})",
        outpath=OUT_DIR / f"B2_ego_well_supported_{b2_node}_{THRESHOLD}_{RUN}.png",
    )

    # Create B3
    draw_ego(
        G,
        b3_node,
        train,
        val,
        test,
        title=f"B3 (weakly-supported): TEST node {b3_node} with {b3_k} TRAIN neighbors\n({THRESHOLD}/{RUN})",
        outpath=OUT_DIR / f"B3_ego_weak_supported_{b3_node}_{THRESHOLD}_{RUN}.png",
    )

    print("\nSummary:")
    print("B2:", b2_node, "train_neighbors=", b2_k)
    print("B3:", b3_node, "train_neighbors=", b3_k)


if __name__ == "__main__":
    main()
