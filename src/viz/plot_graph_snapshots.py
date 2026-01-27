# src/viz/plot_graph_snapshots.py
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

OUT = Path("outputs/figures/06_graph_visuals")
OUT.mkdir(parents=True, exist_ok=True)

THRESHOLD = "threshold_0.10"
RUN = "run_0"

EDGE_PATH = Path(f"graphs/{THRESHOLD}/{RUN}/edges.csv")
LABELS_PATH = Path("data/processed/MOFCSD_with_co2_labels.csv")
SPLIT_PATH = Path("data/processed/splits_seed42.json")


def load_splits(p):
    d = json.loads(p.read_text())
    return set(d["train"]), set(d["val"]), set(d["test"])


def node_split(n, train, val, test):
    if n in train:
        return "train"
    if n in val:
        return "val"
    if n in test:
        return "test"
    return "unlabeled"


def draw_graph(G, node_group, title, outpath, max_nodes=300):
    # Downsample graph FIRST
    if G.number_of_nodes() > max_nodes:
        keep = random.sample(list(G.nodes()), max_nodes)
        G = G.subgraph(keep).copy()

    # Compute layout on FINAL graph
    pos = nx.spring_layout(G, seed=42)

    # Rebuild groups ONLY for nodes in G
    groups = {"train": [], "val": [], "test": [], "unlabeled": []}
    for n in G.nodes():
        groups[node_group.get(n, "unlabeled")].append(n)

    plt.figure(figsize=(9, 7))
    nx.draw_networkx_edges(G, pos, alpha=0.25, width=0.7)

    for gname, nodes in groups.items():
        if nodes:
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=nodes,
                node_size=45,
                alpha=0.85,
                label=gname,
            )

    plt.title(title)
    plt.axis("off")
    plt.legend(markerscale=1.4, fontsize=9)
    plt.tight_layout()
    plt.savefig(outpath, dpi=250)
    plt.close()

    print("Saved:", outpath)



def main():
    ed = pd.read_csv(EDGE_PATH)
    G = nx.from_pandas_edgelist(ed, "source", "target")

    labels = pd.read_csv(LABELS_PATH)
    labeled = set(labels.query("co2_uptake.notna()", engine="python")["refcode"])

    train, val, test = load_splits(SPLIT_PATH)

    groups = {"train": [], "val": [], "test": [], "unlabeled": []}
    for n in G.nodes():
        groups[node_split(n, train, val, test)].append(n)

    # A) Labeled-only subgraph
    Glab = G.subgraph(labeled & G.nodes()).copy()
    draw_graph(
        Glab,
        groups,
        f"Labeled-only subgraph ({THRESHOLD}/{RUN})",
        OUT / f"A_labeled_subgraph_{THRESHOLD}_{RUN}.png",
    )

    # B) Ego graph around a test node
    test_nodes = list(test & Glab.nodes())
    if test_nodes:
        center = random.choice(test_nodes)
        ego = nx.ego_graph(G, center, radius=1)
        draw_graph(
            ego,
            groups,
            f"Ego-graph around TEST node {center}",
            OUT / f"B_ego_graph_{center}_{THRESHOLD}_{RUN}.png",
        )

    # C) Degree distribution
    degrees = [d for _, d in G.degree()]
    plt.figure(figsize=(7, 4))
    plt.hist(degrees, bins=40)
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.title(f"Degree distribution ({THRESHOLD}/{RUN})")
    plt.tight_layout()
    plt.savefig(OUT / f"C_degree_distribution_{THRESHOLD}_{RUN}.png")
    plt.close()

    # D) Component sizes
    comps = sorted([len(c) for c in nx.connected_components(Glab)], reverse=True)
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(comps) + 1), comps, marker="o")
    plt.xlabel("Component rank")
    plt.ylabel("Size")
    plt.title(f"Labeled component sizes ({THRESHOLD}/{RUN})")
    plt.tight_layout()
    plt.savefig(OUT / f"D_component_sizes_{THRESHOLD}_{RUN}.png")
    plt.close()


if __name__ == "__main__":
    main()
