import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from pathlib import Path

IN_PATH = Path("data/processed/MOFCSD_with_co2_labels.csv")
OUT_PATH = Path("graphs/knn_threshold/run_0/edges.csv")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

FEATURES = ["Largest Cavity Diameter", "Pore Limiting Diameter", "Largest Free Sphere"]

df = pd.read_csv(IN_PATH)
df = df[["refcode"] + FEATURES].dropna()
df["refcode"] = df["refcode"].astype(str).str.strip().str.upper()

X = df[FEATURES].astype(float).values
ids = df["refcode"].tolist()

k = 5  # Number of neighbors
nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")  # +1 to exclude self
nn.fit(X)
distances, indices = nn.kneighbors(X)

edges = []
for i, neighbors in enumerate(indices):
    for j, idx in enumerate(neighbors[1:]):  # skip self at index 0
        src = ids[i]
        dst = ids[idx]
        weight = 1.0 - distances[i][j+1]  # similarity = 1 - cosine distance
        edges.append((src, dst, weight))

edges_df = pd.DataFrame(edges, columns=["source", "target", "weight"])
edges_df.to_csv(OUT_PATH, index=False)
print(f"[build_knn_graph] Saved: {OUT_PATH}")
