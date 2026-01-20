# MOF Gas absorption

This repository contains a reproducible data + modeling pipeline for predicting **gas adsorption (CO₂ uptake)** in **Metal–Organic Frameworks (MOFs)** using:
- a **MOF descriptor table** (MOFCSD) as node-level features, and
- **sparsified MOF graphs** (BlackHole runs) as graph structure, and
- **CRAFTED simulated isotherms** as supervised labels (CO₂ uptake at a fixed condition).

The current pipeline builds a clean supervised dataset by enforcing **strict MOF ID matching** (`refcode`) between MOFCSD and CRAFTED, and prepares graph inputs for multiple graph realizations (`run_0..run_3`) to support robustness analysis (mean ± std).

---

## Project status (current)
✅ MOFCSD inspected; master MOF identifier confirmed as `refcode`  
✅ CRAFTED labels created for **CO₂ @ 298 K, 1 bar** with strict ID overlap  
✅ Labels merged into MOFCSD (`MOFCSD_with_co2_labels.csv`)  
✅ Fixed split created (`splits_seed42.json`)  
✅ BlackHole run graphs exported to standardized edge lists (`graphs/run_i/edges.csv`)

Next planned stages:
- build labeled subgraphs per run (optional but thesis-strong),
- train baselines (ridge / RF / kNN),
- train GNN from scratch vs transfer learning,
- few-shot experiments and aggregate results across runs.

---

## Folder structure (high-level)

