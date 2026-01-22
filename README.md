# Learning to Predict Material Properties with Limited Data: Few-Shot Learning

## Overview

This repository contains the code and experiments for my Master’s thesis titled:

**“Learning to Predict Material Properties with Limited Data: Few-Shot Learning”**

The thesis focuses on predicting **CO₂ gas adsorption capacity in Metal–Organic Frameworks (MOFs)** under realistic conditions where **labeled data is scarce**. In practice, only a small subset of known MOFs have experimentally or computationally measured adsorption values, which makes standard machine-learning approaches difficult to apply.

The objective of this work is not to build a perfect predictor, but to study **how machine-learning models behave when trained with very limited labeled data**, and to evaluate whether **graph-based learning** can provide additional benefit in such few-shot regimes.

---

## Motivation and Real-World Context

Metal–Organic Frameworks are promising materials for applications such as:

- carbon capture and storage (CO₂ adsorption),
- gas separation,
- clean energy and environmental sustainability.

However, thousands of MOF structures exist, and **testing each candidate experimentally or via simulation is expensive and time-consuming**. As a result:

- adsorption labels are available for only a small fraction of MOFs,
- existing datasets are sparse and biased,
- learning meaningful structure–property relationships is challenging.

This thesis addresses the following practical question:

> *How can we predict or prioritize promising MOFs for CO₂ adsorption when only a limited number of labeled examples are available?*

---

## Core Idea of the Thesis

This work studies **few-shot learning** for material property prediction by combining:

1. **MOF structural descriptors** (geometric features),
2. **Graph-based similarity information** between MOFs,
3. **Controlled baselines and diagnostics** to understand when learning is feasible and when it fails.

Rather than assuming that more complex models automatically perform better, the thesis explicitly analyzes:

- how many labeled MOFs are actually present in a graph,
- how graph sparsification affects supervision and connectivity,
- whether graph structure provides information beyond standard descriptors.

---

## Datasets Used

### 1. MOF Structural Data
- **MOFCSD.csv**  
  Used as the master table of MOFs, identified by CSD refcodes, with geometric descriptors such as:
  - Largest Cavity Diameter (LCD)
  - Pore Limiting Diameter (PLD)
  - Largest Free Sphere (LFS)

### 2. CO₂ Adsorption Labels
- **CRAFTED 2.0.1 dataset** (simulation-derived isotherms)  
  CO₂ uptake values at fixed conditions (298 K, 1 bar) are extracted and mapped to MOFCSD refcodes.

The use of simulation-derived data allows consistent and standardized adsorption labels, which is important for systematically analyzing few-shot and graph-based learning effects.

### 3. Graph Data
- **BlackHole sparsified MOF similarity graphs**  
  MOFs are connected based on structural similarity. Graphs are provided at multiple sparsification thresholds (e.g., 0.90, 0.10, 0.00), which strongly influence:
  - graph density,
  - number of labeled nodes present in the graph,
  - feasibility of graph-based learning.

---

## Project Structure
MOF-Gas-absorption/
│
├── src/
│ ├── data/ # Data loading, cleaning, label construction, splits
│ ├── graphs/ # Graph preparation, sparsification analysis, diagnostics
│ ├── models/ # Baselines and graph-based model training
│ └── utils/ # Shared helper functions
│
├── data/
│ ├── raw/ # Original input files (not committed)
│ ├── external/ # External datasets (CRAFTED, BlackHole, etc.)
│ └── processed/ # Cleaned datasets, labels, splits
│
├── graphs/ # Exported graph edge lists by threshold/run
├── outputs/ # Metrics, predictions, diagnostic tables
│
└── README.md


Large datasets and generated artifacts are excluded from version control to keep the repository lightweight and reproducible.

---

## Methodology (High-Level)

1. **Label construction**
   - Extract CO₂ uptake values from the CRAFTED dataset.
   - Match adsorption labels to MOFCSD refcodes.
   - Result: a limited set of labeled MOFs under fixed conditions.

2. **Few-shot data splits**
   - Create fixed train/validation/test splits over labeled MOFs.
   - Reuse the same splits across all experiments for fair comparison.

3. **Baseline models**
   - Geometry-only regression models.
   - Descriptor-based machine-learning baselines.
   - Stronger baselines using BlackHole node features without graph learning.

4. **Graph analysis**
   - Study how sparsification thresholds affect:
     - label coverage in the graph,
     - train–test connectivity,
     - potential information leakage.

5. **Graph-based learning**
   - Train graph neural networks under controlled conditions.
   - Compare performance against feature-only baselines and shuffled-graph controls.

---

## Focus of the Thesis

- Learning under **limited supervision**
- Evaluating **few-shot learning behavior** in materials science
- Understanding **when graph structure helps and when it does not**
- Emphasizing **methodological clarity and interpretability** over raw performance

---

## Status

- Data preprocessing and label construction: completed  
- Baseline models and diagnostics: completed  
- Graph preparation across thresholds: completed  
- Graph neural network training: ongoing  

---

## Notes

This repository is developed as part of an academic Master’s thesis.  
The emphasis is on reproducibility, careful analysis, and realistic assumptions rather than claiming state-of-the-art performance.

