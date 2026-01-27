# Learning to Predict Material Properties with Limited Data  
*A data-driven study on CO₂ adsorption in Metal–Organic Frameworks (MOFs)*

---

## 1. Why this problem matters (real-world motivation)
Reducing atmospheric CO₂ is one of the key scientific challenges related to climate change and sustainable energy systems.

Metal–Organic Frameworks (MOFs) are promising materials for applications such as:
- carbon capture and storage
- gas separation and purification
- catalysis and clean energy processes

However, the main bottleneck is material screening.

There are tens of thousands of known MOF structures, but:
- experimental adsorption measurements are expensive and slow
- high-fidelity simulations are computationally heavy
- only a very small fraction of MOFs have measured CO₂ uptake data

From a scientist’s perspective, this raises a practical question:  
**How can we identify promising MOFs for CO₂ adsorption when labeled experimental data is extremely limited?**

This project explores that question using data-driven and graph-based learning methods.

---

## 2. Core idea of the project
The key idea is to combine three types of information:
- Structural descriptors of MOFs (e.g., pore size, cavity diameter)
- Sparse experimental CO₂ adsorption labels (CO₂ uptake at 298 K and 1 bar, available for only a few hundred MOFs)
- Graph-based similarity information between MOFs, derived from large-scale structural comparison methods

Instead of treating each MOF independently, the project assumes that:
- MOFs with similar structures are likely to show similar adsorption behavior

This assumption allows information to be shared across a graph, which is especially useful when labels are scarce.

---

## 3. Data sources used
Three main inputs are used in the workflow:

- MOF structural dataset  
  - A large MOF structure dataset (MOFCSD-style table) containing:
    - unique MOF identifiers (refcodes)
    - basic structural descriptors such as pore diameters

- CO₂ adsorption labels  
  - CO₂ isotherm data extracted from a curated adsorption dataset (CRAFTED)
  - Only measurements at 298 K and 1 bar are retained for consistency
  - After strict ID matching and filtering, **363 MOFs** have usable CO₂ labels

- Graph data (MOF similarity networks)  
  - MOF similarity graphs are taken from a graph-construction pipeline based on structural similarity
  - These graphs are sparsified using different thresholds, producing networks of varying density
  - Multiple graph realizations (“runs”) are used to test robustness

This workflow is inspired by earlier graph-based MOF studies that use large similarity networks to propagate information across chemically related structures.

---

## 4. What the pipeline does (high-level flow)
- Inspect and clean MOF structural data
- Extract CO₂ adsorption labels from raw isotherm files
- Match adsorption labels to MOF structures
- Create fixed train/validation/test splits (few-shot scenario)
- Load MOF similarity graphs at different thresholds
- Analyze labeled-node coverage inside each graph
- Audit connectivity between training and test nodes
- Train baseline ML models
- Train graph-based neural networks
- Compare real graphs vs shuffled (control) graphs
- Summarize results across multiple runs

Each step is implemented as a separate script or notebook for clarity and reproducibility.

---

## 5. Why “few-shot” learning is central here
Although the full MOF dataset contains over 14,000 structures, only **363 MOFs** have CO₂ adsorption labels.

This means:
- traditional data-hungry models are not suitable
- performance must be evaluated carefully
- the goal is not absolute prediction accuracy, but learning efficiency under limited supervision

The project explicitly studies how performance changes when:
- graph density increases
- labeled nodes become more connected
- graph structure is preserved versus randomized

---

## 6. Models explored
Descriptor-based baselines:
- Linear regression (Ridge)
- k-Nearest Neighbors
- Random Forests

Graph-aware baselines:
- Models that use graph-derived features (e.g., neighborhood statistics)

Graph Neural Networks (GNNs):
- Graph Convolutional Networks (GCNs)
- Trained on MOF similarity graphs
- Evaluated across multiple graph runs

Control experiment (shuffled graphs):
- Edge structure is randomized while keeping node features unchanged
- Used to test whether graph structure itself carries useful signal

---

## 7. How results are interpreted
Instead of focusing only on final error values, the analysis emphasizes:
- how label coverage changes with graph sparsification
- how much information can propagate from training to test nodes
- whether real graphs outperform shuffled ones
- stability of results across multiple runs

This helps answer why a model behaves the way it does, not just how well it performs.

---

## 8. Repository structure (overview)
- `src/`
  - `data/` — data inspection, label extraction, splits
  - `graphs/` — graph export, coverage & connectivity analysis
  - `models/` — baselines and GNN training scripts
  - `viz/` — visualization and graph plotting utilities
- `notebooks/`
  - `00_overview_pipeline.ipynb`
  - `01_data_mapping_and_labels.ipynb`
  - `02_splits_and_fewshot.ipynb`
  - `04_baselines.ipynb`
  - `05_gnn_vs_shuffled.ipynb`
  - `06_summary_tables.ipynb`
- `data/processed/` — small reproducible artifacts (splits, stats)
- `outputs/figures/` — generated plots and visual diagnostics
- `README.md`

Large external datasets are intentionally excluded and must be downloaded separately.

---

## 9. What this project demonstrates
- A complete end-to-end workflow for learning material properties with limited labels
- Careful handling of data scarcity and evaluation bias
- Meaningful use of graph structure in materials science
- Clear comparison between classical ML, graph-based features, and GNNs
- Emphasis on understanding, not black-box performance

---

## 10. How to run the pipeline
Each stage can be executed independently.

For a full run:
- `python src/run_all.py`

Notebooks can be run in order for exploratory analysis and visualization.

---

## 11. Related Research Repositories and Foundations
This project builds upon ideas, data structures, and research directions from existing open-source work in the MOF and materials-informatics community, particularly repositories that focus on MOF similarity graphs and graph-based learning.

The following repositories were referenced during the design and validation of this workflow:
- MOFGalaxyNet  
  - Provides a large-scale similarity network of MOFs (nodes = MOFs, edges = similarity)
  - Motivated treating MOFs as a graph rather than independent samples

- BlackHole (Graph Sparsification Framework)  
  - Introduces a method for sparsifying large similarity graphs using thresholds and multiple runs
  - Used here to study how sparsification affects label availability, information flow, and learning behavior under limited data

Different sparsification thresholds (e.g., 0.90, 0.10, 0.00) are analyzed to understand the trade-off between:
- graph size
- label coverage
- potential information leakage across data splits

These references helped shape:
- the decision to use similarity graphs instead of purely tabular data
- the threshold-based graph construction strategy
- the evaluation setup comparing real graphs against shuffled (structure-destroyed) graphs
