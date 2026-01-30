# Learning to Predict Material Properties with Limited Data  
*A data-driven study on CO₂ adsorption in Metal–Organic Frameworks (MOFs)*

---

## 1. Why this problem matters (real-world motivation)
- Reducing atmospheric CO₂ is a central challenge in climate science, clean energy, and sustainable industrial processes.
- Metal–Organic Frameworks (MOFs) are porous materials that can be useful for:
  - carbon capture and storage,
  - gas separation and purification,
  - catalysis and energy-efficient chemical processes.
- The bottleneck is **material screening**:
  - tens of thousands of MOF structures exist,
  - reliable CO₂ adsorption measurements exist for only a small subset,
  - generating new labels is slow/expensive (experiments) or computationally heavy (simulation).
- Practical question:
  - **How can we prioritize which MOFs are worth deeper experimental or computational evaluation when labeled data is limited?**
- This project tackles that using **few-shot learning ideas + graph-based learning**.

---

## 2. Core idea of the project
- The approach combines three information sources:
  - **Structural descriptors** of MOFs (e.g., pore limiting diameter, cavity diameter)
  - **Sparse CO₂ adsorption labels** (CO₂ uptake at 298 K and 1 bar)
  - **MOF similarity graph** where MOFs are nodes and edges represent structural similarity
- Key assumption:
  - **Structurally similar MOFs tend to have similar adsorption behavior.**
- Why this helps:
  - when labels are scarce, the similarity graph can share information across related structures.

---

## 3. Data sources used
- **MOF structural dataset**
  - MOFCSD-style table containing:
    - unique MOF identifiers (**refcode**),
    - structural descriptors (pore/cavity related features).
- **CO₂ adsorption labels**
  - extracted from **CRAFTED** isotherm files,
  - fixed condition: **298 K and 1 bar** for comparability,
  - after strict ID matching and filtering:
    - **363 MOFs** have usable CO₂ uptake labels.
- **MOF similarity graphs**
  - sourced from graph construction work and sparsified via **BlackHole** outputs,
  - tested at multiple sparsification thresholds:
    - **threshold_0.90, threshold_0.10, threshold_0.00**
  - multiple runs per threshold used for robustness.

---

## 4. What the pipeline does (high-level flow)
- Inspect / clean MOF structure table
- Parse isotherm files and build a CO₂ label table
- Match labels to MOF refcodes (ID overlap filtering)
- Create fixed train/val/test splits for labeled MOFs
- Load graph edges for different thresholds (and multiple runs)
- Diagnose label coverage inside graphs + connectivity between splits
- Train baselines (tabular + graph-derived features)
- Train GNNs (and shuffled-graph controls)
- Summarize results and generate plots
- Produce a ranked shortlist of candidate MOFs + confidence
- Show everything in a Streamlit dashboard

---

## 5. Few-shot learning angle (why this is “limited data”)
- Scale mismatch:
  - ~14k MOFs structurally available,
  - only **363** with adsorption labels.
- Implications:
  - standard data-hungry models are not appropriate,
  - evaluation must avoid misleading leakage,
  - the goal is practical prioritization and learning efficiency.
- What is explicitly studied:
  - effect of graph density,
  - effect of labeled-node connectivity,
  - real graph structure vs randomized structure.

---

## 6. Models explored
- **Descriptor-based baselines**
  - Ridge regression
  - kNN regression
  - Random Forest regression
- **Graph-aware baseline (stronger)**
  - uses **BlackHole node features** (graph-derived features) to predict CO₂ uptake,
  - trained across multiple runs for stability.
- **Graph Neural Network (GNN)**
  - GCN trained on MOF similarity graphs,
  - evaluated across runs,
  - compared against shuffled graph control.
- **Control experiment: shuffled graph**
  - edges randomized but node features unchanged,
  - tests whether graph structure adds real signal vs noise.

---

## 7. Key findings (what I observed)
- **A) Threshold choice changes the “learning problem”**
  - At **threshold_0.90**:
    - only ~36–39 labeled MOFs per run → too little supervised signal for graph learning.
  - At **threshold_0.10**:
    - ~271–277 labeled MOFs per run → meaningful for GNN training.
  - At **threshold_0.00**:
    - ~307–313 labeled MOFs per run → highest coverage but graph is larger/denser.
  - Interpretation:
    - threshold selection controls how usable the graph is under few-shot supervision.
- **B) Connectivity audit explains why GNNs can struggle**
  - At 0.90:
    - test nodes rarely connect to train nodes → low exposure.
  - At 0.10 / 0.00:
    - many test nodes have train-neighbors → information can propagate.
- **C) Baselines vs GNN results (current status)**
  - descriptor-only baselines are limited (only a few numeric structure features),
  - graph-derived feature baseline improves validation but test remains challenging,
  - GNN vs shuffled shows only small separation at 0.10 so far, suggesting:
    - node features may dominate,
    - graph signal may be weak/noisy,
    - or the task is hard at this label scale.
  - conclusion:
    - emphasis is on diagnostics + robustness, not claiming “GNN wins”.

---

## 8. End result (practical output)
- Concrete outputs produced:
  - **Top-K MOF shortlist (refcodes)** ranked by predicted CO₂ uptake
  - **Confidence score** from agreement/variation across graph runs
  - **Ranking plot + table** (decision-oriented, not only RMSE)
- Real-world mapping:
  - supports decision-making for experimental/computational screening.
  ### Top-K Candidate Interpretation
  - Identifies top-5 MOF candidates based on predicted CO₂ uptake and confidence scores
  - Visualized with error bars (screen score ± uncertainty)
  - Automatically interpreted based on pore structure and model source (RF vs RF+GNN)
  - Accessible from final section of Streamlit app

---

## 9. Streamlit dashboard (what it shows)
- The Streamlit app presents the pipeline end-to-end:
  - problem statement + goal,
  - data mapping summary (IDs + label overlap),
  - threshold diagnostics (coverage + connectivity),
  - baseline vs GNN vs shuffled comparison,
  - Top-K ranking + confidence visualization.
- Run command:
  - `python -m streamlit run streamlit_app/app.py`

---

## 10. Repository structure (overview)
- Folder layout:
  - `src/`
    - `data/` — data inspection, label extraction, splits
    - `graphs/` — graph export, coverage & connectivity analysis
    - `models/` — baselines, GNN, candidate ranking
    - `viz/` — visualization utilities
  - `notebooks/`
    - `00_overview_pipeline.ipynb`
    - `01_data_mapping_and_labels.ipynb`
    - `02_splits_and_fewshot.ipynb`
    - `04_baselines.ipynb`
    - `05_gnn_vs_shuffled.ipynb`
    - `06_summary_tables.ipynb`
  - `streamlit_app/` — interactive dashboard
  - `outputs/figures/` — generated plots and visuals
  - `data/processed/` — small reproducible artifacts
  - `README.md`
- Note:
  - large external datasets are excluded from version control and must be downloaded separately.

---

## 11. How to run the pipeline
- Run end-to-end:
  - `python src/run_all.py`
- For exploration:
  - run notebooks in order.

---

## 12. Related research foundations
- External inspirations used:
  - **MOFGalaxyNet**
    - large-scale MOF similarity network concept (MOFs as nodes, similarity as edges).
  - **BlackHole**
    - graph sparsification framework used to test thresholds and run-to-run robustness.
- What these foundations supported in this project:
  - similarity-graph representation,
  - threshold-based graph experiments,
  - robustness across multiple graph runs,
  - real vs shuffled control setup.

---

## 13. What this project demonstrates
- End-to-end pipeline for limited-label material property learning
- Few-shot constraints handled explicitly
- Graph diagnostics to avoid misleading conclusions
- Baselines + controls for credibility
- Practical output: ranked shortlist of candidate MOFs
- Interactive dashboard to communicate results clearly

http://localhost:8505/
