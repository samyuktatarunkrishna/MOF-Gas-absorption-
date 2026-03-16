# Learning to Predict Material Properties with Limited Data  
### Few-Shot Learning for CO₂ Adsorption in Metal–Organic Frameworks (MOFs)

This repository contains the code and dashboard for the thesis project:

**“Learning to Predict Material Properties with Limited Data: Few-Shot Learning for Gas Absorption in MOFs.”**

The goal of this project is to study whether machine learning models can predict **CO₂ adsorption capacity in MOFs** when only a small number of labeled materials are available.

The pipeline combines:

- structural descriptors of MOFs  
- sparse CO₂ adsorption labels  
- similarity graphs between MOFs  

to evaluate prediction performance and screening under **limited labeled data conditions**.

---

# Repository Overview

```
MOF-Gas-absorption/

src/                    core pipeline scripts
notebooks/              step-by-step analysis notebooks
streamlit_app/          Streamlit dashboard
outputs/figures/        generated plots
data/processed/         small reproducible artifacts

requirements.txt
README.md
```

Main notebooks used in the thesis:

```
00_overview_pipeline.ipynb
01_data_mapping_and_labels.ipynb
02_splits_and_fewshot.ipynb
04_baselines.ipynb
05_gnn_vs_shuffled.ipynb
06_summary_tables.ipynb
```

---

# Environment Setup

The project was developed using **Python 3.10+**.

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment.

Mac / Linux

```bash
source venv/bin/activate
```

Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Additional packages used for visualization and the dashboard:

```bash
pip install streamlit matplotlib jupyter
```

---

# Required External Data

Large datasets are **not stored in this repository** and must be downloaded separately.

## MOF Structural Data

Clone the MOFGalaxyNet repository:

```bash
git clone https://github.com/MehrdadJalali-AI/MOFGalaxyNet
```

Relevant structural descriptor files should be placed in:

```
data/raw/descriptors/
```

---

## MOF Similarity Graphs

Clone the BlackHole repository:

```bash
git clone https://github.com/MehrdadJalali-AI/BlackHole
```

Copy the sparsified graph folders into:

```
data/raw/graphs/
```

Example graph folders:

```
threshold_0.90
threshold_0.10
threshold_0.00
```

---

## CO₂ Adsorption Labels

The CO₂ adsorption dataset used in this project corresponds to **CO₂ uptake at 298 K and 1 bar** extracted from isotherm data.

Place the label table in:

```
data/raw/labels/
```

---

# Expected Folder Structure

After downloading the required datasets, the directory structure should look like this:

```
MOF-Gas-absorption/

data/
 └── raw/
     ├── descriptors/
     │    MOF structural descriptor files
     │
     ├── graphs/
     │    threshold_0.90/
     │    threshold_0.10/
     │    threshold_0.00/
     │
     └── labels/
          CO2 uptake dataset

src/
notebooks/
streamlit_app/

requirements.txt
README.md
```

---

# Running the Pipeline

From the project root directory run:

```bash
python src/run_all.py
```

This script performs:

- data alignment and label matching  
- few-shot split generation  
- baseline model training  
- graph-based model evaluation  
- export of result artifacts and figures  

---

# Running the Streamlit Dashboard

The Streamlit dashboard visualizes the pipeline and results.

Run:

```bash
streamlit run streamlit_app/app.py
```

The dashboard displays:

- dataset overview and label coverage  
- graph connectivity diagnostics  
- baseline vs graph model comparisons  
- Top-K MOF candidate ranking  

The app will launch locally at:

```
http://localhost:8505/
```

---

# External Research Foundations

This project builds on the following external repositories:

**MOFGalaxyNet**

https://github.com/MehrdadJalali-AI/MOFGalaxyNet

Provides large-scale MOF structural data and similarity graph concepts.

**BlackHole**

https://github.com/MehrdadJalali-AI/BlackHole

Provides sparsified MOF similarity graphs used to test different graph thresholds.

---

# Notes

Large external datasets are excluded from version control to keep the repository lightweight.  
Ensure that all required datasets are downloaded and placed in the correct directories before running the pipeline.