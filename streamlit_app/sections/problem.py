from __future__ import annotations

import streamlit as st
import pandas as pd

from streamlit_app.pathing import PROJECT_ROOT
from streamlit_app.utils import metric_row, show_figure


def page_problem() -> None:
    st.header("Problem & Motivation")

    # --- Executive summary block ---
    st.markdown(
        """
        ### Executive summary (what this dashboard is for)

        **CO₂ capture and separation depend on material discovery — and material discovery has a scale problem.**
        Thousands of MOF structures exist, but **high-quality CO₂ adsorption measurements are scarce and expensive** to obtain
        (lab experiments and high-fidelity simulations both cost time and budget).

        This project frames the task as **few-shot learning**:
        we learn from a **small labeled set** and use structure + similarity information to produce **screening signals**.

        **Goal (decision-support):**
        not “perfect prediction”, but a **reliable way to prioritize which MOFs are worth deeper evaluation next**.
        """
    )

    st.divider()

    # --- Key metrics ---
    total_mofs = 14296
    labeled_mofs = 363
    target = "CO₂ uptake @ 298 K, 1 bar"

    metric_row(
        [
            ("Total MOFs", f"{total_mofs:,}", "Full candidate space available for screening."),
            ("MOFs with CO₂ labels", f"{labeled_mofs:,}", "Only a small subset has usable measurements."),
            ("Target property", target, "Standardized condition for fair comparison."),
        ]
    )

    st.markdown(
        """
        **Business framing:** treat the workflow as a *screening funnel*.
        The model helps allocate expensive lab/simulation resources by ranking candidates early.
        """
    )

    st.divider()

    # --- What is being tackled ---
    st.subheader("What I am tackling (in practical terms)")
    st.markdown(
        """
        **The core bottleneck:** we have **lots of structures** but **very few labels**.

        That creates three real-world problems:
        1. **Data scarcity:** classical ML models are limited because they need many labeled examples.
        2. **Bias risk:** evaluation can look “good” if train/test are too similar (especially on graphs).
        3. **Actionability gap:** even if metrics improve slightly, the key question is:
           **Which MOFs should we test next?**
        """
    )

    st.divider()

    # --- What the pipeline does ---
    st.subheader("What the pipeline does (end-to-end)")
    st.markdown(
        """
        This workflow combines:
        - **Descriptors** (pore/cavity features) → fast, interpretable signals  
        - **Sparse adsorption labels** (from curated isotherms) → supervision under scarcity  
        - **MOF similarity graphs** (MOFGalaxyNet/BlackHole style) → share information across related MOFs  

        The pipeline produces two types of outcomes:
        - **Scientific evaluation:** baselines vs graph-based learning + shuffled-graph controls
        - **Decision output:** a **Top-K shortlist of MOF refcodes** with a **confidence score**
        """
    )

    st.divider()

    # --- Highlight the new “ranking use case” explicitly ---
    st.subheader("Main deliverable: Top-K MOF shortlist (decision support)")
    st.markdown(
        """
        The most useful deployed output is not a plot — it’s a **ranked candidate list**.

        **What the shortlist gives:**
        - a **ranked list of refcodes** (Top-K)
        - a **screening score** (higher = evaluate earlier)
        - a **confidence score** (higher = more reliable recommendation)
        - a simple uncertainty proxy (variation across runs / neighbor agreement)

        **Why this matters:** it directly supports a real screening workflow:
        *“Given limited budget, which MOFs should we evaluate next?”*
        """
    )

    cand_csv = PROJECT_ROOT / "outputs" / "candidates" / "topk_candidates.csv"
    if cand_csv.exists():
        df = pd.read_csv(cand_csv).copy()

        # show a small teaser table (top 10) on the overview page
        cols = ["refcode", "screen_score", "confidence"]
        cols = [c for c in cols if c in df.columns]
        st.markdown("**Preview (Top-10 candidates):**")
        st.dataframe(df[cols].head(10), use_container_width=True)
    else:
        st.info("Shortlist not generated yet. Run: `python src/models/rank_candidates.py`")

    st.divider()

    # --- Quick “results in plain language” ---
    st.subheader("Results (interpreted in plain language)")
    st.markdown(
        """
        The evaluation focuses on **screening usefulness**:

        - **Baselines** show what you can do from descriptors alone (fast but limited).
        - **Graph thresholds** show a trade-off: higher thresholds → cleaner graph but fewer labeled nodes;
          lower thresholds → more coverage but higher risk of train–test proximity.
        - **Shuffled-graph control** checks whether graph structure carries meaningful signal
          (real graph should beat shuffled if structure helps).

        **Key idea:** we don’t chase perfection — we look for signals that are stable and support prioritization.
        """
    )

    # Optional: small “hero” figure (keep it minimal)
    show_figure(
        "Label coverage vs threshold",
        ["outputs/figures/03_graph_threshold_diagnostics/fig1_label_coverage_vs_threshold.png"],
        "Executive view: as the graph becomes less strict, more labeled MOFs become usable for learning.",
        width=650,
    )

    st.divider()

    # --- “How to use this dashboard” block ---
    st.subheader("How to use this dashboard")
    st.markdown(
        """
        - **Few-Shot Setup:** see how scarce labels are and how the split is defined  
        - **Graph Construction & Thresholds:** see trade-offs in coverage vs leakage risk  
        - **Model Results & Comparisons:** see baselines, controls, and the shortlist output  
        - **Top-K shortlist:** pick candidates for deeper evaluation (simulation/experiment)
        """
    )
