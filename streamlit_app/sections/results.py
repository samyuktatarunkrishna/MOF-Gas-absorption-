from __future__ import annotations

import streamlit as st
import pandas as pd

from streamlit_app.utils import show_figure
from streamlit_app.pathing import PROJECT_ROOT


def page_results() -> None:
    st.header("Model Results & Comparisons")

    st.markdown(
        """
        **Interpretation focus: screening value.**  
        The goal isn’t perfect prediction — the goal is a **reliable prioritization signal** that helps choose which MOFs should be evaluated next.
        """
    )

    st.subheader("Baselines (descriptor-only)")
    show_figure(
        "Test RMSE: Best baseline vs BH-feature baseline",
        ["outputs/figures/04_baselines/test_rmse_baseline_vs_bh_feature_v1.png"],
        "Checks whether similarity-informed features add value over plain descriptors.",
        width=650,
    )

    show_figure(
        "Parity plot (baseline)",
        ["outputs/figures/04_baselines/parity_baseline_v1.png"],
        "Quick health-check: predictions should track true values without systematic bias.",
        width=650,
    )

    st.subheader("GNN vs shuffled graph (control)")
    show_figure(
        "RMSE (test): real graph vs shuffled graph",
        ["outputs/figures/05_gnn_vs_shuffled/real_vs_shuffled_rmse_test.png"],
        "Validates whether graph structure carries real signal (vs a randomized control).",
        width=650,
    )

    show_figure(
        "Comparison summary (real vs shuffled)",
        ["outputs/figures/05_gnn_vs_shuffled/test_rmse_comparison_v1.png"],
        "Decision view: consistency across runs matters more than one-off wins.",
        width=650,
    )

    st.divider()
    st.header("Top-K MOF Shortlist (Concrete Output)")

    st.markdown(
        """
        **This is the decision-support output.**  
        A deployed pipeline should not stop at plots — it should produce a **shortlist of MOFs** that are worth deeper evaluation.
        """
    )

    cand_csv = PROJECT_ROOT / "outputs" / "candidates" / "topk_candidates.csv"
    if not cand_csv.exists():
        st.warning("Candidate shortlist not found. Run: `python src/models/rank_candidates.py`")
        return

    df = pd.read_csv(cand_csv).copy()

    st.markdown("### Top candidates (ranked)")
    show_cols = [
        "refcode",
        "screen_score",
        "confidence",
        "pred_uptake_rf",
        "pred_uptake_rf_std",
        "graph_neighbor_mean",
        "graph_neighbor_std",
        "graph_neighbor_support",
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    st.dataframe(df[show_cols].head(50), use_container_width=True)

    st.markdown(
        """
        **How to read this table (business terms):**
        - **screen_score**: overall priority (higher = evaluate earlier)
        - **confidence**: how stable the recommendation is (higher = more reliable)
        - **neighbor_support**: how much labeled evidence is near this MOF in the similarity graph
        - **std values**: uncertainty / disagreement (lower is better)
        """
    )

    # Show the Top-20 figure if available
    show_figure(
        "Top-20 candidates (with uncertainty band)",
        ["outputs/figures/06_summary/topk_candidates_top20_v1.png"],
        "Turns model output into a shortlist — useful for allocating lab/simulation budget.",
        width=650,
    )
