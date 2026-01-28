from __future__ import annotations

import streamlit as st
from streamlit_app.utils import show_figure


def page_graphs() -> None:
    st.header("Graph Construction & Thresholds")

    st.markdown(
        """
        We import MOF similarity graphs from multiple runs. The **threshold** controls how strict we are about keeping edges:

        - **High threshold (0.90)** → small, clean graph; **less label coverage**
        - **Low threshold (0.10 / 0.00)** → bigger graph; more coverage but **higher train–test proximity risk**
        """
    )

    show_figure(
        "Label coverage vs threshold",
        ["outputs/figures/03_graph_threshold_diagnostics/fig1_label_coverage_vs_threshold.png"],
        "Tracks how many labeled MOFs are usable at each threshold (data availability vs strictness).",
        width=680,
    )

    show_figure(
        "Graph size vs threshold",
        ["outputs/figures/03_graph_threshold_diagnostics/fig2_graph_size_vs_threshold.png"],
        "Shows operational scale — lower thresholds increase graph size and computation cost.",
        width=680,
    )

    show_figure(
        "Test→Train exposure vs threshold",
        ["outputs/figures/03_graph_threshold_diagnostics/fig3_test_train_exposure_vs_threshold.png"],
        "Measures potential leakage risk: how often test nodes connect strongly to train nodes.",
        width=680,
    )

    show_figure(
        "Test→Train max edge weight vs threshold",
        ["outputs/figures/03_graph_threshold_diagnostics/fig4_test_train_weight_vs_threshold.png"],
        "Quantifies ‘closeness’ across splits; higher means the test set is less independent.",
        width=680,
    )
