from __future__ import annotations

import streamlit as st
from streamlit_app.utils import metric_row, show_figure


def page_fewshot() -> None:
    st.header("Few-Shot Learning Setup")

    st.markdown(
        "Only **363 MOFs** have CO₂ labels after strict ID matching. We keep a fixed split for fair comparisons."
    )

    metric_row(
        [
            ("Train labeled", "254", "Few-shot supervision for training"),
            ("Val labeled", "54", "Model selection"),
            ("Test labeled", "55", "Final evaluation"),
        ]
    )

    st.markdown("### Data reality check (distributions)")

    show_figure(
        title="Label scarcity in the full MOF set",
        candidate_rel_paths=[
            "outputs/figures/02_splits/split_sizes_v1.png",
        ],
        business_one_liner="Shows how small the labeled set is vs total inventory — highlights why this is a few-shot problem.",
        width=680,
    )

    show_figure(
        title="CO₂ uptake distribution across splits (histogram)",
        candidate_rel_paths=[
            "outputs/figures/02_splits/uptake_by_split_hist_v1.png",
        ],
        business_one_liner="Checks whether train/val/test look comparable — reduces risk of biased evaluation.",
        width=680,
    )

    show_figure(
        title="CO₂ uptake spread across splits (violin)",
        candidate_rel_paths=[
            "outputs/figures/02_splits/uptake_by_split_violin_v1.png",
        ],
        business_one_liner="Confirms the split captures similar ranges — ensures conclusions generalize beyond the training sample.",
        width=680,
    )
