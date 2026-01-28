from __future__ import annotations

import streamlit as st
from streamlit_app.utils import metric_row


def page_intro() -> None:
    st.markdown(
        """
        ### What this dashboard is
        This is a **screening pipeline** for CO₂-adsorbing MOFs: it learns from a *small labeled set* and helps **prioritize candidates** for deeper evaluation (simulation/experiments).

        ### Why this matters (decision perspective)
        In real materials discovery, the bottleneck is not “running one model” — it’s **choosing where to spend expensive lab/simulation budget**.
        If we can rank MOFs so that the *top-K* are more promising than random selection, we save time, cost, and iteration cycles.

        ### What you will see
        1) How limited labels create a **few-shot** setting  
        2) How graph thresholds change **coverage vs leakage risk**  
        3) How baselines vs graph-aware models compare  
        4) How this becomes a **shortlist of MOFs** for next-step evaluation
        """
    )

    metric_row(
        [
            ("Total MOFs", "14,296", "Full MOF descriptor table size"),
            ("MOFs with CO₂ labels", "363", "Matched labels at 298K, 1 bar"),
            ("Target property", "CO₂ uptake @ 298K, 1 bar", "Prediction target used throughout"),
        ]
    )

    st.markdown(
        """
        ### Outcome (what “success” looks like)
        - Not perfect prediction.
        - A **reliable prioritization signal**: shortlist candidates that are worth deeper evaluation.
        """
    )
