from __future__ import annotations

import streamlit as st

from streamlit_app.sections.problem import page_problem
from streamlit_app.sections.fewshot import page_fewshot
from streamlit_app.sections.graphs import page_graphs
from streamlit_app.sections.results import page_results


APP_TITLE = "Learning to Predict Material Properties with Limited Data — Few-Shot Learning"
APP_SUBTITLE = "MOF CO₂ Screening Dashboard (decision-support view: prioritize candidates for deeper evaluation)"


def inject_css() -> None:
    st.markdown(
        """
        <style>
            /* Layout width */
            .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1150px; }

            /* Typography */
            h1, h2, h3 { letter-spacing: -0.02em; }
            h1 { font-size: 2.2rem; margin-bottom: 0.25rem; }
            h2 { font-size: 1.55rem; margin-top: 1.2rem; }
            h3 { font-size: 1.2rem; margin-top: 1.0rem; }

            /* Subtitle style */
            .subtitle {
                color: rgba(255,255,255,0.75);
                font-size: 0.98rem;
                margin-top: -0.25rem;
                margin-bottom: 1.25rem;
            }

            /* Executive callout card */
            .exec-card {
                border: 1px solid rgba(255,255,255,0.08);
                background: rgba(255,255,255,0.03);
                border-radius: 14px;
                padding: 16px 18px;
                margin: 10px 0 14px 0;
            }
            .exec-card b { color: rgba(255,255,255,0.92); }

            /* Smaller images feel less noisy */
            img { border-radius: 10px; }

            /* Sidebar */
            section[data-testid="stSidebar"] {
                border-right: 1px solid rgba(255,255,255,0.06);
            }
            .sidebar-note {
                color: rgba(255,255,255,0.75);
                font-size: 0.90rem;
                line-height: 1.35rem;
            }
            .tiny {
                color: rgba(255,255,255,0.62);
                font-size: 0.85rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def header() -> None:
    st.title(APP_TITLE)
    st.markdown(f"<div class='subtitle'>{APP_SUBTITLE}</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="exec-card">
            <b>What you get here:</b> a clear story from problem → few-shot setup → graph trade-offs → model comparisons → <b>Top-K MOF shortlist</b>.
            <br/>
            <span class="tiny">Interpretation focus: screening value (prioritization), not “perfect prediction”.</span>
        </div>

        <div class="exec-card">
            <b>Key Insight:</b> This view summarizes the model’s top 5 candidate MOFs, combining high predicted CO₂ uptake with low uncertainty (high confidence). We provide both quantitative scores (with error bars) and qualitative reasoning (e.g., pore size insights) for each material. This bridges the gap between model output and scientific interpretability — enabling better shortlisting for downstream lab or simulation validation.
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_nav() -> str:
    with st.sidebar:
        st.markdown("### Navigation")
        page = st.radio(
            "Go to",
            [
                "1) Problem & Motivation",
                "2) Few-Shot Setup",
                "3) Graph Construction & Thresholds",
                "4) Model Results & Shortlist",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown(
            """
            <div class="sidebar-note">
            <b>How to read this dashboard</b><br/>
            • <b>Business lens:</b> screening funnel → rank candidates → spend lab/simulation budget smarter.<br/>
            • <b>Scientific lens:</b> compare baselines vs graph-aware learning; validate structure usefulness using shuffled controls.<br/>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown(
            """
            <div class="tiny">
            <b>Target:</b> CO₂ uptake @ 298 K, 1 bar<br/>
            <b>Data reality:</b> 14,296 MOFs, only 363 labeled<br/>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return page


def route(page: str) -> None:
    if page.startswith("1"):
        page_problem()
    elif page.startswith("2"):
        page_fewshot()
    elif page.startswith("3"):
        page_graphs()
    else:
        page_results()


def main() -> None:
    st.set_page_config(
        page_title="MOF CO₂ Screening — Few-Shot Learning",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    header()
    page = sidebar_nav()
    route(page)


if __name__ == "__main__":
    main()
