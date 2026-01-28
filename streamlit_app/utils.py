from __future__ import annotations

import streamlit as st
from pathlib import Path
from typing import Optional

from streamlit_app.pathing import resolve_first_existing


def inject_css() -> None:
    st.markdown(
        """
        <style>
          /* Make the app feel more "dashboard-like" */
          .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1100px; }
          h1 { font-size: 2.2rem !important; letter-spacing: -0.5px; }
          h2 { font-size: 1.5rem !important; margin-top: 1.2rem; }
          h3 { font-size: 1.15rem !important; margin-top: 1rem; }
          p, li { font-size: 0.98rem !important; line-height: 1.5; }

          /* Subtle card styling */
          .mof-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.07);
            padding: 14px 14px;
            border-radius: 14px;
          }

          /* Smaller captions */
          .caption { opacity: 0.8; font-size: 0.9rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_row(items: list[tuple[str, str, str]]) -> None:
    """
    items: [(label, value, help_text), ...]
    """
    cols = st.columns(len(items))
    for c, (label, value, help_text) in zip(cols, items):
        with c:
            st.metric(label=label, value=value, help=help_text)


def metric_card(title: str, value: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="mof-card">
          <div style="font-size:0.9rem; opacity:0.75;">{title}</div>
          <div style="font-size:1.7rem; font-weight:700; margin-top:6px;">{value}</div>
          <div style="font-size:0.9rem; opacity:0.75; margin-top:6px;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_figure(
    title: str,
    candidate_rel_paths: list[str],
    business_one_liner: str,
    width: int = 720,
) -> None:
    """
    Shows a figure if found; otherwise shows a clean 'not found' box listing searched paths.
    """
    st.subheader(title)

    fig_path: Optional[Path] = resolve_first_existing(candidate_rel_paths)

    if fig_path is None:
        st.info(
            "Figure not found. I looked for:\n\n"
            + "\n".join([f"- `{p}`" for p in candidate_rel_paths])
        )
        return

    st.image(str(fig_path), width=width)

    st.markdown(f"<div class='caption'>Business view: {business_one_liner}</div>", unsafe_allow_html=True)
