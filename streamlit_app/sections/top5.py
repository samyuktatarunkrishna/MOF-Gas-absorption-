import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def page_top5():
    st.header("Top 5 MOF Candidates — Confidence & Interpretation")

    st.markdown("""
    This section interprets the top 5 candidates from our CO₂ uptake predictions. Each candidate is analyzed based on its
    structural descriptors (e.g., Largest Cavity Diameter, Free Sphere) and model prediction confidence.
    """)

    # Load Top-K
    topk_path = "outputs/candidates/topk_candidates.csv"
    try:
        df = pd.read_csv(topk_path)
    except FileNotFoundError:
        st.error("topk_candidates.csv not found. Please run the ranking pipeline first.")
        return

    top5 = df.head(5).copy()
    top5.index = ["MOF A", "MOF B", "MOF C", "MOF D", "MOF E"]

    st.subheader("Summary Table")
    st.dataframe(top5[["refcode", "screen_score", "confidence", "Largest Cavity Diameter",
                       "Pore Limiting Diameter", "Largest Free Sphere", "rationale"]])

    st.subheader("Prediction Score ± Confidence")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(top5.index, top5["screen_score"], yerr=1.0 / top5["confidence"], capsize=8, color="#4C72B0")
    ax.set_ylabel("Predicted CO₂ Uptake Score")
    ax.set_title("Top 5 MOFs: Prediction and Uncertainty")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    st.pyplot(fig)

    st.subheader("Candidate Interpretations")
    for idx, row in top5.iterrows():
        st.markdown(f"**{idx} ({row['refcode']})**")
        st.markdown(f"• **Screen Score**: {row['screen_score']:.3f}, **Confidence**: {row['confidence']:.3f}")
        st.markdown(f"• **Cavity**: {row['Largest Cavity Diameter']} Å, **Limiting**: {row['Pore Limiting Diameter']} Å")
        st.markdown(f"• **Reasoning**: {'High cavity but low limiting suggests bottleneck' if row['Pore Limiting Diameter'] < 2 else 'Open porous structure likely supports high uptake'}")
        st.markdown(f"• **Model Basis**: {row['rationale']}")
        st.markdown("---")
