"""
evaluation/evaluation_dashboard.py

Interactive Streamlit dashboard for thesis evaluation visualization.

Displays:
  - Experiment selection and metrics summary
  - Comparison plots (BLEU, FID, CLIPScore, etc.)
  - Qualitative side-by-side examples
  - Human evaluation summary (if available)

Run:
  streamlit run evaluation/evaluation_dashboard.py
"""

import os
import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

# File paths
RESULTS_CSV = "experiments/results/metrics_logs.csv"
QUAL_DIR = "evaluation/qualitative"
HUMAN_EVAL_SUMMARY = "evaluation/human_study/human_eval_summary.csv"
EVAL_PLOTS_DIR = "experiments/results/evaluation_plots"
METRICS_SUMMARY = "evaluation/reports/metrics_summary.csv"

st.set_page_config(
    page_title="Multimodal Evaluation Dashboard",
    layout="wide",
    page_icon="📊",
)

# ------------------------------------------------------------
# 🧭 Sidebar Configuration
# ------------------------------------------------------------
st.sidebar.title("⚙️ Evaluation Controls")
st.sidebar.markdown("Select experiments and view results interactively.")

# Load experiment options dynamically
if os.path.exists(RESULTS_CSV):
    df_all = pd.read_csv(RESULTS_CSV)
    experiment_names = sorted(df_all["experiment_name"].unique())
else:
    df_all = pd.DataFrame()
    experiment_names = []

selected_exp = st.sidebar.selectbox("Select Experiment", experiment_names)
show_metrics = st.sidebar.checkbox("Show Metric Trends", True)
show_qual = st.sidebar.checkbox("Show Qualitative Examples", True)
show_human = st.sidebar.checkbox("Show Human Evaluation Summary", True)

st.sidebar.markdown("---")
st.sidebar.info("💡 Tip: Combine this dashboard with Streamlit sharing or GitHub Pages for interactive thesis presentation.")

# ------------------------------------------------------------
# 🧮 Quantitative Metrics
# ------------------------------------------------------------
st.title("📈 Thesis Evaluation Dashboard")
st.markdown("""
This dashboard visualizes the **quantitative and qualitative performance** of your thesis system:
> *Integrating Diffusion Models with Retrieval-Augmented Generation and Multi-Constraint Prompting for Multimodal Content Generation.*
""")

if show_metrics:
    st.subheader("📊 Quantitative Metric Trends")

    if not df_all.empty:
        df_exp = df_all[df_all["experiment_name"] == selected_exp]
        if df_exp.empty:
            st.warning("No metrics found for this experiment.")
        else:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(df_exp["metric"], df_exp["delta_percent"], color="skyblue")
            ax.set_xlabel("Δ Percent Improvement")
            ax.set_title(f"Metric Improvements — {selected_exp}")
            st.pyplot(fig)

            st.dataframe(df_exp[["metric", "baseline_value", "improved_value", "delta_percent"]])
    else:
        st.warning("Metrics file not found. Run experiments first.")

# ------------------------------------------------------------
# 🎨 Qualitative Examples
# ------------------------------------------------------------
if show_qual:
    st.subheader("🎨 Qualitative Comparisons")
    qual_images = [f for f in os.listdir(QUAL_DIR) if selected_exp in f] if os.path.exists(QUAL_DIR) else []

    if not qual_images:
        st.warning("No qualitative images found. Run qualitative_examples.py first.")
    else:
        cols = st.columns(2)
        for i, img_file in enumerate(qual_images[:4]):
            img_path = os.path.join(QUAL_DIR, img_file)
            with cols[i % 2]:
                st.image(Image.open(img_path), caption=f"{selected_exp} — Example {i+1}", use_column_width=True)

# ------------------------------------------------------------
# 🧑‍🔬 Human Evaluation Results
# ------------------------------------------------------------
if show_human and os.path.exists(HUMAN_EVAL_SUMMARY):
    st.subheader("🧠 Human Evaluation Summary")

    df_human = pd.read_csv(HUMAN_EVAL_SUMMARY)
    if selected_exp in df_human["experiment"].values:
        row = df_human[df_human["experiment"] == selected_exp].iloc[0]
        st.markdown(f"### {selected_exp}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Factual Accuracy", f"{row['rating_factual']}/5")
        col2.metric("Visual Realism", f"{row['rating_realism']}/5")
        col3.metric("Style Alignment", f"{row['rating_style']}/5")
        col4.metric("Overall Quality", f"{row['rating_overall']}/5")
    else:
        st.warning("No human evaluation data found for this experiment.")
else:
    if show_human:
        st.info("Run human_eval.py to prepare and aggregate human ratings.")

# ------------------------------------------------------------
# 📊 Evaluation Plot Viewer
# ------------------------------------------------------------
st.markdown("---")
st.subheader("📈 Global Evaluation Plots")

if os.path.exists(EVAL_PLOTS_DIR):
    plot_files = [f for f in os.listdir(EVAL_PLOTS_DIR) if f.endswith(".png")]
    cols = st.columns(2)
    for i, pf in enumerate(plot_files[:4]):
        path = os.path.join(EVAL_PLOTS_DIR, pf)
        with cols[i % 2]:
            st.image(Image.open(path), caption=pf.replace("_", " ").replace(".png", ""), use_column_width=True)
else:
    st.info("No plots found. Run `plot_metrics.py` to generate visual results.")

# ------------------------------------------------------------
# 📜 Metrics Summary Table
# ------------------------------------------------------------
st.markdown("---")
st.subheader("📋 Aggregated Metrics Summary")

if os.path.exists(METRICS_SUMMARY):
    st.dataframe(pd.read_csv(METRICS_SUMMARY))
else:
    st.info("Run `metrics_reporter.py` to generate the metrics summary.")

# ------------------------------------------------------------
# 🏁 Footer
# ------------------------------------------------------------
st.markdown("---")
st.markdown("""
**Developed for Thesis Project (2025)**  
*Integrating Diffusion Models with RAG and Multi-Constraint Prompting for Multimodal Content Generation*  
Author: [Noorin Khot]
""")