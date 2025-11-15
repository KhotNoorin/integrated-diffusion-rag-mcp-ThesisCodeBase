"""
frontend/components/evaluation_viewer.py

Small widget to surface evaluation metrics quickly in the frontend.
"""

from __future__ import annotations
import streamlit as st
import os
import pandas as pd

METRICS_CSV = "experiments/results/metrics_logs.csv"
METRICS_SUMMARY = "evaluation/reports/metrics_summary.csv"

def render_evaluation_viewer(short: bool = False):
    st.subheader("Evaluation Viewer")
    if not os.path.exists(METRICS_CSV):
        st.info("No metrics found. Run experiments to populate metrics.")
        return

    df = pd.read_csv(METRICS_CSV)
    latest_exps = df["experiment_name"].unique().tolist()[-5:]
    st.markdown("Showing recent experiments:")
    st.write(latest_exps)

    # Quick summary table for selected experiments (short)
    if short:
        recent = df[df["experiment_name"].isin(latest_exps)]
        pivot = recent.groupby(["experiment_name", "metric"])["delta_percent"].mean().unstack(fill_value=0)
        st.dataframe(pivot.round(2))
    else:
        if os.path.exists(METRICS_SUMMARY):
            st.markdown("Detailed metrics summary:")
            st.dataframe(pd.read_csv(METRICS_SUMMARY))
        else:
            st.dataframe(df)