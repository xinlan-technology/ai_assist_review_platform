import pandas as pd
import streamlit as st

from core import auth, csv_io, ui
from features.screening import state

ui.inject_base_css()
auth.sidebar_user()
ui.app_header("Review Summary", "Progress and results of the human review.")
if state.active_name():
    st.caption(f"Project: **{state.active_name()}**")

with st.sidebar:
    st.subheader("View")
    st.selectbox(
        "Relevance threshold (≥ counts as relevant)",
        state.SCORE_OPTIONS,
        index=state.SCORE_OPTIONS.index(state.DEFAULT_THRESHOLD),
        key="threshold",
    )

if not state.has_results():
    st.info("No results yet. Open **Relevance Screening**, score a CSV, then review.")
    st.stop()

thr = state.threshold()
s = state.summary(thr)

with st.container(border=True):
    st.subheader("Progress")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", s["total"])
    c2.metric("Reviewed", s["reviewed"])
    c3.metric("Remaining", s["remaining"])
    rate = s["agreement_rate"]
    c4.metric("Agreement rate", f"{rate * 100:.0f}%" if rate is not None else "—")
    st.progress(s["reviewed"] / s["total"] if s["total"] else 0.0)
    st.caption(f"Agreed with AI: {s['agreed']}  ·  Disagreed: {s['disagreed']}")

with st.container(border=True):
    st.subheader("Final decisions")
    counts = s["counts"]
    chart_df = pd.DataFrame(
        {"Papers": [counts[lbl] for lbl in state.DECISION_LABELS]},
        index=state.DECISION_LABELS,
    )
    st.bar_chart(chart_df, color="#0E6E55")

with st.container(border=True):
    st.subheader("Results")
    results = state.results_dataframe(thr)
    st.dataframe(results, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇ Download results CSV",
        data=csv_io.to_csv_bytes(results),
        file_name="screening_results.csv",
        mime="text/csv",
    )
