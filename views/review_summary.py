import pandas as pd
import streamlit as st

from core import auth, csv_io, db, ui
from features.extraction import reports, schema
from features.workflow import state

ui.inject_base_css()
auth.sidebar_user()
ui.app_header("Review Summary", "Progress and results across the stages of your workflow.")
state.require_saved_results()

if not state.active_id():
    st.info("Open or create a project on the **My Projects** page first.")
    st.stop()
st.caption(
    f"Project: **{state.active_name()}**  ·  Workflow: **{state.MODE_LABELS[state.mode()]}**"
)

if not state.has_papers():
    st.info("No papers yet. Load a CSV (PRISMA mode) or upload PDFs (direct mode) first.")
    st.stop()


def _screening_block(stage: str, advance_note: str | None) -> None:
    s = state.stage_summary(stage)
    with st.container(border=True):
        st.subheader(state.STAGE_TITLES[stage])
        if s["total"] == 0:
            st.caption("No papers have reached this stage yet.")
            return
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Papers", s["total"])
        c2.metric("AI screened", s["ai_done"])
        c3.metric("Reviewed", s["reviewed"])
        rate = s["agreement_rate"]
        c4.metric(
            "Agreement rate",
            f"{rate * 100:.0f}%" if rate is not None else "—",
            help="Agreed / (agreed + disagreed). Only papers where both the AI "
                 "and you gave a verdict count; human-only decisions are excluded.",
        )
        st.progress(s["reviewed"] / s["total"] if s["total"] else 0.0)
        if s["ai_failed"]:
            st.caption(f"⚠️ {s['ai_failed']} AI call(s) failed — rerun screening to retry.")
        if s["ai_invalid"]:
            st.caption(f"⚠️ {s['ai_invalid']} invalid AI response(s) — decide those "
                       "papers yourself (or retry them) on the screening page.")
        if s["stale"]:
            st.caption(f"⚠️ {s['stale']} AI verdict(s) were produced under earlier "
                       "criteria — use “Re-run outdated papers” on the screening page.")
        if s["human_only"]:
            st.caption(f"{s['human_only']} paper(s) were decided without a usable "
                       "AI verdict (excluded from the agreement rate).")

        counts = s["counts"]
        shown = [lbl for lbl in state.DISPLAY_LABELS if counts.get(lbl)]
        if shown:
            chart_df = pd.DataFrame({"Papers": [counts[lbl] for lbl in shown]}, index=shown)
            st.bar_chart(chart_df, color="#0E6E55")
        if advance_note:
            st.caption(advance_note)


enabled = state.stages()

if state.STAGE_ABSTRACT in enabled:
    n_advance = len(state.stage_papers(state.STAGE_FULLTEXT))
    _screening_block(
        state.STAGE_ABSTRACT,
        f"→ {n_advance} paper(s) (Include + Unsure) advance to full-text screening.",
    )

fulltext_summary = state.stage_summary(state.STAGE_FULLTEXT)
included = fulltext_summary["counts"].get(state.VERDICT_LABELS[state.VERDICT_INCLUDE], 0)
_screening_block(
    state.STAGE_FULLTEXT,
    f"→ {included} paper(s) included after full-text review can proceed to Data Extraction.",
)

with st.container(border=True):
    st.subheader("Results")
    results = state.results_dataframe()
    st.dataframe(results, width="stretch", hide_index=True)
    st.download_button(
        "⬇ Download results CSV",
        data=csv_io.to_csv_bytes(results),
        file_name="review_results.csv",
        mime="text/csv",
    )

with st.container(border=True):
    st.subheader("Data Extraction")
    if not state.config().get("extraction_questions"):
        st.info("Define your extraction questions on the Data Extraction page to begin.")
    else:
        try:
            spec = schema.build_spec(state.config().get("extraction_instructions", ""),
                                     state.config()["extraction_questions"])
            metadata = db.load_fulltexts(auth.current_user(), state.active_id())
        except (ValueError, db.DatabaseError) as exc:
            st.error(str(exc))
        else:
            eligible = [p for _, p in state.stage_papers(state.STAGE_EXTRACTION)]
            counts = reports.summary(eligible, spec, metadata)
            columns = st.columns(4)
            labels = {"Eligible": "total", "AI extracted": "ai_done",
                      "Human confirmed": "confirmed", "Invalid fields": "invalid_fields"}
            for column, (label, key) in zip(columns, labels.items(), strict=True):
                column.metric(label, counts[key])
            st.progress(counts["confirmed"] / counts["total"] if counts["total"] else 0.0)
            if counts["outdated"] or counts["failed"]:
                st.caption(f"{counts['outdated']} outdated · {counts['failed']} unresolved call or response error(s).")
            final = reports.final_dataframe(eligible, spec, metadata)
            st.caption("Only current, human-confirmed answers are included in the final answer columns.")
            st.dataframe(final, width="stretch", hide_index=True)
            st.download_button("Download extraction results CSV", csv_io.to_csv_bytes(final),
                               file_name="extraction_results.csv", mime="text/csv")
            audit = reports.audit_dataframe(state.papers(), {p["uid"] for p in eligible}, spec, metadata)
            st.download_button("Download extraction audit CSV", csv_io.to_csv_bytes(audit),
                               file_name="extraction_audit.csv", mime="text/csv")
            with st.expander("Choice answer frequencies"):
                st.caption("Counts use confirmed papers only. A multiple-choice question can contribute to several options.")
                st.dataframe(reports.choice_counts(eligible, spec, metadata), width="stretch", hide_index=True)
