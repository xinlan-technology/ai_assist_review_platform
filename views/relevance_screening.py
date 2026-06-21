import html

import pandas as pd
import streamlit as st

from core import auth, csv_io, ui
from core.llm import PROVIDERS, PROVIDER_KEY_HELP
from features.screening import scorer, state
from features.screening.rubric import DEFAULT_RUBRIC, DEFAULT_TOPIC

NONE_OPTION = "(none)"
RUBRIC_SCORE = "Score"
RUBRIC_DEF = "Definition"

ui.inject_base_css()
ui.app_header(
    "Relevance Screening",
    "Upload a CSV, score papers against your rubric, then review them one by one.",
)

auth.sidebar_user()

if not state.active_id():
    st.info("Open or create a project on the **My Projects** page first.")
    st.stop()
st.caption(f"Project: **{state.active_name()}**")


def _doi_url(doi: str) -> str:
    doi = doi.strip()
    if doi.startswith("http://") or doi.startswith("https://"):
        return doi
    return f"https://doi.org/{doi}"


def _nearest_ten(value, default: int = 50) -> int:
    try:
        v = max(10, min(100, int(round(float(value) / 10.0) * 10)))
        return v
    except (TypeError, ValueError):
        return default


def _clean(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


with st.sidebar:
    st.subheader("Configuration")
    provider = st.selectbox("Provider", list(PROVIDERS.keys()))
    model = st.selectbox("Model", PROVIDERS[provider])
    api_key = st.text_input(
        "API key", type="password", help=PROVIDER_KEY_HELP.get(provider, "")
    )
    st.caption(
        "Your key is kept only in this app session, never saved to disk or the "
        "database, and sent only to the selected model provider."
    )
    st.selectbox(
        "Relevance threshold (≥ counts as relevant)",
        state.SCORE_OPTIONS,
        index=state.SCORE_OPTIONS.index(state.DEFAULT_THRESHOLD),
        key="threshold",
    )
    st.divider()
    if st.button("💾 Save project", use_container_width=True):
        if state.save_active():
            st.toast("Saved.")

with st.container(border=True):
    st.subheader("Scoring rubric")
    st.caption("Define what each score means; the AI scores against this rubric.")

    topic = st.text_area(
        "Research project / topic",
        value=st.session_state.get("topic", DEFAULT_TOPIC),
        height=80,
    )
    st.session_state["topic"] = topic

    if "rubric_df" not in st.session_state:
        levels = list(range(100, 0, -10))
        st.session_state["rubric_df"] = pd.DataFrame(
            {RUBRIC_SCORE: levels, RUBRIC_DEF: [DEFAULT_RUBRIC.get(s, "") for s in levels]}
        )

    rubric_df = st.data_editor(
        st.session_state["rubric_df"],
        hide_index=True,
        use_container_width=True,
        column_config={
            RUBRIC_SCORE: st.column_config.NumberColumn(RUBRIC_SCORE, disabled=True, width="small"),
            RUBRIC_DEF: st.column_config.TextColumn(RUBRIC_DEF, width="large"),
        },
        key="rubric_editor",
    )
    st.session_state["rubric_df"] = rubric_df
    rubric = {int(row[RUBRIC_SCORE]): str(row[RUBRIC_DEF]) for _, row in rubric_df.iterrows()}

with st.container(border=True):
    st.subheader("Upload data")
    st.caption(
        "A UTF-8 CSV with one paper per row. Below you map a **DOI** (optional), a "
        "**title**, and an **abstract** column — names can be anything. Any other "
        "columns are kept and included in the downloaded results."
    )

    uploaded = st.file_uploader(
        "CSV file", type=["csv"], help="Example columns: DOI, Title, Abstract"
    )

    if uploaded is not None:
        try:
            df = csv_io.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read the file: {exc}")
            df = None

        if df is not None and len(df.columns) > 0:
            st.write(f"{len(df)} rows. Preview of the first 5:")
            st.dataframe(df.head(5), use_container_width=True)

            cols = list(df.columns)
            doi_guess = csv_io.guess_column(cols, csv_io.DOI_CANDIDATES)
            doi_choices = [NONE_OPTION] + cols
            doi_default = doi_guess if (doi_guess and doi_guess.lower() in
                                        [c.lower() for c in csv_io.DOI_CANDIDATES]) else NONE_OPTION

            c1, c2, c3 = st.columns(3)
            doi_col = c1.selectbox("DOI column (optional)", doi_choices,
                                   index=doi_choices.index(doi_default))
            title_col = c2.selectbox(
                "Title column", cols, index=cols.index(csv_io.guess_column(cols, csv_io.TITLE_CANDIDATES))
            )
            abstract_col = c3.selectbox(
                "Abstract column", cols,
                index=cols.index(csv_io.guess_column(cols, csv_io.ABSTRACT_CANDIDATES)),
            )

            st.caption(
                f"Each row is scored by {provider} · {model} "
                f"(~{len(df)} API calls), billed to your own key."
            )

            if st.button("▶ Score papers", type="primary", disabled=not api_key):
                rows = []
                for _, row in df.iterrows():
                    rows.append(
                        {
                            "doi": "" if doi_col == NONE_OPTION else _clean(row.get(doi_col)),
                            "title": _clean(row.get(title_col)),
                            "abstract": _clean(row.get(abstract_col)),
                        }
                    )

                progress = st.progress(0.0, text="Starting…")
                total = len(rows)
                for i, paper in enumerate(rows):
                    try:
                        scored = scorer.score_paper(
                            provider, model, api_key, topic, rubric,
                            paper["title"], paper["abstract"],
                        )
                        paper["ai_score"], paper["ai_reason"] = scored["score"], scored["reasoning"]
                    except Exception as exc:
                        paper["ai_score"], paper["ai_reason"] = None, f"⚠️ Call failed: {exc}"
                    paper["decision"], paper["human_score"] = None, None
                    progress.progress((i + 1) / total, text=f"Scoring… {i + 1}/{total}")
                progress.empty()

                state.init(rows, df.reset_index(drop=True))
                if state.save_active():
                    st.success("Done. Review the papers below.")
                else:
                    st.warning("Scoring finished in this session, but the project was not saved.")

if state.has_results():
    with st.container(border=True):
        st.subheader("Review")
        thr = state.threshold()
        ps = state.papers()
        total = len(ps)
        i = state.current_index()
        paper = ps[i]
        s = state.summary(thr)

        st.progress(
            s["reviewed"] / total if total else 0.0,
            text=f"Reviewed {s['reviewed']} / {total}",
        )

        nav_prev, nav_mid, nav_next = st.columns([1, 2, 1])
        if nav_prev.button("‹ Prev", disabled=i == 0, use_container_width=True):
            state.goto(i - 1)
            st.rerun()
        nav_mid.markdown(
            f"<div style='text-align:center;color:#6A7280;font-size:13px;padding-top:6px;'>"
            f"Paper {i + 1} / {total}</div>",
            unsafe_allow_html=True,
        )
        if nav_next.button("Next ›", disabled=i >= total - 1, use_container_width=True):
            state.goto(i + 1)
            st.rerun()

        st.markdown(f"#### {paper['title'] or '(no title)'}")
        if paper.get("doi"):
            st.markdown(f"[{paper['doi']}]({_doi_url(paper['doi'])})")
        st.write(paper["abstract"] or "_(no abstract)_")

        if paper.get("ai_score") is None:
            badge = "AI scoring failed"
            sug_line = ""
        else:
            badge = f"AI score: {paper['ai_score']} / 100"
            sug_line = (
                '<span style="font-size:12px;color:#0E6E55;margin-left:8px;">'
                f"→ suggests {state.ai_suggestion(paper, thr)} "
                f"(your threshold: {thr})</span>"
            )
        st.markdown(
            f"""<div style="background:#F6F8F7;border:1px solid #E2EBE7;border-radius:8px;
                        padding:10px 12px;margin:6px 0 10px;">
              <span style="background:#0E6E55;color:#fff;font-size:12px;font-weight:600;
                           padding:3px 10px;border-radius:6px;">{badge}</span>
              {sug_line}
              <div style="font-size:12px;color:#5A626B;font-style:italic;margin-top:5px;">
                {html.escape(paper.get("ai_reason") or "")}</div>
            </div>""",
            unsafe_allow_html=True,
        )

        decision = paper.get("decision")
        st.caption("Do you agree with the AI?")
        agree_col, disagree_col = st.columns(2)
        if agree_col.button(
            "✓ Agree", type="primary" if decision == "agree" else "secondary",
            use_container_width=True,
        ):
            state.record(i, "agree")
            state.save_active()
            state.goto(i + 1)
            st.rerun()
        if disagree_col.button(
            "✗ Disagree", type="primary" if decision == "disagree" else "secondary",
            use_container_width=True,
        ):
            state.record(i, "disagree")
            state.save_active()
            st.rerun()

        if decision == "disagree":
            default_score = _nearest_ten(paper.get("human_score") or paper.get("ai_score"))
            human_score = st.selectbox(
                "Your score",
                state.SCORE_OPTIONS,
                index=state.SCORE_OPTIONS.index(default_score),
                key=f"hs_{state.active_id()}_{i}",
            )
            state.set_human_score(i, human_score)
            state.save_active()
            st.caption(f"→ Final: **{state.label(human_score, thr)}**")

        st.divider()
        st.caption(
            f"Your decision for this paper: **{state.final_decision(paper, thr)}**"
            + ("  ·  not reviewed yet" if decision is None else "")
        )
        st.caption("Open the **Review Summary** page for overall progress and to download results.")
