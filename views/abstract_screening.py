import contextlib
import html
import json

import pandas as pd
import streamlit as st

from core import auth, csv_io, db, ui
from features.screening import controls, prompts
from features.workflow import documents, run_controls, runs, state

NONE_OPTION = "(none)"
STAGE = state.STAGE_ABSTRACT

BADGE_COLORS = {
    state.VERDICT_INCLUDE: "#0E6E55",
    state.VERDICT_EXCLUDE: "#B3261E",
    state.VERDICT_UNSURE: "#B26A00",
}

ui.inject_base_css()
ui.app_header(
    "Abstract Screening",
    "Define eligibility criteria, screen titles & abstracts with AI, then review each verdict.",
)

auth.sidebar_user()

if not state.active_id():
    st.info("Open or create a project on the **My Projects** page first.")
    st.stop()
state.require_saved_results()
if STAGE not in state.stages():
    st.info("This project uses the full-text direct workflow — abstract screening is skipped.")
    st.stop()
st.caption(f"Project: **{ui.escape_markdown(state.active_name())}**")

if state.pending_pdf_deletions():
    st.warning("Removed PDF files still await cleanup. The saved paper list is unaffected.")
    if st.button("Retry deleting removed PDFs"):
        try:
            documents.cleanup(state.prepare_save())
        except db.DatabaseError as exc:
            st.error(str(exc))
        else:
            st.rerun()


def _clean(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


with st.sidebar:
    provider, model, api_key = ui.model_controls()
    models = ui.additional_models(provider, model, api_key)
    st.divider()
    if st.button("💾 Save project", width="stretch") and state.save_active():
        st.toast("Saved.")

with st.container(border=True):
    st.subheader("Abstract screening prompt")
    st.caption(
        "Write the screening instructions in your own words — this text is sent "
        "to the AI with each title and abstract. The AI answers Include, Exclude, or Unsure "
        "for each paper; Unsure papers advance to full-text screening."
    )
    criteria = st.text_area(
        "Abstract prompt",
        value=state.config().get("abstract_criteria", ""),
        placeholder=prompts.CRITERIA_PLACEHOLDER,
        height=220,
        label_visibility="collapsed",
        key=f"abstract_prompt_{state.active_id()}",
    )
    if criteria != state.config().get("abstract_criteria", ""):
        state.commit(lambda: state.config().update(abstract_criteria=criteria))

with st.container(border=True):
    st.subheader("Upload papers")
    st.caption(
        "A UTF-8 CSV with one paper per row. Map a **DOI** (optional), a **title**, "
        "and an **abstract** column — names can be anything. Other columns are kept "
        "and included in the exported results."
    )
    st.caption(
        f"Limits: {csv_io.MAX_CSV_BYTES // (1024 * 1024)} MB, "
        f"{csv_io.MAX_CSV_ROWS:,} rows, {csv_io.MAX_CSV_COLUMNS:,} columns, "
        f"{csv_io.MAX_CSV_CELLS:,} data cells, and {csv_io.MAX_CSV_CELL_CHARS:,} characters per cell."
    )

    uploaded = st.file_uploader(
        "CSV file", type=["csv"], help="Example columns: DOI, Title, Abstract",
        max_upload_size=csv_io.MAX_CSV_BYTES // (1024 * 1024),
        key="abstract_csv_upload",
    )

    if uploaded is not None:
        try:
            df = csv_io.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read the file: {ui.escape_markdown(exc)}")
            df = None

        if df is not None and len(df.columns) > 0:
            st.write(f"{len(df)} rows. Preview of the first 5:")
            st.dataframe(df.head(5), width="stretch")

            cols = list(df.columns)
            doi_guess = csv_io.guess_column(cols, csv_io.DOI_CANDIDATES)
            doi_choices = [NONE_OPTION] + cols
            doi_default = doi_guess if (doi_guess and doi_guess.lower() in
                                        [c.lower() for c in csv_io.DOI_CANDIDATES]) else NONE_OPTION

            title_guess = csv_io.guess_column(cols, csv_io.TITLE_CANDIDATES)
            abstract_guess = csv_io.guess_column(cols, csv_io.ABSTRACT_CANDIDATES)

            c1, c2, c3 = st.columns(3)
            doi_col = c1.selectbox("DOI column (optional)", doi_choices,
                                   index=doi_choices.index(doi_default))
            title_col = c2.selectbox(
                "Title column", cols,
                index=cols.index(title_guess) if title_guess else None,
                placeholder="Select the title column",
            )
            abstract_col = c3.selectbox(
                "Abstract column", cols,
                index=cols.index(abstract_guess) if abstract_guess else None,
                placeholder="Select the abstract column",
            )
            if not title_guess or not abstract_guess:
                st.caption("No column name matched; choose the title and abstract columns yourself.")

            replace_ok = bool(title_col) and bool(abstract_col)
            if replace_ok and state.has_papers():
                st.warning(
                    "This project already has papers. Loading this file **replaces** "
                    "them and clears all AI results and decisions."
                )
                replace_ok = st.checkbox(
                    "I understand this replaces all papers, AI results, and decisions.",
                    key="confirm_replace_papers",
                )
            if st.button("Load papers into project", type="primary", disabled=not replace_ok):
                rows = []
                for _, row in df.iterrows():
                    rows.append(state.new_paper(
                        doi="" if doi_col == NONE_OPTION else _clean(row.get(doi_col)),
                        title=_clean(row.get(title_col)),
                        abstract=_clean(row.get(abstract_col)),
                    ))
                try:
                    documents.replace_papers(
                        state.prepare_save(), rows,
                        (list(df.columns), json.loads(df.to_json(orient="records"))),
                    )
                except db.DatabaseError as exc:
                    st.error(f"Could not save the imported papers: {ui.escape_markdown(exc)}")
                else:
                    # A failed cleanup stays in the saved queue for a later retry.
                    with contextlib.suppress(db.DatabaseError):
                        documents.cleanup(state.prepare_save())
                    st.session_state.pop("confirm_replace_papers", None)
                    st.session_state.pop("abstract_csv_upload", None)
                    st.rerun()


def _run_screening(papers: list[dict], *, include: tuple[str, ...] = (runs.NEW,),
                   refresh: bool = False) -> None:
    planned = runs.plan(papers, STAGE, screening_spec, None, models, attempts, include=include)
    progress = st.progress(0.0, text="Starting…")
    ok = runs.execute(papers, STAGE, screening_spec, None, models, include=include, refresh=refresh,
                      progress=lambda n, total: progress.progress(n / total))
    progress.empty()
    if ok:
        st.session_state["abstract:run_notice"] = run_controls.finished(planned)
    if ok or state.prepare_save().store.get("pending_ai_completion") or state.has_unsaved_results():
        st.rerun()


screening_spec = {"criteria": criteria, "prompt_version": prompts.ABSTRACT_PROMPT_VERSION}
if state.has_papers():
    with st.container(border=True):
        st.subheader("AI screening")
        notice = st.session_state.pop("abstract:run_notice", None)
        if notice:
            st.success(notice)
        s = state.stage_summary(STAGE)
        pending = [p for _, p in state.pending_papers(STAGE)]
        stale = [p for _, p in state.stale_papers(STAGE)]
        invalid = state.invalid_papers(STAGE)
        stale_uids = {p["uid"] for p in stale}
        current = [p for _, p in state.stage_papers(STAGE) if p["uid"] not in stale_uids]
        undecided = runs.awaiting_models(current, STAGE)

        parts = [f"{s['total']} papers", f"{s['ai_done']} screened"]
        if s["ai_failed"]:
            parts.append(f"{s['ai_failed']} failed")
        if invalid:
            parts.append(f"{len(invalid)} invalid response")
        parts.append(f"{len(pending)} remaining")
        st.caption("  ·  ".join(parts))

        if not criteria.strip():
            st.info("Write your eligibility criteria above before running the AI.")
        attempts = run_controls.attempts(STAGE)
        new_tasks = saved = refresh_tasks = refresh_saved = []
        known = None
        if attempts is not None:
            known = runs.outcomes(current + stale, STAGE, screening_spec, None, attempts)
            try:
                new_tasks = runs.plan(undecided, STAGE, screening_spec, None, models, attempts, known=known)
                refresh_tasks = runs.plan(stale, STAGE, screening_spec, None, models, attempts,
                                          include=(runs.NEW, runs.FAILED, runs.INVALID), known=known)
            except ValueError as exc:
                st.warning(str(exc))
                attempts = None
            else:
                saved = runs.restorable(undecided, STAGE, screening_spec, None, attempts, known=known)
                refresh_saved = runs.restorable(stale, STAGE, screening_spec, None, attempts,
                                                refresh=True, known=known)
        ready = bool(attempts is not None and criteria.strip()
                     and all(m.get("api_key") for m in models))
        if new_tasks:
            st.caption(
                f"Each remaining paper is screened by {len(models)} selected model(s), "
                "billed to your own keys. Progress is saved after every call, so an "
                "interrupted run resumes where it stopped. Papers you already decided "
                "yourself are not re-billed."
            )
        if saved:
            st.caption(f"{len(saved)} paper(s) already have a saved AI answer; "
                       "it is shown without a new call.")
        if st.button(
            f"▶ Run AI screening ({runs.describe(new_tasks)})",
            type="primary",
            disabled=not ready or not (new_tasks or saved),
        ):
            _run_screening(undecided)
        if attempts is not None:
            run_controls.unlinked_note(STAGE, current, screening_spec, None, models, attempts)
            run_controls.retry_controls(
                STAGE, current, screening_spec, None, models, attempts,
                lambda papers, include: _run_screening(papers, include=include),
                disabled=not ready, known=known,
            )

        if stale:
            reviewed_stale = sum(
                1 for p in stale if state.is_reviewed(p, STAGE)
            )
            st.warning(
                f"⚠️ {len(stale)} paper(s) have results from **earlier criteria**. "
                "Outdated results don't count as reviewed and don't advance to "
                "the next stage — refresh them below."
                + (f" {reviewed_stale} had review decisions; a refresh moves them, "
                   "with the old result, to the paper's history." if reviewed_stale else "")
            )
            rerun_col, archive_col = st.columns(2)
            if rerun_col.button(
                f"↻ Re-run outdated papers ({runs.describe(refresh_tasks, len(stale))})",
                disabled=not ready or not (refresh_tasks or refresh_saved),
                width="stretch",
            ):
                _run_screening(stale, include=(runs.NEW, runs.FAILED, runs.INVALID), refresh=True)
            archive_ok = not reviewed_stale or archive_col.checkbox(
                f"I understand this resets {reviewed_stale} recorded decision(s).",
                key="confirm_archive_abstract",
            )
            if archive_col.button(
                f"Archive outdated without re-running ({len(stale)})",
                help="Moves the old results to history and marks the papers "
                     "pending — useful if you screen manually or want to run "
                     "the AI later.",
                width="stretch",
                disabled=not archive_ok,
            ):
                state.commit(lambda: [state.archive_ai_result(p, STAGE) for p in stale])
                st.session_state.pop("confirm_archive_abstract", None)
                st.rerun()
            if attempts is not None:
                run_controls.retry_controls(
                    STAGE, stale, screening_spec, None, models, attempts,
                    lambda papers, include: _run_screening(papers, include=include, refresh=True),
                    disabled=not ready, scope="outdated", categories=(runs.UNKNOWN,), known=known,
                )

    run_controls.panel(STAGE, state.papers(), screening_spec, models=models)

    with st.container(border=True):
        st.subheader("Review")
        ps = state.papers()
        total = len(ps)
        i = state.cursor(STAGE)
        i = max(0, min(total - 1, i))
        paper = ps[i]
        stg = state.stage_state(paper, STAGE)
        s = state.stage_summary(STAGE)
        chash = state.criteria_hash(criteria)

        st.progress(
            s["reviewed"] / total if total else 0.0,
            text=f"Reviewed {s['reviewed']} / {total}",
        )

        jump_key = f"jump_{STAGE}_{state.active_id()}"
        i = ui.jump_to_paper(
            jump_key, i,
            [f"{n + 1}. {(p.get('title') or '(no title)')[:70]} — {state.display_label(p, STAGE, chash)}"
             for n, p in enumerate(ps)],
        )
        state.goto(STAGE, i, total)
        paper = ps[i]
        stg = state.stage_state(paper, STAGE)

        nav_prev, nav_mid, nav_next = st.columns([1, 2, 1])
        if nav_prev.button("‹ Prev", disabled=i == 0, width="stretch"):
            st.session_state[f"{jump_key}:moved"] = i - 1
            state.goto(STAGE, i - 1, total)
            st.rerun()
        nav_mid.markdown(
            f"<div style='text-align:center;color:#6A7280;font-size:13px;padding-top:6px;'>"
            f"Paper {i + 1} / {total}</div>",
            unsafe_allow_html=True,
        )
        if nav_next.button("Next ›", disabled=i >= total - 1, width="stretch"):
            st.session_state[f"{jump_key}:moved"] = i + 1
            state.goto(STAGE, i + 1, total)
            st.rerun()

        st.markdown(f"#### {ui.escape_markdown(paper['title'] or '(no title)')}")
        if paper.get("doi"):
            label, url = ui.escape_markdown(paper["doi"]), ui.doi_url(paper["doi"])
            st.markdown(f"[{label}]({url})" if url else label)
        st.text(paper["abstract"] or "(no abstract)")

        ai_verdict = stg.get("ai_verdict")
        if ai_verdict:
            color = BADGE_COLORS.get(ai_verdict, "#5A626B")
            st.html(
                f"""<div style="background:#F6F8F7;border:1px solid #E2EBE7;border-radius:8px;
                            padding:10px 12px;margin:6px 0 10px;">
                  <span style="background:{color};color:#fff;font-size:12px;font-weight:600;
                               padding:3px 10px;border-radius:6px;">
                    AI verdict: {state.VERDICT_LABELS[ai_verdict]}</span>
                  <div style="font-size:12px;color:#5A626B;font-style:italic;margin-top:5px;">
                    {html.escape(stg.get("ai_reason") or "")}</div>
                </div>""",
            )
        elif stg.get("ai_error"):
            if stg.get("ai_error_kind") == state.ERROR_INVALID_RESPONSE:
                st.error(
                    f"The AI returned an invalid response (not retried "
                    f"automatically): {ui.escape_markdown(stg['ai_error'])}"
                )
            else:
                st.error(f"AI screening failed for this paper: {ui.escape_markdown(stg['ai_error'])}")
        else:
            st.info("Not screened by the AI yet — run AI screening above, or set a verdict yourself.")

        run_controls.screening_review(paper, STAGE, screening_spec)
        controls.decision_controls(
            paper, STAGE, chash, position=i, total=total,
            stale_note=(
                "This paper's result was produced under earlier criteria, so it is "
                "outdated: it does not count as reviewed and cannot be agreed or "
                "disagreed with. Re-run or archive it in the AI screening section above."
            ),
        )

        st.divider()
        final = state.display_label(paper, STAGE, chash)
        st.caption(
            f"Your decision for this paper: **{final}**"
            + ("  ·  not reviewed yet" if not state.is_reviewed(paper, STAGE, chash) else "")
        )
        st.caption(
            "Include and Unsure papers advance to **Full-text Screening**. "
            "See **Review Summary** for progress and export."
        )
