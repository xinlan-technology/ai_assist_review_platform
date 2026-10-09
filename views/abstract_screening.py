import html

import pandas as pd
import streamlit as st

from core import auth, csv_io, db, fulltext_storage, ui
from core.llm import InvalidModelResponse, error_message
from features.screening import controls, judge, prompts
from features.workflow import state

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
st.caption(f"Project: **{state.active_name()}**")


def _clean(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


with st.sidebar:
    provider, model, api_key = ui.model_controls()
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

    uploaded = st.file_uploader(
        "CSV file", type=["csv"], help="Example columns: DOI, Title, Abstract",
        key="abstract_csv_upload",
    )

    if uploaded is not None:
        try:
            df = csv_io.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read the file: {exc}")
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
                attached = {}
                try:
                    if state.has_papers():
                        attached = db.load_fulltexts(auth.current_user(), state.active_id())
                except db.DatabaseError as exc:
                    st.error(f"Could not inspect existing PDF attachments: {exc}")
                else:
                    rows = []
                    for _, row in df.iterrows():
                        rows.append(state.new_paper(
                            doi="" if doi_col == NONE_OPTION else _clean(row.get(doi_col)),
                            title=_clean(row.get(title_col)),
                            abstract=_clean(row.get(abstract_col)),
                        ))
                    state.commit(lambda: state.replace_papers(rows, df.reset_index(drop=True)))
                    # Persist the replacement before cleaning up old PDFs.
                    try:
                        fulltext_storage.delete_many(
                            [m.get("storage_key") for m in attached.values()
                             if m.get("storage_key")]
                        )
                        db.delete_project_fulltexts(auth.current_user(), state.active_id())
                    except (db.DatabaseError, fulltext_storage.FulltextStorageError) as exc:
                        st.warning(
                            "The new paper list was saved, but some old PDF data "
                            f"could not be cleaned up: {exc}"
                        )
                    else:
                        # Require a fresh upload and confirmation for another replacement.
                        st.session_state.pop("confirm_replace_papers", None)
                        st.session_state.pop("abstract_csv_upload", None)
                        st.rerun()


def _run_screening(targets: list[tuple[int, dict]]) -> bool:
    """Save each result before the next call; stop on persistence failure."""
    if not state.save_active():
        st.error("Stopping before the run — the project could not be saved.")
        return False
    chash = state.criteria_hash(criteria)
    progress = st.progress(0.0, text="Starting…")
    total = len(targets)
    call_failures = invalid_responses = 0
    for n, (_, paper) in enumerate(targets):
        try:
            result = judge.judge_abstract(
                provider, model, api_key, criteria,
                paper["title"], paper["abstract"],
            )
            state.set_ai_result(
                paper, STAGE, result["verdict"], result["reason"],
                provider, model, chash, prompts.ABSTRACT_PROMPT_VERSION,
            )
        except InvalidModelResponse as exc:
            state.set_ai_error(paper, STAGE, error_message(exc, api_key), state.ERROR_INVALID_RESPONSE)
            invalid_responses += 1
        except Exception as exc:
            state.set_ai_error(paper, STAGE, error_message(exc, api_key), state.ERROR_CALL_FAILED)
            call_failures += 1
        if not state.save_result():
            progress.empty()
            st.rerun()
        progress.progress((n + 1) / total, text=f"Screening… {n + 1}/{total}")
    progress.empty()
    if call_failures:
        st.warning(
            f"{call_failures} call(s) failed — run again to retry just those."
        )
    if invalid_responses:
        st.warning(
            f"{invalid_responses} paper(s) got an invalid model response. They are "
            "NOT retried automatically — set a verdict yourself in the review below, "
            "or use “Retry invalid responses”."
        )
    if not call_failures and not invalid_responses:
        st.success("Done. Review the verdicts below.")
    return True


if state.has_papers():
    with st.container(border=True):
        st.subheader("AI screening")
        s = state.stage_summary(STAGE)
        pending = state.pending_papers(STAGE)
        stale = state.stale_papers(STAGE)
        invalid = state.invalid_papers(STAGE)

        parts = [f"{s['total']} papers", f"{s['ai_done']} screened"]
        if s["ai_failed"]:
            parts.append(f"{s['ai_failed']} failed")
        if invalid:
            parts.append(f"{len(invalid)} invalid response")
        parts.append(f"{len(pending)} remaining")
        st.caption("  ·  ".join(parts))

        if not criteria.strip():
            st.info("Write your eligibility criteria above before running the AI.")
        if pending:
            st.caption(
                f"Each remaining paper is screened by {provider} · {model} "
                f"(~{len(pending)} API calls), billed to your own key. Progress is "
                "saved after every paper, so an interrupted run resumes where it "
                "stopped. Papers you already decided yourself are not re-billed."
            )
        if st.button(
            f"▶ Run AI screening ({len(pending)} remaining)",
            type="primary",
            disabled=not api_key or not criteria.strip() or not pending,
        ):
            _run_screening(pending)

        if stale:
            reviewed_stale = sum(
                1 for _, p in stale if state.is_reviewed(p, STAGE)
            )
            st.warning(
                f"⚠️ {len(stale)} paper(s) have results from **earlier criteria**. "
                "Outdated results don't count as reviewed and don't advance to "
                "the next stage — refresh them below."
                + (f" {reviewed_stale} had review decisions, which a refresh "
                   "moves to the project history." if reviewed_stale else "")
            )
            rerun_col, archive_col = st.columns(2)
            if rerun_col.button(
                f"↻ Re-run outdated papers ({len(stale)})",
                disabled=not api_key or not criteria.strip(),
                width="stretch",
            ):
                state.commit(lambda: [state.archive_ai_result(p, STAGE) for _, p in stale])
                if _run_screening(stale):
                    st.rerun()
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
                state.commit(lambda: [state.archive_ai_result(p, STAGE) for _, p in stale])
                st.session_state.pop("confirm_archive_abstract", None)
                st.rerun()

        if invalid and st.button(
            f"↻ Retry invalid responses ({len(invalid)})",
            disabled=not api_key or not criteria.strip(),
        ) and _run_screening(invalid):
            st.rerun()

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

        st.markdown(f"#### {paper['title'] or '(no title)'}")
        if paper.get("doi"):
            st.markdown(f"[{paper['doi']}]({ui.doi_url(paper['doi'])})")
        st.write(paper["abstract"] or "_(no abstract)_")

        ai_verdict = stg.get("ai_verdict")
        if ai_verdict:
            color = BADGE_COLORS.get(ai_verdict, "#5A626B")
            st.markdown(
                f"""<div style="background:#F6F8F7;border:1px solid #E2EBE7;border-radius:8px;
                            padding:10px 12px;margin:6px 0 10px;">
                  <span style="background:{color};color:#fff;font-size:12px;font-weight:600;
                               padding:3px 10px;border-radius:6px;">
                    AI verdict: {state.VERDICT_LABELS[ai_verdict]}</span>
                  <div style="font-size:12px;color:#5A626B;font-style:italic;margin-top:5px;">
                    {html.escape(stg.get("ai_reason") or "")}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        elif stg.get("ai_error"):
            if stg.get("ai_error_kind") == state.ERROR_INVALID_RESPONSE:
                st.error(
                    f"The AI returned an invalid response (not retried "
                    f"automatically): {stg['ai_error']}"
                )
            else:
                st.error(f"AI screening failed for this paper: {stg['ai_error']}")
        else:
            st.info("Not screened by the AI yet — run AI screening above, or set a verdict yourself.")

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
