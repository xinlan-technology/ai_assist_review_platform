from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import streamlit as st

from core import auth, db, fulltext_storage, ui
from core.llm import pdf_input_issue
from features.screening import controls, prompts
from features.workflow import documents, run_controls, runs, state

STAGE = state.STAGE_FULLTEXT
MAX_UPLOAD_PDF_MB = fulltext_storage.MAX_UPLOAD_PDF_BYTES // (1024 * 1024)

BADGE_COLORS = {
    state.VERDICT_INCLUDE: "#0E6E55",
    state.VERDICT_EXCLUDE: "#B3261E",
}

ui.inject_base_css()
ui.app_header(
    "Full-text Screening",
    "Attach original PDFs, screen each one with your own prompt, then confirm or override the AI.",
)
auth.sidebar_user()

if not state.active_id():
    st.info("Open or create a project on the **My Projects** page first.")
    st.stop()
state.require_saved_results()
if STAGE not in state.stages():
    st.info("Full-text screening is not enabled for this project.")
    st.stop()

user = auth.current_user()
project_id = state.active_id()
st.caption(
    f"Project: **{ui.escape_markdown(state.active_name())}**  ·  Workflow: **{state.MODE_LABELS[state.mode()]}**"
)


def _paper_label(paper: dict) -> str:
    title = paper.get("title") or "(no title)"
    doi = paper.get("doi") or ""
    return f"{title} — {doi}" if doi else title


def _read_metadata() -> dict[str, dict]:
    try:
        return db.load_fulltexts(user, project_id)
    except db.DatabaseError as exc:
        st.error(ui.escape_markdown(exc))
        st.stop()


# Bound the cache because each entry holds a complete PDF.
@st.cache_data(show_spinner=False, ttl=300, max_entries=6)
def _cached_pdf(storage_key: str, digest: str) -> bytes:
    data = fulltext_storage.load_pdf(storage_key)
    if not digest or fulltext_storage.sha256(data) != digest:
        raise fulltext_storage.FulltextStorageError("The stored PDF differs from its metadata. Reattach it first.")
    return data


def _prepare_upload(uploaded) -> tuple[bytes, str, int | None, str]:
    data = uploaded.getvalue()
    if len(data) > fulltext_storage.MAX_UPLOAD_PDF_BYTES:
        raise fulltext_storage.FulltextStorageError(
            f"{uploaded.name} is larger than 50 MB. Compress it before uploading."
        )
    page_count = fulltext_storage.inspect_pdf(data)
    digest = fulltext_storage.sha256(data)
    return data, str(uploaded.name), page_count, digest


def _attach_prepared(
    paper: dict,
    data: bytes,
    filename: str,
    page_count: int | None,
    digest: str,
    metadata: dict[str, dict],
    *,
    new_paper: bool = False,
) -> bool:
    save = state.prepare_save()
    failed_cleanup = st.session_state.setdefault("_uncommitted_pdf_cleanup", [])
    entry, changed = documents.attach(
        save, paper, data, filename, page_count, digest, failed_cleanup, new_paper=new_paper,
    )
    metadata[paper["uid"]] = {name: value for name, value in entry.items() if name != "paper_uid"}
    _cached_pdf.clear()
    _cleanup_removed_pdfs()
    return changed


def _show_pdf(pdf_bytes: bytes) -> None:
    st.pdf(pdf_bytes, height=780)


def _cleanup_removed_pdfs() -> bool:
    try:
        return documents.cleanup(state.prepare_save())
    except db.DatabaseError:
        return False


with st.sidebar:
    provider, model, api_key = ui.model_controls()
    models = ui.additional_models(provider, model, api_key)
    st.divider()
    if st.button("💾 Save project", width="stretch", key="save_fulltext_project") and state.save_active():
        st.toast("Saved.")

with st.container(border=True):
    st.subheader("Full-text screening prompt")
    st.caption(
        "Write your own inclusion/exclusion instructions. The platform sends this prompt "
        "together with the original PDF and only fixes the output to Include/Exclude + reason."
    )
    criteria = st.text_area(
        "Full-text prompt",
        value=state.config().get("fulltext_criteria", ""),
        placeholder=prompts.CRITERIA_PLACEHOLDER,
        height=220,
        label_visibility="collapsed",
        key=f"fulltext_prompt_{project_id}",
    )
    if criteria != state.config().get("fulltext_criteria", ""):
        state.commit(lambda: state.config().update(fulltext_criteria=criteria))

fulltexts = _read_metadata()
eligible = state.stage_papers(STAGE)

ui.pending_file_cleanup()

if state.pending_pdf_deletions():
    st.warning(f"{len(state.pending_pdf_deletions())} deleted or replaced PDF file(s) still await cleanup. "
               "The deletion targets are saved; retry when storage is available.")
    if st.button("Retry deleting removed PDFs"):
        if _cleanup_removed_pdfs():
            st.session_state["_fulltext_upload_notice"] = "Pending PDF cleanup completed."
        st.rerun()

with st.container(border=True):
    st.subheader("Original PDFs")
    try:
        fulltext_storage.ensure_configured()
        storage_ready = True
        st.caption(f"Storage: {fulltext_storage.backend_label()} · original PDFs are kept unchanged.")
    except fulltext_storage.FulltextStorageError as exc:
        storage_ready = False
        st.error(ui.escape_markdown(exc))

    if state.mode() == state.MODE_DIRECT:
        st.caption(
            "Upload one or more PDFs. Each PDF becomes one paper; its filename is used as "
            "the initial title. Re-uploading the same file is skipped."
        )
        direct_uploads = st.file_uploader(
            "PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            max_upload_size=MAX_UPLOAD_PDF_MB,
            key="direct_pdf_uploads",
        )
        if st.button(
            "Add PDFs to project",
            type="primary",
            disabled=not storage_ready or not direct_uploads,
            key="add_direct_pdfs",
        ):
            added = skipped = 0
            errors: list[str] = []
            current_uids = {paper["uid"] for paper in state.papers()}
            known_hashes = {
                meta.get("sha256")
                for uid, meta in fulltexts.items()
                if uid in current_uids and meta.get("sha256")
            }
            for position, uploaded in enumerate(direct_uploads):
                try:
                    data, filename, pages, digest = _prepare_upload(uploaded)
                    if digest in known_hashes:
                        skipped += 1
                        continue
                    paper = state.new_paper("", Path(filename).stem, "")
                    changed = _attach_prepared(
                        paper, data, filename, pages, digest, fulltexts, new_paper=True,
                    )
                    known_hashes.add(digest)
                    added += int(changed)
                except Exception as exc:
                    errors.append(f"{getattr(uploaded, 'name', 'PDF')}: {exc}")
                    if isinstance(exc, db.DatabaseError):
                        # Later files were not attempted; name them so none is missed.
                        untried = [getattr(item, "name", "PDF") for item in direct_uploads[position + 1:]]
                        if untried:
                            errors.append("Not processed after the database error: " + ", ".join(untried))
                        break
            if errors:
                st.session_state["_fulltext_upload_errors"] = errors
            if added or skipped:
                st.session_state["_fulltext_upload_notice"] = (
                    f"Added {added} PDF(s)" + (f"; skipped {skipped} duplicate(s)." if skipped else ".")
                )
            if errors or added or skipped:
                st.rerun()
    else:
        if not eligible:
            st.info(
                "No papers have advanced from abstract screening yet. Confirm Include or "
                "Unsure decisions on the Abstract Screening page first."
            )
        else:
            st.caption(
                "Choose an abstract-screened paper, then attach its matching PDF. The explicit "
                "selection prevents an incorrect filename match. Uploading again replaces it."
            )
            option_uids = [p["uid"] for _, p in eligible]
            by_uid = {p["uid"]: p for _, p in eligible}
            selected_uid = st.selectbox(
                "Paper",
                option_uids,
                format_func=lambda uid: (
                    ("✓ " if uid in fulltexts else "○ ") + _paper_label(by_uid[uid])
                ),
                key="prisma_pdf_target",
            )
            matched_upload = st.file_uploader(
                "Matching PDF",
                type=["pdf"],
                accept_multiple_files=False,
                max_upload_size=MAX_UPLOAD_PDF_MB,
                key=f"matched_pdf_{selected_uid}",
            )
            if st.button(
                "Attach PDF" if selected_uid not in fulltexts else "Replace PDF",
                type="primary",
                disabled=not storage_ready or matched_upload is None,
                key="attach_prisma_pdf",
            ):
                try:
                    data, filename, pages, digest = _prepare_upload(matched_upload)
                    changed = _attach_prepared(
                        by_uid[selected_uid], data, filename, pages, digest, fulltexts
                    )
                except Exception as exc:
                    st.error(f"Could not attach the PDF: {ui.escape_markdown(exc)}")
                else:
                    st.session_state["_fulltext_upload_notice"] = (
                        "PDF attached." if changed else "That exact PDF was already attached."
                    )
                    st.rerun()

    notice = st.session_state.pop("_fulltext_upload_notice", None)
    if notice:
        st.success(notice)
    upload_errors = st.session_state.pop("_fulltext_upload_errors", None)
    if upload_errors:
        st.error("Some PDFs could not be added:\n\n" + ui.escape_markdown("\n\n".join(upload_errors)))

    eligible = state.stage_papers(STAGE)
    if eligible:
        status_rows = []
        for _, paper in eligible:
            meta = fulltexts.get(paper["uid"])
            issue = (
                pdf_input_issue(
                    provider,
                    model,
                    meta.get("file_size"),
                    meta.get("page_count"),
                )
                if meta and meta.get("status") == "ok"
                else None
            )
            status_rows.append(
                {
                    "Paper": _paper_label(paper),
                    "PDF": meta.get("filename") if meta else "Not attached",
                    "Pages": meta.get("page_count") if meta else None,
                    "AI status": (
                        f"Manual only — {issue}"
                        if issue
                        else "Ready"
                        if meta and meta.get("status") == "ok"
                        else "Unavailable"
                        if meta
                        else "Not attached"
                    ),
                }
            )
        st.dataframe(pd.DataFrame(status_rows), width="stretch", hide_index=True)


def _has_ready_pdf(paper: dict) -> bool:
    meta = fulltexts.get(paper["uid"])
    return bool(
        meta
        and meta.get("status") == "ok"
        and meta.get("storage_key")
        and meta.get("sha256")
        and any(not pdf_input_issue(m["provider"], m["model"], meta.get("file_size"), meta.get("page_count"))
                for m in models)
    )


def _run_screening(papers: list[dict], *, include: tuple[str, ...] = (runs.NEW,),
                   refresh: bool = False) -> None:
    planned = runs.plan(papers, STAGE, screening_spec, fulltexts, models, attempts, include=include)
    progress = st.progress(0.0, text="Starting…")
    ok = runs.execute(papers, STAGE, screening_spec, fulltexts, models, include=include,
                      refresh=refresh, progress=lambda n, total: progress.progress(n / total))
    progress.empty()
    if ok:
        st.session_state["fulltext:run_notice"] = run_controls.finished(planned)
    if ok or state.prepare_save().store.get("pending_ai_completion") or state.has_unsaved_results():
        st.rerun()


screening_spec = {"criteria": criteria, "prompt_version": prompts.FULLTEXT_PROMPT_VERSION}
eligible = state.stage_papers(STAGE)
if eligible:
    with st.container(border=True):
        st.subheader("AI full-text screening")
        notice = st.session_state.pop("fulltext:run_notice", None)
        if notice:
            st.success(notice)
        summary = state.stage_summary(STAGE)
        pending = [p for _, p in state.pending_papers(STAGE) if _has_ready_pdf(p)]
        stale_all = [p for _, p in state.stale_papers(STAGE)]
        stale_ready = [p for p in stale_all if _has_ready_pdf(p)]
        stale_uids = {p["uid"] for p in stale_all}
        current = [p for _, p in eligible if p["uid"] not in stale_uids and _has_ready_pdf(p)]
        undecided = runs.awaiting_models(current, STAGE)
        missing = sum(1 for _, paper in eligible if not _has_ready_pdf(paper))

        parts = [
            f"{summary['total']} eligible",
            f"{summary['ai_done']} AI screened",
            f"{len(pending)} ready to run",
        ]
        if missing:
            parts.append(f"{missing} without a usable PDF")
        st.caption("  ·  ".join(parts))
        if not criteria.strip():
            st.info("Write your full-text screening prompt above before running the AI.")
        if any(
            (meta := fulltexts.get(paper["uid"]))
            and pdf_input_issue(
                provider, model, meta.get("file_size"), meta.get("page_count")
            )
            for _, paper in eligible
        ):
            st.info(
                "Some PDFs are outside the selected model's size or page limits. They "
                "remain available for human review and are not sent to the model."
            )
        attempts = run_controls.attempts(STAGE)
        new_tasks = saved = refresh_tasks = refresh_saved = []
        known = None
        if attempts is not None:
            known = runs.outcomes(current + stale_ready, STAGE, screening_spec, fulltexts, attempts)
            try:
                new_tasks = runs.plan(undecided, STAGE, screening_spec, fulltexts, models, attempts, known=known)
                refresh_tasks = runs.plan(stale_ready, STAGE, screening_spec, fulltexts, models, attempts,
                                          include=(runs.NEW, runs.FAILED, runs.INVALID), known=known)
            except ValueError as exc:
                st.warning(str(exc))
                attempts = None
            else:
                saved = runs.restorable(undecided, STAGE, screening_spec, fulltexts, attempts, known=known)
                refresh_saved = runs.restorable(stale_ready, STAGE, screening_spec, fulltexts,
                                                attempts, refresh=True, known=known)
        ready = bool(attempts is not None and criteria.strip()
                     and all(m.get("api_key") for m in models))
        if new_tasks:
            st.caption(
                f"The original PDF is sent to {len(models)} selected model(s). "
                "Results are saved after every call."
            )
        if saved:
            st.caption(f"{len(saved)} paper(s) already have a saved AI answer; "
                       "it is shown without a new call.")
        if st.button(
            f"▶ Run AI on ready PDFs ({runs.describe(new_tasks)})",
            type="primary",
            disabled=not ready or not (new_tasks or saved),
            key="run_fulltext_ai",
        ):
            _run_screening(undecided)
        if attempts is not None:
            run_controls.unlinked_note(STAGE, current, screening_spec, fulltexts, models, attempts)
            run_controls.retry_controls(
                STAGE, current, screening_spec, fulltexts, models, attempts,
                lambda papers, include: _run_screening(papers, include=include),
                disabled=not ready, known=known,
            )

        if stale_all:
            reviewed_stale = sum(1 for p in stale_all if state.final_verdict(p, STAGE))
            st.warning(
                f"{len(stale_all)} result(s) use an earlier prompt and no longer count."
                + (f" {reviewed_stale} had review decisions; a refresh moves them, "
                   "with the old result, to the paper's history." if reviewed_stale else "")
            )
            rerun_col, archive_col = st.columns(2)
            if rerun_col.button(
                f"↻ Re-run outdated PDFs ({runs.describe(refresh_tasks, len(stale_ready))})",
                disabled=not ready or not (refresh_tasks or refresh_saved),
                width="stretch",
            ):
                _run_screening(stale_ready, include=(runs.NEW, runs.FAILED, runs.INVALID), refresh=True)
            archive_ok = not reviewed_stale or archive_col.checkbox(
                f"I understand this resets {reviewed_stale} recorded decision(s).",
                key="confirm_archive_fulltext",
            )
            if archive_col.button(
                f"Archive outdated without AI ({len(stale_all)})",
                width="stretch",
                disabled=not archive_ok,
                help="Moves the old results to history and marks those papers pending.",
            ):
                state.commit(lambda: [state.archive_ai_result(p, STAGE) for p in stale_all])
                st.session_state.pop("confirm_archive_fulltext", None)
                st.rerun()
            if attempts is not None:
                run_controls.retry_controls(
                    STAGE, stale_ready, screening_spec, fulltexts, models, attempts,
                    lambda papers, include: _run_screening(papers, include=include, refresh=True),
                    disabled=not ready, scope="outdated", categories=(runs.UNKNOWN,), known=known,
                )

    run_controls.panel(STAGE, [p for _, p in eligible], screening_spec, fulltexts, models=models)

    with st.container(border=True):
        st.subheader("Human review")
        eligible = state.stage_papers(STAGE)
        total = len(eligible)
        pos = max(0, min(total - 1, state.cursor(STAGE)))
        _, paper = eligible[pos]
        stg = state.stage_state(paper, STAGE)
        meta = fulltexts.get(paper["uid"])
        chash = state.criteria_hash(criteria)
        summary = state.stage_summary(STAGE)

        st.progress(
            summary["reviewed"] / summary["total"] if summary["total"] else 0.0,
            text=f"Reviewed {summary['reviewed']} / {summary['total']}",
        )
        jump_key = f"jump_{STAGE}_{project_id}"
        pos = ui.jump_to_paper(
            jump_key, pos,
            [f"{n + 1}. {(p.get('title') or '(no title)')[:70]} — "
             f"{state.display_label(p, STAGE, chash)}"
             + ("" if fulltexts.get(p["uid"]) else " · no PDF")
             for n, (_, p) in enumerate(eligible)],
        )
        state.goto(STAGE, pos, total)
        _, paper = eligible[pos]
        stg = state.stage_state(paper, STAGE)
        meta = fulltexts.get(paper["uid"])

        nav_prev, nav_mid, nav_next = st.columns([1, 2, 1])
        if nav_prev.button("‹ Prev", disabled=pos == 0, width="stretch"):
            st.session_state[f"{jump_key}:moved"] = pos - 1
            state.goto(STAGE, pos - 1, total)
            st.rerun()
        nav_mid.markdown(
            f"<div style='text-align:center;color:#6A7280;font-size:13px;padding-top:6px;'>"
            f"Paper {pos + 1} / {total}</div>",
            unsafe_allow_html=True,
        )
        if nav_next.button("Next ›", disabled=pos >= total - 1, width="stretch"):
            st.session_state[f"{jump_key}:moved"] = pos + 1
            state.goto(STAGE, pos + 1, total)
            st.rerun()

        st.markdown(f"#### {ui.escape_markdown(paper.get('title') or '(no title)')}")
        if paper.get("doi"):
            label, url = ui.escape_markdown(paper["doi"]), ui.doi_url(paper["doi"])
            st.markdown(f"[{label}]({url})" if url else label)

        pdf_col, decision_col = st.columns([3, 2], gap="large")
        with pdf_col:
            st.markdown("##### Original PDF")
            pdf_bytes = None
            attached = bool(meta and meta.get("storage_key") and meta.get("status") == "ok")
            if not attached:
                st.warning(
                    "No PDF is attached to this paper. You can still record your own "
                    "verdict if you have reviewed the full text elsewhere, but the AI "
                    "cannot screen it. A missing PDF alone is not an exclusion reason."
                )
            else:
                try:
                    pdf_bytes = _cached_pdf(meta["storage_key"], meta.get("sha256") or "")
                except fulltext_storage.FulltextStorageError as exc:
                    st.error(ui.escape_markdown(exc))
                if pdf_bytes:
                    st.caption(
                        ui.escape_markdown(meta.get('filename') or 'paper.pdf')
                        + (f" · {meta['page_count']} pages" if meta.get("page_count") else "")
                    )
                    st.download_button(
                        "Download original PDF",
                        data=pdf_bytes,
                        file_name=meta.get("filename") or "paper.pdf",
                        mime="application/pdf",
                        key=f"download_{paper['uid']}",
                    )
                    _show_pdf(pdf_bytes)

        with decision_col:
            st.markdown("##### AI + reviewer decision")
            review_disabled = attached and not pdf_bytes
            if review_disabled:
                st.warning(
                    "This paper's stored PDF could not be read, so a decision would "
                    "not rest on the document on file. Re-attach it first."
                )
            ai_verdict = stg.get("ai_verdict")
            if ai_verdict:
                color = BADGE_COLORS.get(ai_verdict, "#5A626B")
                st.html(
                    f"""<div style="background:#F6F8F7;border:1px solid #E2EBE7;
                                border-radius:8px;padding:10px 12px;margin:6px 0 10px;">
                      <span style="background:{color};color:#fff;font-size:12px;font-weight:600;
                                   padding:3px 10px;border-radius:6px;">
                        AI verdict: {state.VERDICT_LABELS[ai_verdict]}</span>
                      <div style="font-size:12px;color:#5A626B;font-style:italic;margin-top:7px;">
                        {html.escape(stg.get('ai_reason') or '')}</div>
                    </div>""",
                )
                st.caption(
                    ui.escape_markdown(f"{stg.get('provider') or ''} · {stg.get('model') or ''}".strip(" ·"))
                )
            elif stg.get("ai_error"):
                if stg.get("ai_error_kind") == state.ERROR_INVALID_RESPONSE:
                    st.error(f"Invalid AI response: {ui.escape_markdown(stg['ai_error'])}")
                else:
                    st.error(f"AI call failed: {ui.escape_markdown(stg['ai_error'])}")
            else:
                st.info(
                    "No AI verdict yet."
                    + (
                        " You can make an independent decision after reviewing the PDF."
                        if not review_disabled
                        else ""
                    )
                )

            run_controls.screening_review(paper, STAGE, screening_spec, fulltexts,
                                          disabled=review_disabled)
            controls.decision_controls(
                paper, STAGE, chash, position=pos, total=total,
                disabled=review_disabled,
                stale_note=(
                    "This result is outdated because the prompt changed. Re-run or "
                    "archive it first."
                ),
            )

            st.divider()
            final = state.display_label(paper, STAGE, chash)
            st.caption(
                f"Current final decision: **{final}**"
                + (" · not reviewed yet" if not state.is_reviewed(paper, STAGE, chash) else "")
            )
            st.caption("Confirmed Include papers advance to Data Extraction.")
            if final == "Include" and not attached:
                st.warning(
                    "Data extraction reads the PDF, so this paper cannot be extracted "
                    "until one is attached."
                )
            if final == "Include":
                st.page_link("views/extraction.py", label="Continue to Data Extraction →")

        if state.mode() == state.MODE_DIRECT:
            with st.expander("Fix this paper (wrong or unwanted PDF)"):
                st.caption(
                    "Replacing the PDF moves any decision and extraction made from "
                    "the old file into this paper's history. Removing the paper "
                    "deletes it, its decisions and its stored PDF for good."
                )
                replacement = st.file_uploader(
                    "Replacement PDF", type=["pdf"], key=f"replace_pdf_{paper['uid']}",
                    max_upload_size=MAX_UPLOAD_PDF_MB,
                )
                if st.button(
                    "Replace PDF", key=f"do_replace_{paper['uid']}",
                    disabled=replacement is None or not storage_ready,
                ):
                    try:
                        data, filename, pages, digest = _prepare_upload(replacement)
                        clash = next(
                            (uid for uid, other in fulltexts.items()
                             if uid != paper["uid"] and other.get("sha256") == digest),
                            None,
                        )
                        if clash:
                            raise fulltext_storage.FulltextStorageError(
                                "That exact PDF is already attached to another paper in "
                                "this project; remove that paper first if it is a duplicate."
                            )
                        changed = _attach_prepared(paper, data, filename, pages, digest, fulltexts)
                    except Exception as exc:
                        st.error(f"Could not replace the PDF: {ui.escape_markdown(exc)}")
                    else:
                        st.session_state["_fulltext_upload_notice"] = (
                            "PDF replaced." if changed else "That exact PDF was already attached."
                        )
                        st.rerun()
                st.divider()
                remove_ok = st.checkbox(
                    "I understand this permanently removes the paper, its decisions and its PDF.",
                    key=f"confirm_remove_{paper['uid']}",
                )
                if st.button(
                    "Remove this paper", type="primary", disabled=not remove_ok,
                    key=f"do_remove_{paper['uid']}",
                ):
                    try:
                        documents.remove(state.prepare_save(), paper["uid"])
                    except db.DatabaseError as exc:
                        st.error(ui.escape_markdown(exc))
                        st.stop()
                    cleaned = _cleanup_removed_pdfs()
                    _cached_pdf.clear()
                    st.session_state[f"jump_{STAGE}_{project_id}:moved"] = 0
                    if cleaned:
                        st.session_state["_fulltext_upload_notice"] = "Paper and its stored PDF data removed."
                    st.rerun()
else:
    if state.mode() == state.MODE_DIRECT:
        st.info("Upload at least one PDF above to begin full-text screening.")
    else:
        st.info("No papers currently qualify for full-text screening.")
