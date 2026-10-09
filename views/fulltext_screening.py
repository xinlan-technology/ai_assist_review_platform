from __future__ import annotations

import copy
import html
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st

from core import auth, db, fulltext_storage, ui
from core.llm import (
    InvalidModelResponse,
    error_message,
    pdf_input_issue,
)
from features.screening import controls, judge, prompts
from features.workflow import state

STAGE = state.STAGE_FULLTEXT
MAX_UPLOAD_PDF_BYTES = 50 * 1024 * 1024

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
    f"Project: **{state.active_name()}**  ·  Workflow: **{state.MODE_LABELS[state.mode()]}**"
)


def _paper_label(paper: dict) -> str:
    title = paper.get("title") or "(no title)"
    doi = paper.get("doi") or ""
    return f"{title} — {doi}" if doi else title


def _read_metadata() -> dict[str, dict]:
    try:
        return db.load_fulltexts(user, project_id)
    except db.DatabaseError as exc:
        st.error(str(exc))
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
    if len(data) > MAX_UPLOAD_PDF_BYTES:
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
) -> bool:
    """Store the new PDF and commit its metadata before old-file cleanup."""
    uid = paper["uid"]
    old = metadata.get(uid) or {}
    same_source = bool(old.get("sha256") == digest and old.get("storage_key"))

    old_stages = copy.deepcopy(paper.get("stages", {}))
    old_cleanup = copy.deepcopy(state.pending_pdf_deletions())
    key = fulltext_storage.object_key(user, project_id, uid, f"{digest}-{uuid4().hex}")
    entry = {
        "paper_uid": uid, "filename": filename, "storage_key": key, "sha256": digest,
        "file_size": len(data), "page_count": page_count, "status": "ok", "error": None, "text": None,
    }
    try:
        fulltext_storage.save_pdf(key, data)
        state.archive_document_results(
            paper,
            old.get("sha256") or (f"unknown:{old.get('storage_key')}" if old.get("storage_key") else None),
            digest,
        )
        if old.get("storage_key"):
            state.pending_pdf_deletions().append({"paper_uid": None, "storage_key": old["storage_key"]})
        # Commit metadata and archived decisions in the same versioned transaction.
        if not state.save_active(fulltext=entry):
            raise RuntimeError("The project changed or could not be saved. Reload before attaching the PDF again.")
    except Exception:
        paper["stages"] = old_stages
        state.pending_pdf_deletions()[:] = old_cleanup
        # Roll back only this attempt's unique storage key.
        try:
            fulltext_storage.delete_pdf(key)
        except fulltext_storage.FulltextStorageError:
            st.session_state.setdefault("_uncommitted_pdf_cleanup", []).append(key)
        raise

    metadata[uid] = {name: value for name, value in entry.items() if name != "paper_uid"}
    _cached_pdf.clear()
    _cleanup_removed_pdfs()
    # Rewriting identical content repairs a missing object without changing provenance.
    return not same_source


def _show_pdf(pdf_bytes: bytes) -> None:
    st.pdf(pdf_bytes, height=780)


def _cleanup_removed_pdfs() -> bool:
    completed = []
    for target in state.pending_pdf_deletions():
        try:
            fulltext_storage.delete_pdf(target.get("storage_key"))
            if target.get("paper_uid"):
                db.delete_fulltext(user, project_id, target["paper_uid"])
        except (db.DatabaseError, fulltext_storage.FulltextStorageError):
            continue
        completed.append(target)
    if completed:
        def forget_completed():
            pending = state.pending_pdf_deletions()
            pending[:] = [target for target in pending if target not in completed]
        state.commit(forget_completed)
    return not state.pending_pdf_deletions()


with st.sidebar:
    provider, model, api_key = ui.model_controls()
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
        st.error(str(exc))

    if state.mode() == state.MODE_DIRECT:
        st.caption(
            "Upload one or more PDFs. Each PDF becomes one paper; its filename is used as "
            "the initial title. Re-uploading the same file is skipped."
        )
        direct_uploads = st.file_uploader(
            "PDF files",
            type=["pdf"],
            accept_multiple_files=True,
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
            for uploaded in direct_uploads:
                paper = None
                try:
                    data, filename, pages, digest = _prepare_upload(uploaded)
                    if digest in known_hashes:
                        skipped += 1
                        continue
                    paper = state.new_paper("", Path(filename).stem, "")
                    state.append_papers([paper])
                    changed = _attach_prepared(
                        paper, data, filename, pages, digest, fulltexts
                    )
                    known_hashes.add(digest)
                    added += int(changed)
                except Exception as exc:
                    if paper is not None:
                        state.remove_paper(paper["uid"])
                        fulltexts.pop(paper["uid"], None)
                    errors.append(f"{getattr(uploaded, 'name', 'PDF')}: {exc}")
            if errors:
                st.error("Some PDFs could not be added:\n\n" + "\n\n".join(errors))
            if added or skipped:
                st.session_state["_fulltext_upload_notice"] = (
                    f"Added {added} PDF(s)" + (f"; skipped {skipped} duplicate(s)." if skipped else ".")
                )
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
                key=f"matched_pdf_{selected_uid}",
            )
            if st.button(
                "Attach PDF" if selected_uid not in fulltexts else "Replace PDF",
                type="primary",
                disabled=not storage_ready or matched_upload is None,
                key="attach_prisma_pdf",
            ):
                try:
                    if not state.save_active():
                        raise RuntimeError("the project could not be saved before upload")
                    data, filename, pages, digest = _prepare_upload(matched_upload)
                    changed = _attach_prepared(
                        by_uid[selected_uid], data, filename, pages, digest, fulltexts
                    )
                except Exception as exc:
                    st.error(f"Could not attach the PDF: {exc}")
                else:
                    st.session_state["_fulltext_upload_notice"] = (
                        "PDF attached." if changed else "That exact PDF was already attached."
                    )
                    st.rerun()

    notice = st.session_state.pop("_fulltext_upload_notice", None)
    if notice:
        st.success(notice)

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
        and not pdf_input_issue(
            provider, model, meta.get("file_size"), meta.get("page_count")
        )
    )


def _run_screening(targets: list[tuple[int, dict]]) -> bool:
    if not state.save_active():
        st.error("Stopping before the run — the project could not be saved.")
        return False
    chash = state.criteria_hash(criteria)
    progress = st.progress(0.0, text="Starting…")
    total = len(targets)
    call_failures = invalid_responses = 0
    for n, (_, paper) in enumerate(targets):
        meta = fulltexts.get(paper["uid"]) or {}
        try:
            pdf_bytes = fulltext_storage.load_pdf(meta["storage_key"])
            if fulltext_storage.sha256(pdf_bytes) != meta.get("sha256"):
                raise fulltext_storage.FulltextStorageError("The stored PDF differs from its metadata. Reattach it first.")
            result = judge.judge_fulltext(
                provider,
                model,
                api_key,
                criteria,
                paper.get("title", ""),
                pdf_bytes,
                meta.get("filename") or "paper.pdf",
            )
            state.set_ai_result(
                paper,
                STAGE,
                result["verdict"],
                result["reason"],
                provider,
                model,
                chash,
                prompts.FULLTEXT_PROMPT_VERSION,
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
        progress.progress((n + 1) / total, text=f"Screening PDFs… {n + 1}/{total}")
    progress.empty()
    if call_failures:
        st.warning(f"{call_failures} call(s) failed; run again to retry only those papers.")
    if invalid_responses:
        st.warning(
            f"{invalid_responses} response(s) did not follow the required output. "
            "Decide them yourself or retry explicitly."
        )
    if not call_failures and not invalid_responses:
        st.success("AI full-text screening finished. Review the decisions below.")
    return True


eligible = state.stage_papers(STAGE)
if eligible:
    with st.container(border=True):
        st.subheader("AI full-text screening")
        summary = state.stage_summary(STAGE)
        pending_all = state.pending_papers(STAGE)
        pending = [(i, p) for i, p in pending_all if _has_ready_pdf(p)]
        stale_all = state.stale_papers(STAGE)
        stale_ready = [(i, p) for i, p in stale_all if _has_ready_pdf(p)]
        invalid_all = state.invalid_papers(STAGE)
        invalid_ready = [(i, p) for i, p in invalid_all if _has_ready_pdf(p)]
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
        if pending:
            st.caption(
                f"The original PDF is sent directly to {provider} · {model}; one API call "
                "per paper. Results are saved after every call."
            )
        if st.button(
            f"▶ Run AI on ready PDFs ({len(pending)})",
            type="primary",
            disabled=not api_key or not criteria.strip() or not pending,
            key="run_fulltext_ai",
        ):
            _run_screening(pending)

        if stale_all:
            st.warning(
                f"{len(stale_all)} result(s) use an earlier prompt and no longer count."
            )
            rerun_col, archive_col = st.columns(2)
            if rerun_col.button(
                f"↻ Re-run outdated PDFs ({len(stale_ready)})",
                disabled=not api_key or not criteria.strip() or not stale_ready,
                width="stretch",
            ):
                state.commit(lambda: [state.archive_ai_result(p, STAGE) for _, p in stale_ready])
                if _run_screening(stale_ready):
                    st.rerun()
            reviewed_stale = sum(1 for _, p in stale_all if state.final_verdict(p, STAGE))
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
                state.commit(lambda: [state.archive_ai_result(p, STAGE) for _, p in stale_all])
                st.session_state.pop("confirm_archive_fulltext", None)
                st.rerun()

        if invalid_ready and st.button(
            f"↻ Retry invalid responses ({len(invalid_ready)})",
            disabled=not api_key or not criteria.strip(),
        ) and _run_screening(invalid_ready):
            st.rerun()

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

        st.markdown(f"#### {paper.get('title') or '(no title)'}")
        if paper.get("doi"):
            st.markdown(f"[{paper['doi']}]({ui.doi_url(paper['doi'])})")

        pdf_col, decision_col = st.columns([3, 2], gap="large")
        with pdf_col:
            st.markdown("##### Original PDF")
            pdf_bytes = None
            attached = bool(meta and meta.get("storage_key") and meta.get("status") == "ok")
            if not attached:
                st.warning(
                    "No PDF is attached to this paper. You can still record your own "
                    "verdict — for example, exclude a paper whose full text cannot be "
                    "obtained — but the AI cannot screen it."
                )
            else:
                try:
                    pdf_bytes = _cached_pdf(meta["storage_key"], meta.get("sha256") or "")
                except fulltext_storage.FulltextStorageError as exc:
                    st.error(str(exc))
                if pdf_bytes:
                    st.caption(
                        f"{meta.get('filename') or 'paper.pdf'}"
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
                st.markdown(
                    f"""<div style="background:#F6F8F7;border:1px solid #E2EBE7;
                                border-radius:8px;padding:10px 12px;margin:6px 0 10px;">
                      <span style="background:{color};color:#fff;font-size:12px;font-weight:600;
                                   padding:3px 10px;border-radius:6px;">
                        AI verdict: {state.VERDICT_LABELS[ai_verdict]}</span>
                      <div style="font-size:12px;color:#5A626B;font-style:italic;margin-top:7px;">
                        {html.escape(stg.get('ai_reason') or '')}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"{stg.get('provider') or ''} · {stg.get('model') or ''}".strip(" ·")
                )
            elif stg.get("ai_error"):
                if stg.get("ai_error_kind") == state.ERROR_INVALID_RESPONSE:
                    st.error(f"Invalid AI response: {stg['ai_error']}")
                else:
                    st.error(f"AI call failed: {stg['ai_error']}")
            else:
                st.info(
                    "No AI verdict yet."
                    + (
                        " You can make an independent decision after reviewing the PDF."
                        if not review_disabled
                        else ""
                    )
                )

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
                    "Replacement PDF", type=["pdf"], key=f"replace_pdf_{paper['uid']}"
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
                        st.error(f"Could not replace the PDF: {exc}")
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
                    removed_uid = paper["uid"]
                    storage_key = (meta or {}).get("storage_key")
                    # Persist removal before deleting the stored PDF.
                    state.commit(lambda: state.remove_paper_with_cleanup(removed_uid, storage_key))
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
