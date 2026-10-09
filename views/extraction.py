from __future__ import annotations

import contextlib
from copy import deepcopy
from uuid import uuid4

import streamlit as st

from core import auth, db, fulltext_storage, ui
from core.llm import pdf_input_issue
from features.extraction import schema, service, state as extraction
from features.workflow import state


@st.cache_data(show_spinner=False, ttl=300, max_entries=6)
def _load_pdf(storage_key: str, digest: str) -> bytes:
    data = fulltext_storage.load_pdf(storage_key)
    if fulltext_storage.sha256(data) != digest:
        raise fulltext_storage.FulltextStorageError(
            "This PDF differs from its saved metadata. Reattach it in Full-text Screening."
        )
    return data


def _review_key(prefix: str, paper: dict, spec: dict, digest: str) -> str:
    """Scope widget answers to the paper, specification, PDF, and form revision."""
    revision = extraction.form_nonce(extraction.get(paper))
    return f"{prefix}:{paper['uid']}:{schema.spec_hash(spec)}:{digest}:{revision}"


def _widget_answers(spec: dict, review_key: str) -> tuple[dict | None, list[str]]:
    """Read widget answers; None distinguishes an unrendered form from a cleared one."""
    answers, checked, rendered = {}, [], False
    for question in spec["questions"]:
        qid = question["id"]
        base = f"{review_key}:{qid}"
        kind = question["type"]
        if kind == "open_text":
            if f"{base}:text" not in st.session_state:
                continue
            text = (st.session_state.get(f"{base}:text") or "").strip()
            values = [schema.NOT_REPORTED] if st.session_state.get(f"{base}:missing") else ([text] if text else [])
        elif kind == "single_choice":
            if f"{base}:single" not in st.session_state:
                continue
            chosen = st.session_state.get(f"{base}:single")
            values = [chosen] if chosen else []
        else:
            if f"{base}:multiple" not in st.session_state:
                continue
            values = list(st.session_state.get(f"{base}:multiple") or [])
        rendered = True
        missing = values == [schema.NOT_REPORTED]
        uses_other = kind != "open_text" and schema.OTHER in values
        answers[qid] = {
            "values": values,
            "other_text": (st.session_state.get(f"{base}:other") or "") if uses_other else "",
            "page": None if missing else st.session_state.get(f"{base}:page"),
            "quote": "" if missing else (st.session_state.get(f"{base}:quote") or ""),
            "issue": "",
        }
        if st.session_state.get(f"{base}:checked"):
            checked.append(qid)
    return (answers if rendered else None), checked


def _form_shape(answer: object) -> dict:
    """Normalize to the rendered form so display defaults do not count as edits."""
    if not isinstance(answer, dict):
        return {}
    values = list(answer.get("values") or [])
    missing = values == [schema.NOT_REPORTED]
    return {
        "values": values,
        "other_text": (answer.get("other_text") or "") if schema.OTHER in values else "",
        "page": None if missing else answer.get("page"),
        "quote": "" if missing else (answer.get("quote") or ""),
    }


def _comparable(answer: object) -> tuple:
    if not isinstance(answer, dict):
        return ((), "", None, "")
    return (
        tuple(answer.get("values") or []),
        answer.get("other_text") or "",
        answer.get("page"),
        (answer.get("quote") or "").strip(),
    )


def _autosave_open_paper(prefix: str, spec: dict, metadata: dict) -> None:
    """Save draft answers before Streamlit drops widgets omitted by a rerun."""
    open_form = st.session_state.get(f"{prefix}:open_form")
    if not open_form:
        return
    paper = next((p for p in state.papers() if p["uid"] == open_form["uid"]), None)
    if paper is None:
        return
    digest = open_form["digest"]
    if not open_form.get("editable"):
        return
    if not digest or extraction.is_stale(paper, spec, digest):
        return
    if open_form["review_key"] != _review_key(prefix, paper, spec, digest):
        return
    answers, checked = _widget_answers(spec, open_form["review_key"])
    if answers is None:
        return
    record = extraction.get(paper)
    if record.get("review_state") == "confirmed":
        return  # Confirmed answers require an explicit save.
    blank = all(_comparable(answer) == ((), "", None, "") for answer in answers.values())
    if blank and not record.get("final_answers"):
        return
    baseline = record.get("final_answers") or record.get("ai_answers") or {}
    # Checkbox-only changes do not trigger a project write.
    if all(_comparable(answers[qid]) == _comparable(_form_shape(baseline.get(qid)))
           for qid in answers):
        return
    try:
        extraction.save_review(paper, spec, digest, answers, open_form.get("pages"), confirm=False)
        extraction.get(paper)["checked_questions"] = checked
    except ValueError:
        return
    if not state.save_result():
        st.rerun()
    st.session_state[f"{prefix}:autosaved"] = paper.get("title") or "(no title)"


def _question_builder(prefix: str) -> dict | None:
    config = state.config()
    draft_key = f"{prefix}:setup_draft"
    if draft_key not in st.session_state:
        st.session_state[draft_key] = deepcopy({
            "instructions": config.get("extraction_instructions", ""),
            "questions": config.get("extraction_questions", []),
        })
    draft = st.session_state[draft_key]
    with st.container(border=True):
        st.subheader("Extraction setup")
        st.caption("Define what to extract. Changes take effect only after you save this setup.")
        draft["instructions"] = st.text_area(
            "Extraction instructions", value=draft["instructions"], height=130,
            placeholder="Extract findings of this study, not studies cited in its introduction.",
            key=f"{prefix}:instructions",
        )
        for index, question in enumerate(draft["questions"]):
            key = f"{prefix}:question:{question['id']}"
            with st.expander(f"Question {index + 1}: {question.get('text') or 'New question'}", expanded=True):
                question["text"] = st.text_input("Question", value=question.get("text", ""), key=f"{key}:text")
                question["type"] = st.selectbox(
                    "Answer type", list(schema.TYPES),
                    index=list(schema.TYPES).index(question.get("type", "single_choice")),
                    format_func=schema.TYPES.get, key=f"{key}:type",
                )
                if question["type"] != "open_text":
                    options = [option for option in question.get("options", [])
                               if option not in (schema.OTHER, schema.NOT_REPORTED)]
                    text = st.text_area("Options (one per line)", value="\n".join(options),
                                        height=100, key=f"{key}:options")
                    question["options"] = [line.strip() for line in text.splitlines() if line.strip()]
                    st.caption("Other and Not reported are added automatically.")
                else:
                    question["options"] = []
                question["guidance"] = st.text_input(
                    "Question guidance (optional)", value=question.get("guidance", ""), key=f"{key}:guidance",
                )
                up, down, remove = st.columns(3)
                if up.button("Move up", disabled=index == 0, key=f"{key}:up", width="stretch"):
                    draft["questions"][index - 1:index + 1] = [question, draft["questions"][index - 1]]
                    st.rerun()
                if down.button("Move down", disabled=index == len(draft["questions"]) - 1,
                               key=f"{key}:down", width="stretch"):
                    draft["questions"][index:index + 2] = [draft["questions"][index + 1], question]
                    st.rerun()
                if remove.button("Remove", key=f"{key}:remove", width="stretch"):
                    draft["questions"].pop(index)
                    st.rerun()
        if st.button("+ Add question", disabled=len(draft["questions"]) >= 30, key=f"{prefix}:add"):
            draft["questions"].append({
                "id": uuid4().hex, "text": "", "type": "single_choice", "options": [], "guidance": "",
            })
            st.rerun()
        st.caption(f"{len(draft['questions'])} / 30 questions · All questions require a final answer.")
        try:
            proposed = schema.build_spec(draft["instructions"], draft["questions"])
            setup_error = None
        except ValueError as exc:
            proposed, setup_error = None, str(exc)
        try:
            current = schema.build_spec(config.get("extraction_instructions", ""), config.get("extraction_questions", []))
        except ValueError:
            current = None
        changed = proposed != current
        has_results = any(any(k != "history" for k in extraction.get(paper)) for paper in state.papers())
        acknowledged = True
        if changed and has_results:
            acknowledged = st.checkbox(
                "I understand that changing the setup makes existing extraction results outdated.",
                key=f"{prefix}:ack:{schema.spec_hash(proposed or draft)}",
            )
        if setup_error:
            st.info(setup_error)
        if st.button("Save extraction setup", type="primary", key=f"{prefix}:save_setup",
                     disabled=proposed is None or not changed or not acknowledged):
            def save_setup():
                config.update(extraction_instructions=proposed["instructions"],
                              extraction_questions=deepcopy(proposed["questions"]))
            state.commit(save_setup)
            st.rerun()
        if changed and current:
            st.caption("The saved setup remains active until you save your changes.")
    return current


def _run(papers: list[dict], spec: dict, metadata: dict, provider: str,
         model: str, api_key: str, archive: bool = False) -> None:
    if archive:
        state.commit(lambda: [extraction.archive(paper) for paper in papers])
    else:
        papers = [paper for paper in papers if not extraction.get(paper).get("review_state")]
        if not papers:
            st.info("Nothing to run: those papers now hold your own draft or confirmed answers. "
                    "Use “Archive draft and start over” on a paper to let the AI answer it again.")
            return
    progress = st.progress(0.0, text="Preparing extraction…")
    checkpoints = 0

    def save_checkpoint():
        nonlocal checkpoints
        checkpoints += 1
        return state.save_active() if checkpoints == 1 else state.save_result()

    with st.spinner("Extracting from original PDFs…"):
        finished = service.run(
            papers, spec, metadata, provider, model, api_key, save_checkpoint,
            progress=lambda done, total: progress.progress(done / total, text=f"Saved {done} / {total} papers"),
        )
    if finished or st.session_state.get(state.unsaved_results_key()):
        st.rerun()
    st.error("Extraction did not start because the project could not be saved.")
    st.stop()


def _answer_input(question: dict, base: dict, key: str, pages: int | None, disabled: bool) -> dict:
    values = base.get("values") if isinstance(base.get("values"), list) else []
    kind = question["type"]
    if kind == "open_text":
        missing = st.checkbox("Not reported", value=values == [schema.NOT_REPORTED],
                              key=f"{key}:missing", disabled=disabled)
        text = st.text_area("Your answer", value=values[0] if values and values != [schema.NOT_REPORTED] else "", height=90,
                            disabled=disabled or missing, key=f"{key}:text")
        selected = [schema.NOT_REPORTED] if missing else [text.strip()] if text.strip() else []
    elif kind == "single_choice":
        options = question["options"]
        selected_value = st.selectbox(
            "Your answer", options, index=options.index(values[0]) if len(values) == 1 and values[0] in options else None,
            placeholder="Select an answer", key=f"{key}:single", disabled=disabled,
        )
        selected = [selected_value] if selected_value else []
    else:
        selected = st.multiselect("Your answers", question["options"],
                                  default=[v for v in values if v in question["options"]],
                                  key=f"{key}:multiple", disabled=disabled)
        if schema.NOT_REPORTED in selected and len(selected) > 1:
            st.warning("Not reported must be selected alone.")
    other = ""
    if kind != "open_text" and schema.OTHER in selected:
        other = st.text_input("Explain Other", value=base.get("other_text", ""),
                              key=f"{key}:other", disabled=disabled)
    missing = selected == [schema.NOT_REPORTED]
    page = base.get("page")
    if type(page) is not int or page < 1 or (pages and page > pages):
        page = None
    page = st.number_input("Evidence PDF page", min_value=1, max_value=pages,
                           value=page, step=1, placeholder="Page number", key=f"{key}:page",
                           disabled=disabled or missing)
    quote = st.text_area("Supporting quote", value=base.get("quote", ""), height=90,
                         key=f"{key}:quote", disabled=disabled or missing)
    return {"values": selected, "other_text": other, "page": None if missing else page,
            "quote": "" if missing else quote, "issue": ""}


def _human_review(papers: list[dict], spec: dict, metadata: dict, prefix: str) -> None:
    with st.container(border=True):
        st.subheader("Human review")
        autosaved = st.session_state.pop(f"{prefix}:autosaved", None)
        if autosaved:
            st.caption(f"Saved your in-progress answers for “{autosaved}” as a draft.")
        pos = max(0, min(state.cursor(state.STAGE_EXTRACTION), len(papers) - 1))
        prev, middle, following = st.columns([1, 2, 1])
        jump_key = f"{prefix}:jump"
        pos = ui.jump_to_paper(
            jump_key, pos,
            [f"{n + 1}. {(p.get('title') or '(no title)')[:70]} — "
             f"{extraction.status(p, spec, metadata.get(p['uid'], {}).get('sha256', ''))}"
             for n, p in enumerate(papers)],
        )
        state.goto(state.STAGE_EXTRACTION, pos, len(papers))

        if prev.button("‹ Prev", disabled=pos == 0, key=f"{prefix}:prev", width="stretch"):
            _autosave_open_paper(prefix, spec, metadata)
            st.session_state[f"{jump_key}:moved"] = pos - 1
            state.goto(state.STAGE_EXTRACTION, pos - 1, len(papers))
            st.rerun()
        middle.caption(f"Paper {pos + 1} / {len(papers)}")
        if following.button("Next ›", disabled=pos == len(papers) - 1, key=f"{prefix}:next", width="stretch"):
            _autosave_open_paper(prefix, spec, metadata)
            st.session_state[f"{jump_key}:moved"] = pos + 1
            state.goto(state.STAGE_EXTRACTION, pos + 1, len(papers))
            st.rerun()
        paper = papers[pos]
        meta = metadata.get(paper["uid"], {})
        digest = meta.get("sha256", "")
        record = extraction.get(paper)
        stale = extraction.is_stale(paper, spec, digest)
        st.markdown(f"#### {paper.get('title') or '(no title)'}")
        st.caption(f"Status: {extraction.status(paper, spec, digest)}")
        left, right = st.columns([3, 2], gap="large")
        pdf = None
        with left:
            st.markdown("##### Original PDF")
            if meta.get("status") == "ok" and meta.get("storage_key") and digest:
                try:
                    pdf = _load_pdf(meta["storage_key"], digest)
                except fulltext_storage.FulltextStorageError as exc:
                    st.error(str(exc))
            if pdf:
                st.caption(f"{meta.get('filename') or 'paper.pdf'} · {meta.get('page_count') or '?'} pages")
                st.download_button("Download original PDF", pdf, file_name=meta.get("filename") or "paper.pdf",
                                   mime="application/pdf", key=f"{prefix}:download:{paper['uid']}")
                st.pdf(pdf, height=780)
            else:
                st.warning("Attach a usable PDF in Full-text Screening before reviewing extraction.")
        with right:
            if stale:
                st.warning("This extraction is outdated. Re-run or archive it above before reviewing.")
                return
            if record.get("review_state") == "draft":
                st.caption("Starting over archives this paper's current AI answers and human draft in history, "
                           "then resets review. You can run AI extraction afterward; archiving makes no AI call.")
                if st.button("Archive draft and start over", key=f"{prefix}:{paper['uid']}:archive_draft"):
                    state.commit(lambda: extraction.archive(paper))
                    st.rerun()
            if record.get("ai_error"):
                st.error(f"AI extraction: {record['ai_error']}")
            elif not record.get("ai_answers"):
                st.info("No AI answers yet. You can extract the data manually from the PDF.")
            st.caption("AI answers and evidence are proposals. Check the PDF, edit as needed, and review every question.")
            review_key = _review_key(prefix, paper, spec, digest)
            st.session_state[f"{prefix}:open_form"] = {
                "uid": paper["uid"], "review_key": review_key,
                "digest": digest, "pages": meta.get("page_count"),
                "editable": bool(pdf),
            }
            answers, checked, errors = {}, [], []
            for index, question in enumerate(spec["questions"]):
                qid = question["id"]
                st.markdown(f"##### {index + 1}. {question['text']}")
                if question["guidance"]:
                    st.caption(question["guidance"])
                ai = record.get("ai_answers", {}).get(qid)
                if ai:
                    st.write("AI proposed answer:", "; ".join(ai["values"]))
                    if ai.get("other_text"):
                        st.caption(ai["other_text"])
                    if ai.get("page"):
                        st.caption(f"AI proposed evidence · PDF page {ai['page']}")
                        st.text(ai.get("quote", ""))
                if qid in record.get("field_errors", {}):
                    st.warning(record["field_errors"][qid])
                    with st.expander("View invalid AI answer"):
                        st.json(record.get("invalid_answers", {}).get(qid))
                base = record.get("final_answers", {}).get(qid, ai or {})
                if not isinstance(base, dict):
                    base = {}
                answer = _answer_input(question, base, f"{review_key}:{qid}", meta.get("page_count"), not pdf)
                answers[qid] = answer
                try:
                    schema.validate_answer(question, answer, meta.get("page_count"))
                except ValueError as exc:
                    errors.append(str(exc))
                    st.caption(str(exc))
                if st.checkbox("Reviewed this question", key=f"{review_key}:{qid}:checked", disabled=not pdf,
                               value=qid in record.get("checked_questions", []) or record.get("review_state") == "confirmed"):
                    checked.append(qid)
                st.divider()
            draft_col, confirm_col = st.columns(2)
            save_draft = draft_col.button("Save draft", disabled=not pdf, width="stretch", key=f"{review_key}:draft")
            confirm = confirm_col.button("Confirm extraction", type="primary", width="stretch",
                                         disabled=not pdf or bool(errors) or len(checked) != len(spec["questions"]),
                                         key=f"{review_key}:confirm")
            if save_draft or confirm:
                def save_review():
                    extraction.save_review(paper, spec, digest, answers, meta.get("page_count"), confirm=confirm)
                    extraction.get(paper)["checked_questions"] = checked
                state.commit(save_review)
                if confirm:
                    state.goto(state.STAGE_EXTRACTION, pos + 1, len(papers))
                st.rerun()


ui.inject_base_css()
ui.app_header("Data Extraction", "Define your questions, extract from original PDFs, then confirm each paper.")
auth.sidebar_user()
if not state.active_id():
    st.info("Open or create a project on the **My Projects** page first.")
    st.stop()
project_id = state.active_id()
prefix = f"extraction:{project_id}"
st.caption(f"Project: **{state.active_name()}**")
state.require_saved_results()

with st.sidebar:
    provider, model, api_key = ui.model_controls()

try:
    metadata = db.load_fulltexts(auth.current_user(), project_id)
except db.DatabaseError as exc:
    st.error(str(exc))
    st.stop()

# An invalid saved setup has no review form to autosave.
with contextlib.suppress(ValueError):
    _autosave_open_paper(
        prefix,
        schema.build_spec(
            state.config().get("extraction_instructions", ""),
            state.config().get("extraction_questions", []),
        ),
        metadata,
    )

spec = _question_builder(prefix)
if spec is None:
    st.info("Save at least one extraction question to continue.")
    st.stop()
papers = [paper for _, paper in state.stage_papers(state.STAGE_EXTRACTION)]
if not papers:
    st.info("No papers qualify yet. Confirm Include in Full-text Screening first.")
    st.stop()

with st.container(border=True):
    st.subheader("AI extraction")
    setup_issue = schema.provider_issue(provider, spec)
    if setup_issue:
        st.warning(setup_issue + " Manual review remains available.")
    stale = [p for p in papers if extraction.is_stale(p, spec, metadata.get(p["uid"], {}).get("sha256", ""))]
    ready, blocked = [], []
    for paper in papers:
        meta = metadata.get(paper["uid"], {})
        reason = None
        if not meta.get("storage_key") or not meta.get("sha256") or meta.get("status") != "ok":
            reason = "Attach a usable PDF in Full-text Screening."
        else:
            reason = pdf_input_issue(provider, model, meta.get("file_size"), meta.get("page_count"))
        if reason:
            blocked.append({"Paper": paper.get("title") or "(no title)", "Issue": reason})
        else:
            ready.append(paper)
    pending = [p for p in ready if extraction.pending(p, spec, metadata[p["uid"]]["sha256"])]
    invalid = [p for p in ready if extraction.status(p, spec, metadata[p["uid"]]["sha256"]) == "Invalid"]
    outdated_ready = [p for p in ready if p in stale]
    confirmed = sum(bool(extraction.confirmed_answers(p, spec, metadata.get(p["uid"], {}).get("sha256", ""))) for p in papers)
    st.progress(confirmed / len(papers), text=f"Human confirmed {confirmed} / {len(papers)} papers")
    st.caption(f"{len(pending)} ready to run · {len(stale)} outdated · {len(blocked)} PDFs unavailable for AI")
    st.caption(f"All questions are sent with the original PDF to {provider} · {model}. One request per paper; each result is saved immediately.")
    if blocked:
        with st.expander("PDFs requiring attention"):
            st.dataframe(blocked, hide_index=True, width="stretch")
    if st.button(f"▶ Run AI extraction ({len(pending)})", type="primary", key=f"{prefix}:run",
                 disabled=not api_key or not pending or bool(setup_issue)):
        _run(pending, spec, metadata, provider, model, api_key)
    if invalid:
        st.caption("Invalid fields can be corrected manually. Retrying archives the current AI answers.")
        if st.button(f"↻ Retry invalid responses ({len(invalid)})", disabled=not api_key or bool(setup_issue), key=f"{prefix}:retry"):
            _run(invalid, spec, metadata, provider, model, api_key, archive=True)
    if stale:
        st.warning("Outdated answers do not count as complete. Re-running or archiving preserves their history and resets their review.")
        rerun, archive = st.columns(2)
        if rerun.button(f"↻ Re-run outdated PDFs ({len(outdated_ready)})", key=f"{prefix}:rerun", width="stretch",
                        disabled=not api_key or not outdated_ready or bool(setup_issue)):
            _run(outdated_ready, spec, metadata, provider, model, api_key, archive=True)
        if archive.button(f"Archive outdated without AI ({len(stale)})", key=f"{prefix}:archive", width="stretch"):
            state.commit(lambda: [extraction.archive(p) for p in stale])
            st.rerun()

_human_review(papers, spec, metadata, prefix)
