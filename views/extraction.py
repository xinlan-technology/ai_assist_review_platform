from __future__ import annotations

import contextlib
from copy import deepcopy
from uuid import uuid4

import streamlit as st

from core import auth, db, fulltext_storage, ui
from features.extraction import drafts, schema, state as extraction
from features.workflow import run_controls, runs, state


@st.cache_data(show_spinner=False, ttl=300, max_entries=6)
def _load_pdf(storage_key: str, digest: str) -> bytes:
    data = fulltext_storage.load_pdf(storage_key)
    if fulltext_storage.sha256(data) != digest:
        raise fulltext_storage.FulltextStorageError(
            "This PDF differs from its saved metadata. Reattach it in Full-text Screening."
        )
    return data


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
            with st.expander(
                f"Question {index + 1}: {ui.escape_markdown(question.get('text') or 'New question')}",
                expanded=True,
            ):
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
        intent = schema.spec_hash(proposed or draft)
        remembered = f"{prefix}:acknowledged"
        if changed and has_results:
            acknowledged = st.checkbox(
                "I understand that changing the setup makes existing extraction results outdated.",
                # A prompt can hide this box for a run; the acknowledgement of
                # this exact setup is remembered across it.
                value=st.session_state.get(remembered) == intent,
                key=f"{prefix}:ack:{intent}",
            )
        if remembered in st.session_state and (
                not acknowledged or st.session_state[remembered] != intent):
            # Unticking the box or editing the setup withdraws the acknowledgement.
            del st.session_state[remembered]
        if setup_error:
            st.info(ui.escape_markdown(setup_error))
        if st.button("Save extraction setup", type="primary", key=f"{prefix}:save_setup",
                     disabled=proposed is None or not changed or not acknowledged):
            active = drafts.saved_spec()
            if active is not None:
                # A new setup rebuilds the review form; unsaved edits to
                # confirmed answers are settled before that happens.
                drafts.hold(prefix, active)
                st.session_state[remembered] = intent
                drafts.resolve(prefix)

            def save_setup():
                config.update(extraction_instructions=proposed["instructions"],
                              extraction_questions=deepcopy(proposed["questions"]))
            state.commit(save_setup)
            st.session_state.pop(remembered, None)
            st.rerun()
        if changed and current:
            st.caption("The saved setup remains active until you save your changes.")
    return current


def _run(papers: list[dict], spec: dict, metadata: dict, models: list[dict], attempts: list[dict],
         *, include: tuple[str, ...] = (runs.NEW,), refresh: bool = False) -> None:
    if include == (runs.NEW,) and not refresh:
        papers = [paper for paper in papers if not extraction.get(paper).get("review_state")]
        if not papers:
            st.info("Nothing to run: those papers now hold your own draft or confirmed answers. "
                    "Use “Archive draft and start over” on a paper to let the AI answer it again.")
            return
    planned = runs.plan(papers, state.STAGE_EXTRACTION, spec, metadata, models, attempts, include=include)
    progress = st.progress(0.0, text="Preparing extraction…")
    with st.spinner("Extracting from original PDFs…"):
        finished = runs.execute(
            papers, state.STAGE_EXTRACTION, spec, metadata, models, include=include, refresh=refresh,
            progress=lambda done, total: progress.progress(done / total, text=f"Saved {done} / {total} calls"),
        )
    if finished:
        st.session_state[f"extraction:{state.active_id()}:run_notice"] = run_controls.finished(planned)
    if (finished or state.has_unsaved_results()
            or state.prepare_save().store.get("pending_ai_completion")):
        st.rerun()
    st.error("The extraction run stopped. Completed attempts remain in run history.")
    st.stop()


def _candidate_source(
    question: dict, record: dict, candidates: list[dict], spec: dict,
    digest: str, pages: int | None, key: str, disabled: bool,
) -> tuple[dict, str | None, dict | None, str]:
    """Compare immutable proposals; selection only initializes local widgets."""
    qid = question["id"]
    ai = record.get("ai_answers", {}).get(qid)
    current = record.get("final_answers", {}).get(qid, ai or {})
    current = current if isinstance(current, dict) else {}
    source_id = record.get("source_run_ids", {}).get(qid)
    if not source_id and ai:
        source_id = record.get("source_run_id")
    proposal = record.get("source_answers", {}).get(qid, ai)
    if not candidates:
        return current, source_id, proposal, ""
    choices, labels, rows = {}, {"": "Current answer"}, []
    for run in candidates:
        result = run.get("result") or {}
        run_id = run["id"]
        label = f"{run.get('provider', '')} · {run.get('model', '')} · {run_id}"
        answer, issue = {}, ""
        try:
            answer = extraction.candidate_answer(run, spec, digest, qid, pages)
            choices[run_id] = answer
            labels[run_id] = label
        except ValueError as exc:
            issue = result.get("field_errors", {}).get(qid) or result.get("ai_error") or str(exc)
        rows.append({
            "Run": label, "Answer": "; ".join(answer.get("values", [])),
            "Other": answer.get("other_text", ""), "PDF page": answer.get("page"),
            "Evidence": answer.get("quote", ""), "Status": run.get("status", ""),
            "Issue": issue,
        })
    with st.expander("Compare model answers", expanded=len(candidates) > 1):
        st.dataframe(rows, hide_index=True, width="stretch")
        if len({(tuple(sorted(a["values"])), a.get("other_text", "")) for a in choices.values()}) > 1:
            st.warning("Model answers differ. Check the evidence before choosing your final answer.")
    source = st.selectbox(
        "Use answer from", ["", *choices], format_func=labels.get,
        key=f"{key}:source", disabled=disabled or not choices,
        help="Choosing a run does not save or confirm it. Check its evidence, then explicitly save your review.",
    )
    if source:
        return deepcopy(choices[source]), source, deepcopy(choices[source]), source
    return current, source_id, proposal, ""


def _source_metadata(record: dict, candidates: list[dict], qid: str, source_id: str) -> dict:
    stored = record.get("source_metadata", {}).get(qid, {})
    if stored.get("id") == source_id:
        return deepcopy(stored)
    source = next((run for run in candidates if run["id"] == source_id), None)
    if source is None and record.get("source_run_id") == source_id:
        source = record
    if source is None:
        return {}
    return {"id": source_id, "provider": source.get("provider", ""),
            "model": source.get("model", ""), "completed_at": source.get("completed_at")}


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
            st.caption(f"Saved your in-progress answers for “{ui.escape_markdown(autosaved)}” as a draft.")
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
            drafts.autosave(prefix, spec)
            st.session_state[f"{jump_key}:moved"] = pos - 1
            state.goto(state.STAGE_EXTRACTION, pos - 1, len(papers))
            st.rerun()
        middle.caption(f"Paper {pos + 1} / {len(papers)}")
        if following.button("Next ›", disabled=pos == len(papers) - 1, key=f"{prefix}:next", width="stretch"):
            drafts.autosave(prefix, spec)
            st.session_state[f"{jump_key}:moved"] = pos + 1
            state.goto(state.STAGE_EXTRACTION, pos + 1, len(papers))
            st.rerun()
        paper = papers[pos]
        open_form = st.session_state.get(f"{prefix}:open_form")
        if open_form and open_form["uid"] != paper["uid"]:
            # Leaving a confirmed paper with unsaved edits needs the reviewer's decision.
            drafts.hold(prefix, spec)
            drafts.resolve(prefix)
        meta = metadata.get(paper["uid"], {})
        digest = meta.get("sha256", "")
        record = extraction.get(paper)
        stale = extraction.is_stale(paper, spec, digest)
        try:
            history = runs.load(state.STAGE_EXTRACTION, paper["uid"])
        except db.DatabaseError as exc:
            st.error(ui.escape_markdown(str(exc)))
            history = []
        candidates = [run for run in history
                      if runs.compatible(run, state.STAGE_EXTRACTION, spec, paper, metadata)]
        st.markdown(f"#### {ui.escape_markdown(paper.get('title') or '(no title)')}")
        st.caption(f"Status: {extraction.status(paper, spec, digest)}")
        if history:
            run_controls.history(state.STAGE_EXTRACTION, paper, spec, metadata, records=history)
            incompatible = [run for run in history if run not in candidates]
            if incompatible:
                with st.expander("Incompatible extraction history (read-only)"):
                    st.caption("These answers used a different question setup, prompt, or PDF and cannot be adopted.")
                    for old_run in incompatible:
                        st.caption(ui.escape_markdown(run_controls.label(old_run)))
                        st.json({"setup": old_run.get("prompt_snapshot"), "result": old_run.get("result")})
        left, right = st.columns([3, 2], gap="large")
        pdf = None
        with left:
            st.markdown("##### Original PDF")
            if meta.get("status") == "ok" and meta.get("storage_key") and digest:
                try:
                    pdf = _load_pdf(meta["storage_key"], digest)
                except fulltext_storage.FulltextStorageError as exc:
                    st.error(ui.escape_markdown(str(exc)))
            if pdf:
                st.caption(ui.escape_markdown(
                    f"{meta.get('filename') or 'paper.pdf'} · {meta.get('page_count') or '?'} pages"
                ))
                st.download_button("Download original PDF", pdf, file_name=meta.get("filename") or "paper.pdf",
                                   mime="application/pdf", key=f"{prefix}:download:{paper['uid']}")
                st.pdf(pdf, height=780)
            else:
                st.warning("Attach a usable PDF in Full-text Screening before reviewing extraction.")
        with right:
            if stale:
                st.warning("This extraction is outdated. Archive it above to start a current review. "
                           "New runs preserve your existing answers in the meantime.")
                return
            if record.get("review_state") == "draft":
                st.caption("Starting over archives this paper's current AI answers and human draft in history, "
                           "then resets review. You can run AI extraction afterward; archiving makes no AI call.")
                if st.button("Archive draft and start over", key=f"{prefix}:{paper['uid']}:archive_draft"):
                    state.commit(lambda: extraction.archive(paper))
                    st.rerun()
            if record.get("ai_error"):
                st.error(f"AI extraction: {ui.escape_markdown(record['ai_error'])}")
            elif not record.get("ai_answers"):
                st.info("No AI answers yet. You can extract the data manually from the PDF.")
            st.caption("AI answers and evidence are proposals. Check the PDF, edit as needed, and review every question.")
            if candidates:
                st.caption("Choose a source independently for each question.")
            if record.get("review_state") == "confirmed":
                st.caption("These answers are confirmed. Changes take effect only when you save them.")
            else:
                st.caption("Your edits are kept as a draft when you leave this paper.")
            review_key = drafts.review_key(prefix, paper, spec, digest)
            answers, checked, errors, source_run_ids, source_answers = {}, [], [], {}, {}
            source_metadata, input_keys, chosen_sources = {}, {}, []
            for index, question in enumerate(spec["questions"]):
                qid = question["id"]
                st.markdown(f"##### {index + 1}. {ui.escape_markdown(question['text'])}")
                if question["guidance"]:
                    st.caption(ui.escape_markdown(question["guidance"]))
                ai = record.get("ai_answers", {}).get(qid)
                if ai:
                    st.caption("AI proposed answer:")
                    st.text("; ".join(ai["values"]))
                    if ai.get("other_text"):
                        st.caption(ui.escape_markdown(ai["other_text"]))
                    if ai.get("page"):
                        st.caption(f"AI proposed evidence · PDF page {ai['page']}")
                        st.text(ai.get("quote", ""))
                if qid in record.get("field_errors", {}):
                    st.warning(ui.escape_markdown(record["field_errors"][qid]))
                    with st.expander("View invalid AI answer"):
                        st.json(record.get("invalid_answers", {}).get(qid))
                base, source_id, proposal, selected_source = _candidate_source(
                    question, record, candidates, spec, digest, meta.get("page_count"),
                    f"{review_key}:{qid}", not pdf,
                )
                if source_id:
                    source_run_ids[qid] = source_id
                    source_info = _source_metadata(record, candidates, qid, source_id)
                    if source_info:
                        source_metadata[qid] = source_info
                if proposal:
                    source_answers[qid] = proposal
                input_key = f"{review_key}:{qid}"
                if selected_source:
                    input_key += f":run:{selected_source}"
                    chosen_sources.append(qid)
                input_keys[qid] = input_key
                answer = _answer_input(question, base, input_key, meta.get("page_count"), not pdf)
                answers[qid] = answer
                try:
                    schema.validate_answer(question, answer, meta.get("page_count"))
                except ValueError as exc:
                    errors.append(str(exc))
                    st.caption(ui.escape_markdown(str(exc)))
                if st.checkbox("Reviewed this question", key=f"{input_key}:checked", disabled=not pdf,
                               value=not selected_source and (qid in record.get("checked_questions", [])
                                                              or record.get("review_state") == "confirmed")):
                    checked.append(qid)
                st.divider()
            # Recorded only once every question is drawn, so an interrupted
            # render can never be mistaken for the reviewer's edits.
            drafts.remember(prefix, paper, review_key, digest, meta.get("page_count"), bool(pdf),
                            input_keys, answers, checked, source_run_ids, source_answers,
                            source_metadata, chosen_sources)
            draft_col, confirm_col = st.columns(2)
            save_draft = draft_col.button("Save draft", disabled=not pdf, width="stretch", key=f"{review_key}:draft")
            confirm = confirm_col.button("Confirm extraction", type="primary", width="stretch",
                                         disabled=not pdf or bool(errors) or len(checked) != len(spec["questions"]),
                                         key=f"{review_key}:confirm")
            if save_draft or confirm:
                def save_review():
                    extraction.save_review(
                        paper, spec, digest, answers, meta.get("page_count"), confirm=confirm,
                        source_run_ids=source_run_ids, source_answers=source_answers,
                        source_metadata=source_metadata,
                    )
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
st.caption(f"Project: **{ui.escape_markdown(state.active_name())}**")
state.require_saved_results()

with st.sidebar:
    provider, model, api_key = ui.model_controls()
    models = ui.additional_models(provider, model, api_key)

try:
    metadata = db.load_fulltexts(auth.current_user(), project_id)
except db.DatabaseError as exc:
    st.error(ui.escape_markdown(str(exc)))
    st.stop()

drafts.resolve(prefix)
# An invalid saved setup has no review form to autosave.
with contextlib.suppress(ValueError):
    drafts.autosave(
        prefix,
        schema.build_spec(
            state.config().get("extraction_instructions", ""),
            state.config().get("extraction_questions", []),
        ),
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
    notice = st.session_state.pop(f"{prefix}:run_notice", None)
    if notice:
        st.success(notice)
    setup_issue = (schema.provider_issue(provider, spec)
                   if all(schema.provider_issue(m["provider"], spec) for m in models) else None)
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
            reasons = [runs.input_issue(paper, state.STAGE_EXTRACTION, spec, metadata, m) for m in models]
            reason = reasons[0] if all(reasons) else None
        if reason:
            blocked.append({"Paper": paper.get("title") or "(no title)", "Issue": reason})
        else:
            ready.append(paper)
    stale_uids = {p["uid"] for p in stale}
    pending = [p for p in ready if extraction.pending(p, spec, metadata[p["uid"]]["sha256"])]
    outdated_ready = [p for p in ready if p["uid"] in stale_uids]
    current = [p for p in ready if p["uid"] not in stale_uids]
    unreviewed = runs.awaiting_models(current, state.STAGE_EXTRACTION)
    confirmed = sum(bool(extraction.confirmed_answers(p, spec, metadata.get(p["uid"], {}).get("sha256", ""))) for p in papers)
    st.progress(confirmed / len(papers), text=f"Human confirmed {confirmed} / {len(papers)} papers")
    st.caption(f"{len(pending)} ready to run · {len(stale)} outdated · {len(blocked)} PDFs unavailable for AI")
    st.caption(f"All questions are sent with the original PDF to {len(models)} selected model(s). "
               "One request per paper and model; each result is saved immediately.")
    if blocked:
        with st.expander("PDFs requiring attention"):
            st.dataframe(blocked, hide_index=True, width="stretch")
    attempts = run_controls.attempts(state.STAGE_EXTRACTION)
    new_tasks = saved = refresh_tasks = refresh_saved = []
    known = None
    if attempts is not None:
        known = runs.outcomes(ready, state.STAGE_EXTRACTION, spec, metadata, attempts)
        try:
            new_tasks = runs.plan(unreviewed, state.STAGE_EXTRACTION, spec, metadata, models, attempts, known=known)
            refresh_tasks = runs.plan(outdated_ready, state.STAGE_EXTRACTION, spec, metadata, models, attempts,
                                      include=(runs.NEW, runs.FAILED, runs.INVALID), known=known)
        except ValueError as exc:
            st.warning(str(exc))
            attempts = None
        else:
            saved = runs.restorable(unreviewed, state.STAGE_EXTRACTION, spec, metadata, attempts, known=known)
            refresh_saved = runs.restorable(outdated_ready, state.STAGE_EXTRACTION, spec, metadata,
                                            attempts, refresh=True, known=known)
    can_run = bool(attempts is not None and not setup_issue and all(m.get("api_key") for m in models))
    if saved:
        st.caption(f"{len(saved)} paper(s) already have a saved AI answer; it is shown without a new call.")
    if st.button(f"▶ Run AI extraction ({runs.describe(new_tasks)})", type="primary", key=f"{prefix}:run",
                 disabled=not can_run or not (new_tasks or saved)):
        _run(unreviewed, spec, metadata, models, attempts)
    if attempts is not None:
        run_controls.unlinked_note(state.STAGE_EXTRACTION, current, spec, metadata, models, attempts)
        st.caption("Invalid fields can also be corrected by hand. A retry never replaces answers you have reviewed.")
        run_controls.retry_controls(
            state.STAGE_EXTRACTION, current, spec, metadata, models, attempts,
            lambda targets, include: _run(targets, spec, metadata, models, attempts, include=include),
            disabled=not can_run, known=known,
        )
    if stale:
        st.warning("Outdated answers do not count as complete. Re-running moves them, with your review, "
                   "to the paper's history and shows the new answers for you to review.")
        rerun, archive = st.columns(2)
        if rerun.button(f"↻ Re-run outdated PDFs ({runs.describe(refresh_tasks, len(outdated_ready))})",
                        key=f"{prefix}:rerun", width="stretch",
                        disabled=not can_run or not (refresh_tasks or refresh_saved)):
            _run(outdated_ready, spec, metadata, models, attempts,
                 include=(runs.NEW, runs.FAILED, runs.INVALID), refresh=True)
        if archive.button(f"Archive outdated without AI ({len(stale)})", key=f"{prefix}:archive", width="stretch"):
            state.commit(lambda: [extraction.archive(p) for p in stale])
            st.rerun()
        if attempts is not None:
            run_controls.retry_controls(
                state.STAGE_EXTRACTION, outdated_ready, spec, metadata, models, attempts,
                lambda targets, include: _run(targets, spec, metadata, models, attempts, include=include, refresh=True),
                disabled=not can_run, scope="outdated", categories=(runs.UNKNOWN,), known=known,
            )
run_controls.panel(state.STAGE_EXTRACTION, papers, spec, metadata, models=models)
_human_review(papers, spec, metadata, prefix)
