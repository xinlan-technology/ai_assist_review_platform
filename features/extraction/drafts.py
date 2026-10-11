"""Unsaved review-form edits.

An unconfirmed paper keeps its edits as a draft whenever the reviewer leaves
its form. Confirmed answers never change on their own: edits made to them are
set aside until the reviewer saves or discards them.
"""
from __future__ import annotations

from copy import deepcopy

import streamlit as st

from core import ui
from features.extraction import schema, state as extraction
from features.workflow import state


def review_key(prefix: str, paper: dict, spec: dict, digest: str) -> str:
    """Scope widget answers to the paper, specification, PDF, and form revision."""
    revision = extraction.form_nonce(extraction.get(paper))
    return f"{prefix}:{paper['uid']}:{schema.spec_hash(spec)}:{digest}:{revision}"


def _keys(form: dict, spec: dict) -> dict:
    return form.get("keys") or {q["id"]: f"{form['review_key']}:{q['id']}" for q in spec["questions"]}


def _widget_answers(spec: dict, keys: dict) -> tuple[dict | None, list[str]]:
    """Read the form's widgets; None unless every question was drawn completely."""
    answers, checked = {}, []
    for question in spec["questions"]:
        qid = question["id"]
        base = keys.get(qid)
        kind = question["type"]
        main = {"open_text": "text", "single_choice": "single"}.get(kind, "multiple")
        # An interrupted render lacks the later widgets. The review box is drawn
        # last, so a missing one must not be read as an unticked one.
        if not base or any(f"{base}:{part}" not in st.session_state
                           for part in (main, "page", "quote", "checked")):
            return None, []
        if kind == "open_text":
            text = (st.session_state.get(f"{base}:text") or "").strip()
            values = [schema.NOT_REPORTED] if st.session_state.get(f"{base}:missing") else ([text] if text else [])
        elif kind == "single_choice":
            chosen = st.session_state.get(f"{base}:single")
            values = [chosen] if chosen else []
        else:
            values = list(st.session_state.get(f"{base}:multiple") or [])
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
    return answers, checked


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


def _stamp(record: dict) -> list:
    """Identify the saved review a form was drawn from."""
    return [record.get("review_state"), record.get("reviewed_at"), record.get("draft_saved_at")]


def remember(prefix: str, paper: dict, key: str, digest: str, pages: int | None, editable: bool,
             keys: dict, answers: dict, checked: list[str], sources: dict, proposals: dict,
             metadata: dict, selected: list[str]) -> None:
    """Record a completely drawn form, with its content as a fallback for a page switch."""
    st.session_state[f"{prefix}:open_form"] = {
        "uid": paper["uid"], "review_key": key, "digest": digest, "pages": pages,
        "editable": editable, "keys": dict(keys), "answers": deepcopy(answers),
        "checked": list(checked), "sources": dict(sources), "proposals": deepcopy(proposals),
        "metadata": deepcopy(metadata), "selected": list(selected),
        "stamp": _stamp(extraction.get(paper)),
    }


def _provenance(form: dict) -> dict:
    if "sources" not in form:
        return {}
    return {"source_run_ids": deepcopy(form["sources"]),
            "source_answers": deepcopy(form.get("proposals", {})),
            "source_metadata": deepcopy(form.get("metadata", {}))}


def _current(prefix: str, spec: dict) -> tuple[dict, dict, dict, list[str]] | None:
    """The open form with its paper and present content, when that differs from what is saved."""
    form = st.session_state.get(f"{prefix}:open_form")
    if not form or not form.get("editable"):
        return None
    paper = next((p for p in state.papers() if p["uid"] == form["uid"]), None)
    if paper is None:
        return None
    digest = form["digest"]
    if not digest or extraction.is_stale(paper, spec, digest):
        return None
    if form["review_key"] != review_key(prefix, paper, spec, digest):
        return None
    record = extraction.get(paper)
    if "stamp" in form and form["stamp"] != _stamp(record):
        return None  # The saved review is newer than this form.
    answers, checked = _widget_answers(spec, _keys(form, spec))
    if answers is None:
        # Widgets do not outlive a page switch; use the last content read from them.
        if "answers" not in form:
            return None
        answers, checked = deepcopy(form["answers"]), list(form.get("checked", []))
    else:
        form["answers"], form["checked"] = deepcopy(answers), list(checked)
    baseline = record.get("final_answers") or record.get("ai_answers") or {}
    # An untouched form is not an edit, but clearing an AI answer is one.
    changed = any(_comparable(answers[qid]) != _comparable(_form_shape(baseline.get(qid)))
                  for qid in answers)
    saved_sources = record.get("source_run_ids", {})
    changed = changed or any(saved_sources.get(qid) != form.get("sources", {}).get(qid)
                             for qid in form.get("selected", []))
    if record.get("review_state") != "confirmed":
        # Ticked review boxes are progress on a paper that is still being reviewed.
        changed = changed or sorted(checked) != sorted(record.get("checked_questions", []))
    return (form, paper, answers, checked) if changed else None


def autosave(prefix: str, spec: dict) -> None:
    """Keep an unconfirmed paper's edits as a draft before its form is dropped."""
    found = _current(prefix, spec)
    if not found:
        return
    form, paper, answers, checked = found
    if extraction.get(paper).get("review_state") == "confirmed":
        return  # Confirmed answers change only through an explicit choice.
    was_pending = state.has_unsaved_results()
    state.begin_result()
    before = deepcopy(paper)
    try:
        extraction.save_review(paper, spec, form["digest"], answers, form.get("pages"),
                               confirm=False, **_provenance(form))
        extraction.get(paper)["checked_questions"] = checked
    except ValueError:
        paper.clear()
        paper.update(before)
        if not was_pending:
            state.clear_unsaved_results()
        return
    if not state.save_result():
        st.rerun()
    st.session_state[f"{prefix}:autosaved"] = paper.get("title") or "(no title)"


def hold(prefix: str, spec: dict) -> None:
    """Set a confirmed paper's unsaved edits aside when the reviewer leaves its form."""
    key = f"{prefix}:held_edits"
    if st.session_state.get(key):
        return
    found = _current(prefix, spec)
    if not found:
        return
    form, paper, answers, checked = found
    if extraction.get(paper).get("review_state") != "confirmed":
        return
    st.session_state[key] = {
        "uid": paper["uid"], "review_key": form["review_key"], "digest": form["digest"],
        "pages": form.get("pages"), "answers": answers, "checked": checked,
        "provenance": _provenance(form),
    }
    st.session_state.pop(f"{prefix}:open_form", None)


def saved_spec() -> dict | None:
    """The saved question setup, or None when it cannot produce a review form."""
    config = state.config()
    try:
        return schema.build_spec(config.get("extraction_instructions", ""),
                                 config.get("extraction_questions", []))
    except ValueError:
        return None


def _release(prefix: str, held: dict) -> None:
    """Forget held edits and their widget values so the form shows the saved answers."""
    st.session_state.pop(f"{prefix}:held_edits", None)
    for key in list(st.session_state):
        if str(key).startswith(held["review_key"]):
            del st.session_state[key]


def resolve(prefix: str) -> None:
    """Stop the page until held edits to confirmed answers are saved or discarded."""
    held = st.session_state.get(f"{prefix}:held_edits")
    if not held:
        return
    spec = saved_spec()
    paper = next((p for p in state.papers() if p["uid"] == held["uid"]), None)
    if (spec is None or paper is None or extraction.is_stale(paper, spec, held["digest"])
            or held["review_key"] != review_key(prefix, paper, spec, held["digest"])
            or extraction.get(paper).get("review_state") != "confirmed"):
        _release(prefix, held)
        return
    st.warning(f"You changed the confirmed answers of “{ui.escape_markdown(paper.get('title') or '(no title)')}” "
               "and left the paper without saving. Confirmed answers are never changed "
               "automatically: choose what to do with your changes.")
    complete = len(held["checked"]) == len(spec["questions"])
    for question in spec["questions"]:
        try:
            schema.validate_answer(question, held["answers"].get(question["id"]), held["pages"])
        except ValueError:
            complete = False

    def save(confirm: bool) -> None:
        def change():
            extraction.save_review(paper, spec, held["digest"], held["answers"], held["pages"],
                                   confirm=confirm, **held["provenance"])
            extraction.get(paper)["checked_questions"] = list(held["checked"])
        state.commit(change)
        _release(prefix, held)
        st.rerun()

    keep, draft, discard = st.columns(3)
    if keep.button("Save and keep confirmed", key=f"{prefix}:held:confirm", width="stretch",
                   type="primary", disabled=not complete):
        save(True)
    if draft.button("Save as a draft", key=f"{prefix}:held:draft", width="stretch"):
        save(False)
    if discard.button("Discard my changes", key=f"{prefix}:held:discard", width="stretch"):
        _release(prefix, held)
        st.rerun()
    if not complete:
        st.caption("Keeping the paper confirmed needs a valid answer and a ticked review box "
                   "for every question; a draft has to be confirmed again.")
    st.stop()


def leave(prefix: str) -> None:
    """Another page is shown: keep a draft, or set edits to confirmed answers aside."""
    if f"{prefix}:open_form" not in st.session_state:
        return
    spec = saved_spec()
    if spec is not None:
        autosave(prefix, spec)
        hold(prefix, spec)
    st.session_state.pop(f"{prefix}:open_form", None)
