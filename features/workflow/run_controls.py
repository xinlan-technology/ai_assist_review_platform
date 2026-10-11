"""Shared model-comparison controls and immutable run exports."""
import json

import pandas as pd
import streamlit as st

from core import csv_io, db, ui
from features.workflow import runs, state


def recover_pending() -> None:
    prepared = state.prepare_save()
    pending = prepared.store.get("pending_ai_completion")
    if not pending:
        return
    st.error("An AI answer has not been saved to run history. Keep this session open. "
             "Retry saving it below; do not repeat the paid call.")
    if st.button("Save pending AI answer", key=f"runs:{prepared.pid}:recover"):
        try:
            completed = db.finish_ai_run(prepared.user_email, prepared.pid,
                                         verify_paper=False, **pending)
        except db.DatabaseError as exc:
            st.error(ui.escape_markdown(exc))
        else:
            paper = next((p for p in prepared.store["papers"] if p["uid"] == completed["paper_uid"]), None)
            if paper and runs.publish_initial(paper, completed["stage"], completed):
                prepared.store[state._UNSAVED_KEY] = True
            prepared.store.pop("pending_ai_completion", None)
            runs.forget_index()
            if state.has_unsaved_results():
                state.save_result()
            st.rerun()
    st.download_button("Download unsaved AI answer", json.dumps(pending, ensure_ascii=False, indent=2),
                       file_name="unsaved_ai_answer.json", mime="application/json", on_click="ignore")
    discard = st.checkbox("I have backed up this answer and want to discard its unsaved copy.",
                          key=f"runs:{prepared.pid}:discard-ok")
    if st.button("Discard pending AI answer", disabled=not discard,
                 key=f"runs:{prepared.pid}:discard"):
        prepared.store.pop("pending_ai_completion", None)
        prepared.store[state._UNSAVED_KEY] = False
        st.rerun()
    st.stop()


def label(run: dict) -> str:
    return f"{run['provider']} · {run['model']} · {run.get('started_at') or 'legacy'} · {run['id'][:8]}"


def history(stage: str, paper: dict, spec: dict, metadata: dict | None = None,
            records: list[dict] | None = None) -> None:
    records = runs.load(stage, paper["uid"]) if records is None else records
    if not records:
        return
    rows = [{
        "Run": r["id"], "Provider": r["provider"], "Model": r["model"],
        "Started": r.get("started_at"), "Status": r["status"],
        "Comparable": runs.compatible(r, stage, spec, paper, metadata),
        "Verdict": (r.get("result") or {}).get("ai_verdict", ""),
        "Reason / error": (r.get("result") or {}).get("ai_reason")
        or (r.get("result") or {}).get("ai_error", ""),
    } for r in records]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Only matching prompt, input and template versions are comparable. "
               "Running means the outcome is unknown; it is never retried automatically.")


def _export(stage: str, as_csv: bool):
    """Build an export only when its download is requested."""
    def build():
        exported = [{k: v for k, v in r.items() if k not in {"user_email", "project_id"}}
                    for r in runs.load(stage)]
        if not as_csv:
            return json.dumps(exported, ensure_ascii=False, indent=2)
        flat = [{k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                 for k, v in r.items()} for r in exported]
        return csv_io.to_csv_bytes(pd.DataFrame(flat))
    return build


def _papers(tasks: list[tuple[dict, dict]]) -> list[dict]:
    return list({paper["uid"]: paper for paper, _ in tasks}.values())


def retry_controls(stage: str, papers: list[dict], spec: dict, metadata: dict | None,
                   models: list[dict], records: list[dict], start, *, disabled: bool = False,
                   scope: str = "current", known: tuple | None = None,
                   categories: tuple[str, ...] = (runs.FAILED, runs.INVALID, runs.UNKNOWN)) -> None:
    """Targeted retries for attempts that ended without a usable answer.

    ``start(papers, include)`` runs the page's batch; a pair that already
    succeeded is never retried here.
    """
    prefix = f"runs:{stage}:{state.active_id()}:retry:{scope}"
    labels = {runs.FAILED: "Retry failed calls", runs.INVALID: "Retry invalid responses"}
    for category in categories:
        if category not in labels:
            continue
        tasks = runs.plan(papers, stage, spec, metadata, models, records, include=(category,),
                          known=known)
        if tasks and st.button(
                f"↻ {labels[category]} ({runs.describe(tasks)}, may be billed again)",
                key=f"{prefix}:{category}", disabled=disabled):
            start(_papers(tasks), (category,))
    if runs.UNKNOWN not in categories:
        return
    unknown = runs.plan(papers, stage, spec, metadata, models, records, include=(runs.UNKNOWN,),
                        known=known)
    if unknown:
        st.caption(f"{runs.describe(unknown)} ended without a recorded outcome. "
                   "The provider may already have charged for them.")
        accepted = st.checkbox("Repeat them anyway; I accept a possible second charge.",
                               key=f"{prefix}:unknown-ok")
        if st.button(f"↻ Repeat attempts with unknown outcome ({runs.describe(unknown)})",
                     key=f"{prefix}:unknown", disabled=disabled or not accepted):
            start(_papers(unknown), (runs.UNKNOWN,))


def unlinked_note(stage: str, papers: list[dict], spec: dict, metadata: dict | None,
                  models: list[dict], records: list[dict]) -> None:
    """Say which calls the main run leaves to the comparison panel, and why."""
    left = runs.plan(runs.unlinked_answers(papers, stage), stage, spec, metadata, models, records)
    if left:
        st.caption(f"{runs.describe(left)} are not started by this button: those papers show an AI "
                   "answer recorded before run history was kept. Add models to them in "
                   "Model comparison and run history.")


def attempts(stage: str) -> list[dict] | None:
    """Attempt identities for the page's counters, or None after showing the error."""
    try:
        return runs.index(stage)
    except db.DatabaseError as exc:
        st.error(ui.escape_markdown(exc))
        return None


def finished(tasks: list[tuple[dict, dict]]) -> str:
    if tasks:
        return (f"Run finished: {runs.describe(tasks)}. Review the answers below; "
                "failed or invalid attempts are offered for retry here.")
    return "Saved AI answers are shown below; no new call was needed."


def panel(stage: str, papers: list[dict], spec: dict, metadata: dict | None = None,
          models: list[dict] | None = None) -> None:
    recover_pending()
    prefix = f"runs:{stage}:{state.active_id()}"
    # Nothing below is loaded or computed until the reviewer asks for it.
    if not st.toggle("Model comparison and run history", key=f"{prefix}:open"):
        return
    with st.container(border=True):
        try:
            records = runs.index(stage)
        except db.DatabaseError as exc:
            st.error(ui.escape_markdown(exc))
            return
        lookup = {p["uid"]: p for p in papers}
        if st.checkbox("All papers in this stage", value=True, key=f"{prefix}:all"):
            selected = list(lookup)
        else:
            selected = st.multiselect(
                "Papers to compare", list(lookup),
                format_func=lambda uid: lookup[uid].get("title") or uid,
                key=f"{prefix}:papers",
            )
        repeat = st.checkbox("Repeat selected models", key=f"{prefix}:repeat",
                             help="Create a new paid attempt even when this model already ran.")
        tasks, issue = [], ""
        if models:
            try:
                tasks = runs.plan([lookup[uid] for uid in selected], stage, spec, metadata,
                                  models, records, repeat=repeat)
            except ValueError as exc:
                issue = str(exc)
                st.warning(issue)
        destinations = ", ".join(dict.fromkeys(m["provider"] for m in models or []))
        st.caption(f"{len(tasks)} separate API calls. Input is sent to: {destinations or 'no provider'}. "
                   "Full-text and extraction calls send the original PDF. Each attempt is saved "
                   "separately; existing human decisions remain unchanged. PDF limits apply per model.")
        blocked = [{"Paper": lookup[uid].get("title") or uid,
                    "Model": f"{m['provider']} · {m['model']}", "Issue": reason}
                   for uid in selected for m in models or []
                   if (reason := runs.input_issue(lookup[uid], stage, spec, metadata, m))]
        if blocked:
            st.caption("These paper/model pairs will not be sent:")
            st.dataframe(blocked, hide_index=True, width="stretch")
        if repeat:
            st.warning("Repeating creates additional charges, including for attempts whose outcome is unknown.")
        ready = bool(spec.get("questions") if stage == state.STAGE_EXTRACTION else spec.get("criteria", "").strip())
        if st.button(f"Run model comparison ({len(tasks)} calls)", key=f"{prefix}:run",
                     disabled=not ready or not tasks or bool(issue)
                     or any(not m.get("api_key", "").strip() for m in models or [])):
            progress = st.progress(0.0)
            if runs.execute([lookup[uid] for uid in selected], stage, spec, metadata, models,
                            repeat=repeat, progress=lambda n, total: progress.progress(n / total)):
                st.rerun()
            if (state.prepare_save().store.get("pending_ai_completion")
                    or state.has_unsaved_results()):
                st.rerun()
            progress.empty()
        if records:
            st.caption(f"{len(records)} saved attempts in this stage. Failed and unknown outcomes are included.")
            st.download_button("Download all AI runs (JSON)", _export(stage, as_csv=False),
                               file_name=f"{stage}_ai_runs.json", mime="application/json",
                               key=f"{prefix}:export")
            st.download_button("Download all AI runs (CSV)", _export(stage, as_csv=True),
                               file_name=f"{stage}_ai_runs.csv", mime="text/csv",
                               key=f"{prefix}:csv")


def screening_review(paper: dict, stage: str, spec: dict, metadata: dict | None = None,
                     *, disabled: bool = False) -> None:
    try:
        records = runs.load(stage, paper["uid"])
    except db.DatabaseError as exc:
        st.error(ui.escape_markdown(exc))
        return
    if not records:
        return
    with st.expander("Compare AI answers and confirm one final verdict", expanded=len(records) > 1):
        history(stage, paper, spec, metadata, records)
        candidates = {r["id"]: r for r in records if r["status"] == "succeeded"
                      and runs.compatible(r, stage, spec, paper, metadata)}
        if not candidates:
            st.info("No successful result matches the current input. Run a model or review manually.")
            return
        if len({r["result"]["ai_verdict"] for r in candidates.values()}) > 1:
            st.warning("Models disagree. Review their reasons before confirming your final verdict.")
        prefix = f"runs:review:{stage}:{state.active_id()}:{paper['uid']}"
        source_ids = list(candidates)
        current_id = state.stage_state(paper, stage).get("source_run_id")
        chosen = st.selectbox("Reference AI result", source_ids,
                              index=source_ids.index(current_id) if current_id in candidates else 0,
                              format_func=lambda rid: label(candidates[rid]), key=f"{prefix}:source")
        reference = candidates[chosen]
        st.text(reference["result"].get("ai_reason") or "")
        verdicts = state.STAGE_VERDICTS[stage]
        saved = state.current_final_verdict(paper, stage, reference["config_hash"])
        default = saved if chosen == current_id and saved else reference["result"]["ai_verdict"]
        verdict = st.selectbox("Final verdict", verdicts,
                               index=verdicts.index(default),
                               format_func=lambda v: state.VERDICT_LABELS[v],
                               key=f"{prefix}:verdict:{chosen}:{reference['config_hash']}")
        if st.button("Confirm final verdict", key=f"{prefix}:confirm", disabled=disabled):
            state.commit(lambda: runs.confirm_screening(paper, stage, spec, reference, verdict, metadata))
            st.rerun()
        st.caption("Selecting or comparing an answer does not change your decision. "
                   "Only Confirm final verdict replaces it.")
