"""Durable AI attempts, independent of the human-approved workflow state."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from time import perf_counter
import uuid

import streamlit as st

from core import auth, db, fulltext_storage, ui
from core.llm import InvalidModelResponse, PROVIDERS, pdf_input_issue
from features.extraction import judge as extraction_judge, schema, state as extraction
from features.screening import judge
from features.workflow import state


# What a (paper, model) pair needs, judged from its attempts on the current input.
NEW = "new"            # never attempted
FAILED = "failed"      # latest attempt failed and none succeeded
INVALID = "invalid"    # latest attempt was unusable and none succeeded
UNKNOWN = "unknown"    # latest attempt never reported an outcome
REPEAT = "repeat"      # an attempt already succeeded
EVERY_ATTEMPT = (NEW, FAILED, INVALID, UNKNOWN, REPEAT)
_UNFINISHED = {"call_failed": FAILED, "invalid_response": INVALID, "running": UNKNOWN}
_INDEX_KEY = "runs:index:"


def source_hash(stage: str, paper: dict, metadata: dict | None = None) -> str:
    if stage == state.STAGE_ABSTRACT:
        content = [paper.get("title", ""), paper.get("abstract", "")]
        return hashlib.sha256(json.dumps(content, ensure_ascii=False).encode()).hexdigest()
    return ((metadata or {}).get(paper["uid"]) or {}).get("sha256") or ""


def config_hash(stage: str, spec: dict) -> str:
    return schema.spec_hash(spec) if stage == state.STAGE_EXTRACTION else state.criteria_hash(spec["criteria"])


def compatible(run: dict, stage: str, spec: dict, paper: dict,
               metadata: dict | None = None) -> bool:
    if (run.get("paper_uid") != paper["uid"] or run.get("stage") != stage
            or run.get("prompt_version") != spec["prompt_version"]):
        return False
    digest = source_hash(stage, paper, metadata)
    return bool(digest and run.get("source_hash") == digest
                and run.get("config_hash") == config_hash(stage, spec))


def load(stage: str, paper_uid: str | None = None) -> list[dict]:
    return db.load_ai_runs(auth.current_user(), state.active_id(), stage, paper_uid)


def index(stage: str) -> list[dict]:
    """Attempt identities for a stage, without answers; cached until this session adds one."""
    key = f"{_INDEX_KEY}{state.active_id()}"
    cache = st.session_state.setdefault(key, {})
    if stage not in cache:
        cache[stage] = db.load_ai_run_index(auth.current_user(), state.active_id(), stage)
    return cache[stage]


def forget_index() -> None:
    st.session_state.pop(f"{_INDEX_KEY}{state.active_id()}", None)


def outcomes(papers: list[dict], stage: str, spec: dict, metadata: dict | None,
             records: list[dict]) -> tuple[dict[tuple[str, str, str], str], set[str]]:
    """Classify each attempted (paper, provider, model) on the current input.

    Also returns the papers that hold a successful attempt from any model.
    """
    digests = {paper["uid"]: source_hash(stage, paper, metadata) for paper in papers}
    chash, version = config_hash(stage, spec), spec["prompt_version"]
    pairs, succeeded = {}, set()
    for run in records:
        digest = digests.get(run.get("paper_uid"))
        if (not digest or run.get("stage") != stage or run.get("source_hash") != digest
                or run.get("config_hash") != chash or run.get("prompt_version") != version):
            continue
        key = (run["paper_uid"], run["provider"], run["model"])
        if run.get("status") == "succeeded":
            pairs[key] = REPEAT
            succeeded.add(run["paper_uid"])
        elif pairs.get(key) != REPEAT:
            # Records arrive in chronological order, so the last one wins.
            pairs[key] = _UNFINISHED.get(run.get("status"), UNKNOWN)
    return pairs, succeeded


def input_issue(paper: dict, stage: str, spec: dict, metadata: dict | None, model: dict) -> str | None:
    if stage == state.STAGE_ABSTRACT:
        return None
    meta = (metadata or {}).get(paper["uid"], {})
    if not (meta.get("status") == "ok" and meta.get("storage_key") and meta.get("sha256")):
        return "Attach a usable PDF first."
    if stage == state.STAGE_EXTRACTION:
        issue = schema.provider_issue(model["provider"], spec)
        if issue:
            return issue
    return pdf_input_issue(model["provider"], model["model"], meta.get("file_size"), meta.get("page_count"))


def plan(papers: list[dict], stage: str, spec: dict, metadata: dict | None,
         models: list[dict], records: list[dict], *, repeat: bool = False,
         include: tuple[str, ...] | None = None, known: tuple | None = None) -> list[tuple[dict, dict]]:
    """Return the (paper, model) calls of the requested categories.

    By default only never-attempted pairs; ``repeat`` plans every pair.
    ``known`` reuses an ``outcomes`` result that covers these papers.
    """
    if stage not in state.STAGE_TITLES or not 1 <= len(models) <= 3:
        raise ValueError("Choose between one and three models for a valid stage.")
    identities = [(m["provider"], m["model"]) for m in models]
    if len(set(identities)) != len(identities):
        raise ValueError("Choose each model once; use repeat runs for another attempt.")
    if any(provider not in PROVIDERS or model not in PROVIDERS[provider]
           for provider, model in identities):
        raise ValueError("Choose a supported provider and model.")
    wanted = set(include if include is not None else EVERY_ATTEMPT if repeat else (NEW,))
    pairs, _ = known or outcomes(papers, stage, spec, metadata, records)
    tasks = []
    for paper in papers:
        for model in models:
            category = pairs.get((paper["uid"], model["provider"], model["model"]), NEW)
            if category in wanted and not input_issue(paper, stage, spec, metadata, model):
                tasks.append((paper, model))
    return tasks


def awaiting_models(papers: list[dict], stage: str) -> list[dict]:
    """Undecided papers, which the main run serves with every selected model.

    An answer recorded before attempts were kept cannot be matched to a call,
    so such a paper gets further models only through the comparison panel.
    """
    return [paper for paper in papers if _shown(paper, stage) in ("open", "linked")]


def unlinked_answers(papers: list[dict], stage: str) -> list[dict]:
    """Undecided papers whose answer predates recorded attempts."""
    return [paper for paper in papers if _shown(paper, stage) == "unlinked"]


def _shown(paper: dict, stage: str) -> str:
    """Classify what a paper currently shows: decided, open, linked or unlinked."""
    shown = (paper.get("stages") or {}).get(stage) or {}
    if stage == state.STAGE_EXTRACTION:
        decided, answered = bool(shown.get("review_state")), bool(shown.get("ai_answers"))
    else:
        decided = state.final_verdict(paper, stage) is not None
        answered = shown.get("ai_verdict") is not None
    if decided:
        return "decided"
    if not answered:
        return "open"
    return "linked" if shown.get("source_run_id") else "unlinked"


def describe(tasks: list[tuple[dict, dict]], papers: int | None = None) -> str:
    """Count papers and calls separately: several models may serve one paper."""
    if papers is None:
        papers = len({paper["uid"] for paper, _ in tasks})
    return (f"{papers} paper{'' if papers == 1 else 's'} · "
            f"{len(tasks)} call{'' if len(tasks) == 1 else 's'}")


def _candidate(paper: dict, stage: str, spec: dict, meta: dict, model: dict,
               keys: list[str]) -> dict:
    provider, name, key = model["provider"], model["model"], model["api_key"].strip()
    candidate = {"stages": {}}
    try:
        if stage == state.STAGE_ABSTRACT:
            result = judge.judge_abstract(provider, name, key, spec["criteria"],
                                          paper.get("title", ""), paper.get("abstract", ""))
        else:
            pdf = fulltext_storage.load_pdf(meta["storage_key"])
            if fulltext_storage.sha256(pdf) != meta["sha256"]:
                raise ValueError("The stored PDF differs from its metadata. Reattach it first.")
            if stage == state.STAGE_EXTRACTION:
                result = extraction_judge.extract_pdf(provider, name, key, spec, pdf, meta["filename"])
            else:
                result = judge.judge_fulltext(provider, name, key, spec["criteria"],
                                              paper.get("title", ""), pdf, meta["filename"])
        if stage == state.STAGE_EXTRACTION:
            # Redact before raw-answer truncation can retain part of a key.
            extraction.set_ai_result(candidate, spec, meta["sha256"], _redact(result, keys),
                                     provider, name, meta.get("page_count"))
        else:
            state.set_ai_result(candidate, stage, result["verdict"], result["reason"],
                                provider, name, config_hash(stage, spec), spec["prompt_version"])
    except Exception as exc:
        kind = "invalid_response" if isinstance(exc, (InvalidModelResponse, KeyError)) else "call_failed"
        message = _redact(str(exc), keys)[:4000]
        if stage == state.STAGE_EXTRACTION:
            extraction.set_error(candidate, spec, meta["sha256"], message, kind)
        else:
            state.set_ai_error(candidate, stage, message, kind)
    record = state.stage_state(candidate, stage)
    record.update(provider=provider, model=name)
    record.pop("history", None)
    return record


def _redact(value, keys: list[str]):
    if isinstance(value, str):
        for key in sorted(set(keys), key=len, reverse=True):
            if key:
                value = value.replace(key, "[redacted]")
        return value
    if isinstance(value, dict):
        return {_redact(k, keys): _redact(v, keys) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v, keys) for v in value]
    return value


def can_publish(paper: dict, stage: str, run: dict | None = None) -> bool:
    """Whether an attempt may become the paper's proposal without touching human work.

    A complete answer may replace a partly invalid proposal nobody has reviewed.
    """
    current = state.stage_state(paper, stage)
    if ("final_answers" in current or current.get("review_state")
            or current.get("decision") or current.get("human_verdict")
            or current.get("ai_verdict")):
        return False
    if current.get("ai_answers"):
        return bool(current.get("field_errors") and run and run.get("status") == "succeeded")
    return True


def publish_initial(paper: dict, stage: str, run: dict) -> bool:
    """Seed an empty review without replacing any answer or human work."""
    if not can_publish(paper, stage, run):
        return False
    current = state.stage_state(paper, stage)
    candidate = deepcopy(run["result"])
    kept = {key: current[key] for key in ("history", "hv_nonce") if current.get(key)}
    nonce = extraction.form_nonce(current)
    current.clear()
    current.update(candidate, source_run_id=run["id"], **kept)
    if stage == state.STAGE_EXTRACTION:
        current["form_nonce"] = nonce + 1
    return True


def outdated(paper: dict, stage: str, spec: dict, metadata: dict | None = None) -> bool:
    if stage == state.STAGE_EXTRACTION:
        return extraction.is_stale(paper, spec, source_hash(stage, paper, metadata))
    return state.is_stale(paper, stage, config_hash(stage, spec))


def adopt(paper: dict, stage: str, spec: dict, metadata: dict | None, run: dict,
          *, refresh: bool = False) -> bool:
    """Make a completed attempt the paper's proposal; it still needs confirmation.

    ``refresh`` first archives an outdated result, only for a successful attempt.
    """
    if refresh and run.get("status") == "succeeded" and outdated(paper, stage, spec, metadata):
        if stage == state.STAGE_EXTRACTION:
            extraction.archive(paper)
        else:
            state.archive_ai_result(paper, stage)
    return publish_initial(paper, stage, run)


def _restore(papers: list[dict], stage: str, spec: dict, metadata: dict | None,
             models: list[dict], records: list[dict], refresh: bool) -> int:
    """Show answers that were already paid for instead of calling a model again."""
    order = {(m["provider"], m["model"]): rank for rank, m in enumerate(models)}
    wanted = {paper["uid"] for paper in papers}
    by_paper: dict[str, list[dict]] = {}
    for run in records:
        if run.get("status") == "succeeded" and run.get("result") and run.get("paper_uid") in wanted:
            by_paper.setdefault(run["paper_uid"], []).append(run)
    restored = 0
    for paper in papers:
        usable = [run for run in by_paper.get(paper["uid"], [])
                  if compatible(run, stage, spec, paper, metadata)]
        if not usable:
            continue
        # Prefer the selected models in their order, then the earliest attempt.
        best = min(usable, key=lambda run: order.get((run["provider"], run["model"]), len(order)))
        restored += adopt(paper, stage, spec, metadata, best, refresh=refresh)
    return restored


def restorable(papers: list[dict], stage: str, spec: dict, metadata: dict | None,
               records: list[dict], *, refresh: bool = False, known: tuple | None = None) -> list[dict]:
    """Papers whose saved successful attempt would be shown without a new call."""
    _, succeeded = known or outcomes(papers, stage, spec, metadata, records)
    return [paper for paper in papers if paper["uid"] in succeeded
            and (can_publish(paper, stage, {"status": "succeeded"})
                 or (refresh and outdated(paper, stage, spec, metadata)))]


def execute(papers: list[dict], stage: str, spec: dict, metadata: dict | None,
            models: list[dict], *, repeat: bool = False,
            include: tuple[str, ...] | None = None, refresh: bool = False,
            progress=None) -> bool:
    """Run the planned calls, saving each attempt and each new proposal as it arrives.

    Answers already saved for the current input are shown instead of called again.
    """
    if any(not m.get("api_key", "").strip() for m in models):
        st.error("Enter an API key for every selected provider.")
        return False
    if not state.save_active():
        st.error("Stopping before the run — the project could not be saved.")
        return False
    prepared = state.prepare_save()
    if prepared.store.get("pending_ai_completion"):
        st.error("Save the pending AI result before starting another run.")
        return False
    try:
        records = db.load_ai_runs(prepared.user_email, prepared.pid, stage)
        tasks = plan(papers, stage, spec, metadata, models, records, repeat=repeat, include=include)
        if _restore(papers, stage, spec, metadata, models, records, refresh):
            prepared.store[state._UNSAVED_KEY] = True
            if not state.save_result():
                return False
        if not tasks:
            return True
        batch_id = uuid.uuid4().hex
        keys = [m["api_key"].strip() for m in models]
        for position, (paper, model) in enumerate(tasks):
            attempt = db.start_ai_run(
                prepared.user_email, prepared.pid, paper_uid=paper["uid"], stage=stage,
                batch_id=batch_id, provider=model["provider"], model=model["model"],
                config_hash=config_hash(stage, spec), source_hash=source_hash(stage, paper, metadata),
                prompt_version=spec["prompt_version"], prompt_snapshot=_redact(deepcopy(spec), keys),
                expected_version=state.project_version() or None,
            )
            started = perf_counter()
            result = _redact(_candidate(
                paper, stage, spec, (metadata or {}).get(paper["uid"], {}), model, keys,
            ), keys)
            if stage == state.STAGE_EXTRACTION and "invalid_answers" in result:
                # Malformed fields are diagnostics, not structured credential settings.
                result["invalid_answers"] = {
                    qid: (raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False))[:12000]
                    for qid, raw in result.get("invalid_answers", {}).items()
                }
            status = result.get("ai_error_kind") or ("invalid_response" if result.get("field_errors") else "succeeded")
            completion = {
                "run_id": attempt["id"], "status": status, "result": result,
                "duration_seconds": round(perf_counter() - started, 3),
            }
            # Keep a paid answer in session if its independent database write fails.
            prepared.store["pending_ai_completion"] = completion
            completed = db.finish_ai_run(prepared.user_email, prepared.pid,
                                         verify_paper=False, **completion)
            prepared.store.pop("pending_ai_completion", None)
            if adopt(paper, stage, spec, metadata, completed, refresh=refresh):
                # Marked without a session access, so no interruption can separate
                # the changed paper from its recovery marker.
                prepared.store[state._UNSAVED_KEY] = True
                if not state.save_result():
                    return False
            if progress:
                progress(position + 1, len(tasks))
    except (db.DatabaseError, ValueError) as exc:
        st.error(ui.escape_markdown(exc))
        return False
    finally:
        forget_index()
    return True


def confirm_screening(paper: dict, stage: str, spec: dict, run: dict,
                      verdict: str, metadata: dict | None = None) -> None:
    if not compatible(run, stage, spec, paper, metadata) or run.get("status") != "succeeded":
        raise ValueError("Choose a successful result for the current prompt and document.")
    state._validate_verdict(stage, verdict)
    result = run["result"]
    state._validate_verdict(stage, result["ai_verdict"])
    state.archive_ai_result(paper, stage)
    state.set_ai_result(paper, stage, result["ai_verdict"], result.get("ai_reason", ""),
                        run["provider"], run["model"], run["config_hash"], run["prompt_version"])
    current = state.stage_state(paper, stage)
    current.update(source_run_id=run["id"], completed_at=run.get("completed_at"))
    if verdict == result["ai_verdict"]:
        state.record_agree(paper, stage)
    else:
        state.set_human_verdict(paper, stage, verdict)
