"""Project schema, migration, screening decisions, and session persistence.

Screening decisions are ``agree`` (accept AI), ``disagree`` (confirm an override),
or ``human`` (no usable AI verdict). Agreement rates compare actual verdicts.
Extraction reviews are managed by ``features.extraction.state``.

``final_verdict`` preserves recorded outcomes for history and export;
``current_final_verdict`` rejects stale criteria hashes for workflow use.
PDF metadata and imported spreadsheets are stored separately from project JSON.
"""
from __future__ import annotations

import hashlib
import json
import math
import uuid
from copy import deepcopy
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from core import auth, db

SCHEMA_VERSION = 4

MODE_DIRECT = "direct"
MODE_PRISMA = "prisma"

MODE_LABELS = {
    MODE_DIRECT: "Full-text direct review",
    MODE_PRISMA: "PRISMA (abstract screening first)",
}
MODE_DESCRIPTIONS = {
    MODE_DIRECT: "Upload PDFs → full-text screening → data extraction → results.",
    MODE_PRISMA: "Upload a CSV → abstract screening → full-text screening → data extraction → results.",
}

STAGE_ABSTRACT = "abstract"
STAGE_FULLTEXT = "fulltext"
STAGE_EXTRACTION = "extraction"

STAGE_TITLES = {
    STAGE_ABSTRACT: "Abstract Screening",
    STAGE_FULLTEXT: "Full-text Screening",
    STAGE_EXTRACTION: "Data Extraction",
}

MODE_STAGES = {
    MODE_DIRECT: [STAGE_FULLTEXT, STAGE_EXTRACTION],
    MODE_PRISMA: [STAGE_ABSTRACT, STAGE_FULLTEXT, STAGE_EXTRACTION],
}

VERDICT_INCLUDE = "include"
VERDICT_EXCLUDE = "exclude"
VERDICT_UNSURE = "unsure"

STAGE_VERDICTS = {
    STAGE_ABSTRACT: [VERDICT_INCLUDE, VERDICT_EXCLUDE, VERDICT_UNSURE],
    STAGE_FULLTEXT: [VERDICT_INCLUDE, VERDICT_EXCLUDE],
}

VERDICT_LABELS = {
    VERDICT_INCLUDE: "Include",
    VERDICT_EXCLUDE: "Exclude",
    VERDICT_UNSURE: "Unsure",
}
LABEL_PENDING = "Pending"
LABEL_ERROR = "Error"
LABEL_STALE = "Outdated"
LABEL_WITHHELD = "Withheld (earlier stage changed)"
DISPLAY_LABELS = list(VERDICT_LABELS.values()) + [LABEL_PENDING, LABEL_ERROR, LABEL_STALE]

DECISION_AGREE = "agree"
DECISION_DISAGREE = "disagree"
DECISION_HUMAN = "human"

ERROR_CALL_FAILED = "call_failed"  # Eligible for retry.
ERROR_INVALID_RESPONSE = "invalid_response"  # Requires review or explicit retry.

ABSTRACT_ADVANCE = (VERDICT_INCLUDE, VERDICT_UNSURE)

_AI_FIELDS = (
    "ai_verdict", "ai_reason", "ai_error", "ai_error_kind",
    "provider", "model", "criteria_hash", "prompt_version", "completed_at",
)

_KEY = "project_store"
_VERSION_KEY = "project_version"
_SOURCE_KEY = "project_source_unsaved"
_CONFLICT_KEY = "project_conflict"


class UnsupportedSchemaError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def criteria_hash(text: str) -> str:
    return hashlib.sha256((text or "").strip().encode("utf-8")).hexdigest()[:16]


def _validate_verdict(stage: str, verdict: str) -> None:
    allowed = STAGE_VERDICTS.get(stage)
    if not allowed or verdict not in allowed:
        raise ValueError(
            f"Verdict {verdict!r} is not allowed at stage {stage!r} "
            f"(allowed: {allowed})"
        )


def _default_config() -> dict:
    return {
        "abstract_criteria": "",
        "fulltext_criteria": "",
        "extraction_instructions": "",
        "extraction_questions": [],
    }


def new_project_data(mode: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": mode if mode in MODE_STAGES else MODE_PRISMA,
        "config": _default_config(),
        "papers": [],
        "original_columns": [],
        "original_records": [],
        "cursors": {},
    }


def new_paper(doi: str, title: str, abstract: str) -> dict:
    # PDF metadata belongs in the fulltexts table, keyed by this UID.
    return {
        "uid": uuid.uuid4().hex,
        "doi": doi,
        "title": title,
        "abstract": abstract,
        "stages": {},
    }


def stage_state(paper: dict, stage: str) -> dict:
    return paper.setdefault("stages", {}).setdefault(stage, {})


def final_verdict(paper: dict, stage: str) -> str | None:
    """Raw recorded outcome, regardless of criteria staleness."""
    s = stage_state(paper, stage)
    decision = s.get("decision")
    if decision == DECISION_AGREE:
        return s.get("ai_verdict")
    if decision in (DECISION_DISAGREE, DECISION_HUMAN):
        return s.get("human_verdict")
    return None


def current_final_verdict(paper: dict, stage: str, chash: str) -> str | None:
    """Return the recorded outcome only when its criteria hash is current."""
    s = stage_state(paper, stage)
    decision = s.get("decision")
    if decision in (DECISION_AGREE, DECISION_DISAGREE):
        if s.get("criteria_hash") != chash:
            return None
        return s.get("ai_verdict") if decision == DECISION_AGREE else s.get("human_verdict")
    if decision == DECISION_HUMAN:
        if s.get("review_criteria_hash") != chash:
            return None
        return s.get("human_verdict")
    return None


def is_reviewed(paper: dict, stage: str, chash: str | None = None) -> bool:
    if chash is None:
        return final_verdict(paper, stage) is not None
    return current_final_verdict(paper, stage, chash) is not None


def is_stale(paper: dict, stage: str, chash: str) -> bool:
    """Whether an AI or human-only outcome uses a different criteria hash."""
    s = stage_state(paper, stage)
    if s.get("ai_verdict") is not None:
        return s.get("criteria_hash") != chash
    return (s.get("decision") == DECISION_HUMAN
            and s.get("human_verdict") is not None
            and s.get("review_criteria_hash") != chash)


def display_label(paper: dict, stage: str, chash: str | None = None) -> str:
    if chash is not None and is_stale(paper, stage, chash):
        return LABEL_STALE
    fv = final_verdict(paper, stage) if chash is None else current_final_verdict(paper, stage, chash)
    if fv:
        return VERDICT_LABELS[fv]
    s = stage_state(paper, stage)
    if s.get("ai_verdict") is None and s.get("ai_error"):
        return LABEL_ERROR
    return LABEL_PENDING


def record_agree(paper: dict, stage: str) -> None:
    stage_state(paper, stage)["decision"] = DECISION_AGREE


def record_disagree(paper: dict, stage: str) -> None:
    """Clear a prior override on entry; repeated clicks are idempotent."""
    s = stage_state(paper, stage)
    if s.get("decision") == DECISION_DISAGREE:
        return
    s["decision"] = DECISION_DISAGREE
    if s.get("human_verdict") is not None:
        s["human_verdict"] = None
        s["hv_nonce"] = int(s.get("hv_nonce", 0)) + 1


def set_human_verdict(paper: dict, stage: str, verdict: str,
                      review_hash: str | None = None) -> None:
    """Record an override, requiring a criteria hash for human-only decisions."""
    _validate_verdict(stage, verdict)
    s = stage_state(paper, stage)
    if s.get("ai_verdict"):
        s["human_verdict"] = verdict
        s["decision"] = DECISION_DISAGREE
        return
    if not review_hash:
        raise ValueError(
            "A human-only verdict requires review_hash (the hash of the "
            "criteria the reviewer was shown)"
        )
    s["human_verdict"] = verdict
    s["decision"] = DECISION_HUMAN
    s["review_criteria_hash"] = review_hash


def set_ai_result(
    paper: dict, stage: str, verdict: str, reason: str,
    provider: str, model: str, chash: str, prompt_version: str,
) -> None:
    _validate_verdict(stage, verdict)
    stage_state(paper, stage).update(
        ai_verdict=verdict,
        ai_reason=reason,
        ai_error=None,
        ai_error_kind=None,
        provider=provider,
        model=model,
        criteria_hash=chash,
        prompt_version=prompt_version,
        completed_at=_now(),
    )


def set_ai_error(paper: dict, stage: str, message: str, kind: str) -> None:
    stage_state(paper, stage).update(
        ai_verdict=None, ai_reason=None, ai_error=str(message), ai_error_kind=kind,
    )


def archive_ai_result(paper: dict, stage: str) -> None:
    """Archive the AI and human outcomes, leaving the stage pending."""
    s = stage_state(paper, stage)
    if (s.get("ai_verdict") is None and not s.get("ai_error")
            and s.get("decision") is None and s.get("human_verdict") is None):
        return
    entry = {k: s.get(k) for k in _AI_FIELDS}
    entry["decision"] = s.get("decision")
    entry["human_verdict"] = s.get("human_verdict")
    entry["review_criteria_hash"] = s.get("review_criteria_hash")
    entry["archived_at"] = _now()
    history = s.setdefault("history", [])
    history.append(entry)
    s["history"] = history[-10:]
    for key in _AI_FIELDS:
        s.pop(key, None)
    s.pop("review_criteria_hash", None)
    s["decision"] = None
    s["human_verdict"] = None
    s["hv_nonce"] = int(s.get("hv_nonce", 0)) + 1


def archive_if_source_changed(
    paper: dict,
    stage: str,
    old_source_hash: str | None,
    new_source_hash: str | None,
) -> bool:
    """Invalidate a stage outcome when its source PDF changes."""
    if not new_source_hash or old_source_hash == new_source_hash:
        return False
    s = stage_state(paper, stage)
    has_current_outcome = bool(
        s.get("ai_verdict") is not None
        or s.get("ai_error")
        or s.get("decision") is not None
        or s.get("human_verdict") is not None
    )
    # A first PDF invalidates any verdict made without it.
    if not old_source_hash and not has_current_outcome:
        return False
    archive_ai_result(paper, stage)
    return True


def archive_document_results(paper: dict, old_hash: str | None, new_hash: str) -> bool:
    """Invalidate both document-based stages before replacing their source."""
    from features.extraction import state as extraction

    changed = archive_if_source_changed(paper, STAGE_FULLTEXT, old_hash, new_hash)
    if old_hash != new_hash and extraction.get(paper).get("spec_hash"):
        extraction.archive(paper)
        changed = True
    return changed


def eligible_papers(papers_list: list[dict], mode: str, stage: str,
                    hashes: dict[str, str]) -> list[tuple[int, dict]]:
    """Return eligible (index, paper) pairs using each stage's current hash."""
    pairs = list(enumerate(papers_list))
    if stage == STAGE_ABSTRACT:
        return pairs
    if stage == STAGE_FULLTEXT:
        if mode == MODE_DIRECT:
            return pairs
        return [(i, p) for i, p in pairs
                if current_final_verdict(p, STAGE_ABSTRACT, hashes[STAGE_ABSTRACT])
                in ABSTRACT_ADVANCE]
    if stage == STAGE_EXTRACTION:
        return [(i, p) for i, p in eligible_papers(papers_list, mode, STAGE_FULLTEXT, hashes)
                if current_final_verdict(p, STAGE_FULLTEXT, hashes[STAGE_FULLTEXT])
                == VERDICT_INCLUDE]
    raise ValueError(f"Unknown workflow stage: {stage}")


def pending_of(papers_list: list[dict], mode: str, stage: str,
               hashes: dict[str, str]) -> list[tuple[int, dict]]:
    """Select unjudged papers, excluding invalid replies and human decisions.

    Stale human decisions require explicit rerun or archival to avoid rebilling.
    """
    out = []
    for i, p in eligible_papers(papers_list, mode, stage, hashes):
        s = stage_state(p, stage)
        if s.get("ai_verdict") is not None:
            continue
        if s.get("ai_error_kind") == ERROR_INVALID_RESPONSE:
            continue
        if final_verdict(p, stage) is not None:
            continue
        out.append((i, p))
    return out


def stale_of(papers_list: list[dict], mode: str, stage: str,
             hashes: dict[str, str]) -> list[tuple[int, dict]]:
    return [(i, p) for i, p in eligible_papers(papers_list, mode, stage, hashes)
            if is_stale(p, stage, hashes[stage])]


def invalid_of(papers_list: list[dict], mode: str, stage: str,
               hashes: dict[str, str]) -> list[tuple[int, dict]]:
    out = []
    for i, p in eligible_papers(papers_list, mode, stage, hashes):
        s = stage_state(p, stage)
        if (s.get("ai_verdict") is None
                and s.get("ai_error_kind") == ERROR_INVALID_RESPONSE
                and final_verdict(p, stage) is None):
            out.append((i, p))
    return out


def summarize(papers_list: list[dict], mode: str, stage: str,
              hashes: dict[str, str]) -> dict:
    chash = hashes.get(stage)
    elig = eligible_papers(papers_list, mode, stage, hashes)
    total = len(elig)
    ai_done = ai_failed = ai_invalid = stale = 0
    agreed = disagreed = human_only = reviewed = 0
    counts = dict.fromkeys(DISPLAY_LABELS, 0)
    for _, p in elig:
        s = stage_state(p, stage)
        if s.get("ai_verdict") is not None:
            ai_done += 1
        elif s.get("ai_error"):
            if s.get("ai_error_kind") == ERROR_INVALID_RESPONSE:
                ai_invalid += 1
            else:
                ai_failed += 1
        if chash is not None and is_stale(p, stage, chash):
            stale += 1
        fv = (current_final_verdict(p, stage, chash) if chash is not None
              else final_verdict(p, stage))
        if fv is not None:
            reviewed += 1
            # Button choice alone does not determine agreement.
            if s.get("decision") in (DECISION_AGREE, DECISION_DISAGREE) \
                    and s.get("ai_verdict") is not None:
                if fv == s.get("ai_verdict"):
                    agreed += 1
                else:
                    disagreed += 1
            else:
                human_only += 1
        counts[display_label(p, stage, chash)] += 1
    decided = agreed + disagreed  # Excludes human-only decisions.
    return {
        "total": total,
        "ai_done": ai_done,
        "ai_failed": ai_failed,
        "ai_invalid": ai_invalid,
        "stale": stale,
        "reviewed": reviewed,
        "remaining": total - reviewed,
        "agreed": agreed,
        "disagreed": disagreed,
        "human_only": human_only,
        "agreement_rate": (agreed / decided) if decided else None,
        "counts": counts,
    }


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))


def _v1_verdict(score, threshold: int) -> str | None:
    if not _is_number(score):
        return None
    return VERDICT_INCLUDE if score >= threshold else VERDICT_EXCLUDE


def _v1_criteria_text(topic: str, rubric: list[dict], threshold: int) -> str:
    include_defs = [r.get("definition", "") for r in rubric
                    if r.get("score", 0) >= threshold and r.get("definition")]
    exclude_defs = [r.get("definition", "") for r in rubric
                    if r.get("score", 0) < threshold and r.get("definition")]
    parts = []
    if topic:
        parts.append(f"Research topic:\n{topic}")
    if include_defs:
        parts.append("Include a paper if it matches any of the following:\n"
                     + "\n".join(f"- {d}" for d in include_defs))
    if exclude_defs:
        parts.append("Exclude a paper if it at best matches only the following:\n"
                     + "\n".join(f"- {d}" for d in exclude_defs))
    return "\n\n".join(parts)


def _migrate_v1(data: dict) -> dict:
    threshold = int(data.get("threshold", 80))
    rubric = data.get("rubric") or []
    out = new_project_data(MODE_PRISMA)
    criteria = _v1_criteria_text(str(data.get("topic", "") or ""), rubric, threshold)
    out["config"]["abstract_criteria"] = criteria
    chash = criteria_hash(criteria)
    for old in data.get("papers") or []:
        paper = new_paper(old.get("doi", ""), old.get("title", ""), old.get("abstract", ""))
        score = old.get("ai_score")
        ai_verdict = _v1_verdict(score, threshold)
        decision = old.get("decision")
        human_verdict = None
        if decision == DECISION_DISAGREE:
            human_verdict = _v1_verdict(old.get("human_score"), threshold)
        if ai_verdict is None:
            # Without AI, only a confirmed human score is a valid decision.
            if decision == DECISION_DISAGREE and human_verdict is not None:
                decision = DECISION_HUMAN
            else:
                decision = None
        stage: dict = {
            "ai_verdict": ai_verdict,
            "ai_reason": old.get("ai_reason") if ai_verdict else None,
            "ai_error": None if ai_verdict else old.get("ai_reason"),
            "ai_error_kind": None if ai_verdict else ERROR_CALL_FAILED,
            "decision": decision,
            "human_verdict": human_verdict,
            "legacy_ai_score": score if _is_number(score) else None,
            "legacy_human_score": old.get("human_score"),
        }
        if ai_verdict:
            stage["criteria_hash"] = chash
            stage["prompt_version"] = "v1-rubric"
            stage["completed_at"] = None  # unknown; predates result timestamps
        if decision == DECISION_HUMAN:
            stage["review_criteria_hash"] = chash
        paper["stages"][STAGE_ABSTRACT] = stage
        out["papers"].append(paper)
    out["original_columns"] = data.get("original_columns", [])
    out["original_records"] = data.get("original_records", [])
    out["cursors"] = {STAGE_ABSTRACT: int(data.get("idx", 0))}
    return out


def migrate(data: dict) -> dict:
    """Upgrade supported legacy data; reject unsupported schema versions."""
    if not data:
        return new_project_data(MODE_PRISMA)
    version = data.get("schema_version")
    if version is None or version == 1:
        return _migrate_v1(data)
    # Schemas 2 and 3 store the spreadsheet inline; schema 4 uses a table.
    if version not in (2, 3, SCHEMA_VERSION):
        raise UnsupportedSchemaError(
            f"This project was saved with a newer version of the platform "
            f"(schema {version}, this app reads schema {SCHEMA_VERSION}). "
            "Update the deployed app before opening it."
        )
    data["schema_version"] = SCHEMA_VERSION
    config = _default_config()
    config.update(data.get("config") or {})
    data["config"] = config
    data.setdefault("papers", [])
    data.setdefault("original_columns", [])
    data.setdefault("original_records", [])
    data.setdefault("cursors", {})
    if data.get("mode") not in MODE_STAGES:
        data["mode"] = MODE_PRISMA
    return data


def active_id() -> str | None:
    return st.session_state.get("active_project_id")


def active_name() -> str | None:
    return st.session_state.get("active_project_name")


def set_active(pid: str | None, name: str | None) -> None:
    st.session_state["active_project_id"] = pid
    st.session_state["active_project_name"] = name


def _as_version(value: object) -> int:
    """Coerce a stored/returned row version; 0 means "unknown, save anyway"."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def project_version() -> int:
    """Row version this session last read or wrote; 0 when unknown."""
    return _as_version(st.session_state.get(_VERSION_KEY, 0))


def load_into_session(data: dict, version: int | None = None,
                      source_df: pd.DataFrame | None = None,
                      project_id: str | None = None,
                      source: tuple[list, list] | None = None) -> None:
    """Replace session data, preserving the version when ``version`` is None.

    ``source_df`` bypasses source loading for rollback. Pass ``project_id``
    when switching projects; otherwise source loading uses the active project.
    """
    data = migrate(data)
    inline_records = data.get("original_records") or []
    if source_df is not None:
        df = source_df
    elif inline_records:
        # Move legacy inline spreadsheets to their own row on the next save.
        df = pd.DataFrame(inline_records, columns=data.get("original_columns") or None)
        st.session_state[_SOURCE_KEY] = True
    elif source is not None:
        columns, records = source
        df = pd.DataFrame(records, columns=columns or None)
        st.session_state[_SOURCE_KEY] = False
    else:
        columns, records = [], []
        pid = project_id if project_id is not None else active_id()
        if pid:
            # Propagate read failures to avoid silently exporting empty columns.
            columns, records = db.load_project_source(auth.current_user(), pid)
        df = pd.DataFrame(records, columns=columns or None)
        st.session_state[_SOURCE_KEY] = False
    st.session_state[_KEY] = {
        "mode": data["mode"],
        "config": data["config"],
        "papers": data["papers"],
        "original_df": df,
        "cursors": data["cursors"],
        "pending_pdf_deletions": data.get("pending_pdf_deletions", []),
    }
    if version is not None:
        st.session_state[_VERSION_KEY] = _as_version(version)
        st.session_state.pop(_CONFLICT_KEY, None)


def reload_project(pid: str) -> None:
    """Read review work and its spreadsheet from one database snapshot."""
    data, version, source = db.load_project_bundle(auth.current_user(), pid)
    if not version:
        raise db.DatabaseError("Project no longer exists or is not accessible.")
    load_into_session(data, version, project_id=pid, source=source)
    # A discarded form must not autosave its old widget values into the reload.
    prefixes = (f"extraction:{pid}:", f"abstract_prompt_{pid}", f"fulltext_prompt_{pid}",
                f"jump_abstract_{pid}", f"jump_fulltext_{pid}", "abstract:", "fulltext:")
    for key in list(st.session_state):
        if str(key).startswith(prefixes):
            del st.session_state[key]


def clear_session() -> None:
    st.session_state.pop(_KEY, None)
    st.session_state.pop(_SOURCE_KEY, None)
    st.session_state.pop(_VERSION_KEY, None)
    st.session_state.pop(_CONFLICT_KEY, None)


def _store() -> dict:
    if _KEY not in st.session_state:
        load_into_session({})
    return st.session_state[_KEY]


def loaded() -> bool:
    return _KEY in st.session_state


def snapshot() -> dict:
    """Return project data without the separately stored imported spreadsheet."""
    store = _store()
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": store["mode"],
        "config": store["config"],
        "papers": store["papers"],
        "cursors": store["cursors"],
        "pending_pdf_deletions": store.get("pending_pdf_deletions", []),
    }


def _pending_source() -> tuple[list, list] | None:
    """Return unsaved source data for the same transaction as the project."""
    if not st.session_state.get(_SOURCE_KEY):
        return None
    df: pd.DataFrame = _store()["original_df"]
    records = json.loads(df.to_json(orient="records")) if len(df.columns) else []
    return list(df.columns), records


def save_active(*, fulltext: dict | None = None) -> bool:
    """Save this version, optionally updating PDF metadata in the same transaction."""
    pid = active_id()
    if not pid:
        return True
    user_email = auth.current_user()
    source = _pending_source()
    expected = project_version() or None
    options = {"fulltext": fulltext} if fulltext is not None else {}
    try:
        version = db.save_project(user_email, pid, snapshot(),
                                  expected_version=expected, source=source, **options)
    except db.ProjectConflictError as exc:
        st.session_state[_CONFLICT_KEY] = str(exc)
        st.error(str(exc))
        return False
    except db.DatabaseError as exc:
        st.error(str(exc))
        return False
    st.session_state[_VERSION_KEY] = _as_version(version)
    st.session_state.pop(_CONFLICT_KEY, None)
    if source is not None:
        st.session_state[_SOURCE_KEY] = False
    return True


def commit(change) -> None:
    """Attempt to save an edit, rolling back handled save failures."""
    before, before_df = deepcopy(snapshot()), _store()["original_df"]
    source_pending = st.session_state.get(_SOURCE_KEY, False)
    try:
        change()
        saved = save_active()
    except ValueError as exc:
        load_into_session(before, source_df=before_df)
        st.session_state[_SOURCE_KEY] = source_pending
        st.error(str(exc))
        st.stop()
        return
    except Exception:
        load_into_session(before, source_df=before_df)
        st.session_state[_SOURCE_KEY] = source_pending
        raise
    if not saved:
        load_into_session(before, source_df=before_df)
        st.session_state[_SOURCE_KEY] = source_pending
        st.error("Changes were not saved. Your previous saved data is unchanged; try again.")
        st.stop()


def unsaved_results_key() -> str:
    return f"_unsaved_results_{active_id()}"


def save_result() -> bool:
    """Retain an AI result or human draft when its checkpoint fails."""
    saved = save_active()
    if not saved:
        st.session_state[unsaved_results_key()] = True
    return saved


def require_saved_results() -> None:
    if not st.session_state.get(unsaved_results_key()):
        return
    conflict = st.session_state.get(_CONFLICT_KEY)
    st.error(
        "Results or drafts are waiting to be saved. Keep this session open and retry "
        "saving before continuing." + (f" {conflict}" if conflict else "")
    )
    if st.button("Retry saving results", type="primary") and save_active():
        st.session_state.pop(unsaved_results_key(), None)
        st.rerun()
    if conflict:
        st.warning("Another session changed this project. This copy cannot overwrite it. "
                   "Download the unsaved work before discarding this session's changes.")
        recovery = dict(snapshot())
        frame = _store()["original_df"]
        recovery.update(original_columns=list(frame.columns),
                        original_records=json.loads(frame.to_json(orient="records")))
        st.download_button("Download unsaved work", json.dumps(recovery, ensure_ascii=False),
                           file_name="unsaved_review.json", mime="application/json", on_click="ignore")
        discard = st.checkbox("I have backed up my work and want to discard this session's changes.")
        if st.button("Discard local changes and reload", disabled=not discard):
            try:
                reload_project(active_id())
            except (db.DatabaseError, UnsupportedSchemaError) as exc:
                st.error(str(exc))
            else:
                st.session_state.pop(unsaved_results_key(), None)
                st.rerun()
    st.stop()


def mode() -> str:
    return _store()["mode"]


def stages() -> list[str]:
    return MODE_STAGES[mode()]


def config() -> dict:
    return _store()["config"]


def papers() -> list[dict]:
    return _store()["papers"]


def has_papers() -> bool:
    return bool(papers())


def replace_papers(rows: list[dict], original_df: pd.DataFrame) -> None:
    store = _store()
    store["papers"] = rows
    store["original_df"] = original_df
    store["cursors"] = {}
    st.session_state[_SOURCE_KEY] = True


def append_papers(rows: list[dict]) -> None:
    if not rows:
        return
    store = _store()
    store["papers"].extend(rows)
    store["cursors"][STAGE_FULLTEXT] = min(
        int(store["cursors"].get(STAGE_FULLTEXT, 0)),
        max(len(store["papers"]) - 1, 0),
    )


def remove_paper(paper_uid: str) -> bool:
    """Remove one paper from project JSON; PDF cleanup is handled by the caller."""
    store = _store()
    before = len(store["papers"])
    store["papers"] = [p for p in store["papers"] if p.get("uid") != paper_uid]
    store["cursors"][STAGE_FULLTEXT] = 0
    return len(store["papers"]) != before


def pending_pdf_deletions() -> list[dict]:
    return _store().setdefault("pending_pdf_deletions", [])


def remove_paper_with_cleanup(paper_uid: str, storage_key: str | None) -> None:
    """Keep the deletion target durable until both storage and metadata are gone."""
    if remove_paper(paper_uid):
        pending_pdf_deletions().append({"paper_uid": paper_uid, "storage_key": storage_key})


def current_criteria_hash(stage: str) -> str:
    return criteria_hash(config().get(f"{stage}_criteria", ""))


def current_hashes() -> dict[str, str]:
    return {stage: current_criteria_hash(stage) for stage in STAGE_VERDICTS}


def stage_papers(stage: str) -> list[tuple[int, dict]]:
    return eligible_papers(papers(), mode(), stage, current_hashes())


def pending_papers(stage: str) -> list[tuple[int, dict]]:
    return pending_of(papers(), mode(), stage, current_hashes())


def stale_papers(stage: str) -> list[tuple[int, dict]]:
    return stale_of(papers(), mode(), stage, current_hashes())


def invalid_papers(stage: str) -> list[tuple[int, dict]]:
    return invalid_of(papers(), mode(), stage, current_hashes())


def stage_summary(stage: str) -> dict:
    return summarize(papers(), mode(), stage, current_hashes())


def cursor(stage: str) -> int:
    return int(_store()["cursors"].get(stage, 0))


def goto(stage: str, i: int, limit: int) -> None:
    _store()["cursors"][stage] = max(0, min(max(limit - 1, 0), i))


def _has_record(paper: dict, stage: str) -> bool:
    """Whether review data exists; errors alone do not count as withheld work."""
    s = stage_state(paper, stage)
    return any(s.get(key) is not None for key in
               ("ai_verdict", "decision", "human_verdict"))


def _stage_columns(base: pd.DataFrame, papers_list: list[dict], mode_key: str,
                   stage: str, prefix: str, hashes: dict[str, str]) -> None:
    eligible = {i for i, _ in eligible_papers(papers_list, mode_key, stage, hashes)}
    chash = hashes.get(stage)
    agreed_text = {DECISION_AGREE: "Yes", DECISION_DISAGREE: "No",
                   DECISION_HUMAN: "Human only"}

    def cell(i: int, fn) -> str:
        # Preserve recorded work when an earlier decision removes eligibility.
        if i not in eligible and not _has_record(papers_list[i], stage):
            return ""
        return fn(papers_list[i], stage_state(papers_list[i], stage))

    n = range(len(papers_list))
    base[f"{prefix}: AI verdict"] = [
        cell(i, lambda p, s: VERDICT_LABELS.get(s.get("ai_verdict"), LABEL_ERROR if s.get("ai_error") else ""))
        for i in n
    ]
    base[f"{prefix}: AI reason"] = [
        cell(i, lambda p, s: s.get("ai_reason") or s.get("ai_error") or "") for i in n
    ]
    base[f"{prefix}: Agreed?"] = [
        cell(i, lambda p, s: agreed_text.get(s.get("decision"), "")) for i in n
    ]
    base[f"{prefix}: Your verdict"] = [
        cell(i, lambda p, s: VERDICT_LABELS.get(s.get("human_verdict"), "")
             if s.get("decision") in (DECISION_DISAGREE, DECISION_HUMAN) else "")
        for i in n
    ]
    base[f"{prefix}: Final"] = [
        LABEL_WITHHELD if (i not in eligible and _has_record(papers_list[i], stage))
        else cell(i, lambda p, s: display_label(p, stage, chash))
        for i in n
    ]


def results_dataframe() -> pd.DataFrame:
    store = _store()
    papers_list = papers()
    mode_key = mode()
    hashes = current_hashes()
    base: pd.DataFrame = store["original_df"].copy()
    if len(base) != len(papers_list):
        base = pd.DataFrame({"Title": [p.get("title") for p in papers_list],
                             "DOI": [p.get("doi") for p in papers_list],
                             "Abstract": [p.get("abstract") for p in papers_list]})
    if STAGE_ABSTRACT in MODE_STAGES[mode_key]:
        _stage_columns(base, papers_list, mode_key, STAGE_ABSTRACT, "Abstract", hashes)
    if STAGE_FULLTEXT in MODE_STAGES[mode_key]:
        _stage_columns(base, papers_list, mode_key, STAGE_FULLTEXT, "Full-text", hashes)
    return base
