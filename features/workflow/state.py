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

SCHEMA_VERSION = 5

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
    "source_run_id",
)

_KEY = "project_store"
_PROJECT_ID_KEY = "project_id"
_VERSION_KEY = "project_version"
_SOURCE_KEY = "project_source_unsaved"
_CONFLICT_KEY = "project_conflict"
_UNSAVED_KEY = "unsaved_results"
_RUNS_MIGRATED_KEY = "ai_runs_migrated"


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
        _RUNS_MIGRATED_KEY: True,
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
    s.setdefault("history", []).append(entry)
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
    # Its AI results predate recorded attempts, so the first save must import them.
    out.pop(_RUNS_MIGRATED_KEY)
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
    if version not in (2, 3, 4, SCHEMA_VERSION):
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
    # A store without an owner belongs to the project that was active when it loaded.
    previous_id = active_id()
    store = st.session_state.get(_KEY)
    if store is not None:
        store.setdefault(_PROJECT_ID_KEY, previous_id)
    st.session_state["active_project_id"] = pid
    st.session_state["active_project_name"] = name


def _as_version(value: object) -> int:
    """Coerce a stored/returned row version; 0 means unknown."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def project_version() -> int:
    """Row version this session last read or wrote; 0 when unknown."""
    return _as_version(_store().get(_VERSION_KEY, 0))


def load_into_session(data: dict, version: int | None = None,
                      project_id: str | None = None,
                      source: tuple[list, list] | None = None,
                      reset_widgets: bool = False,
                      keep_widgets: tuple[str, ...] = ()) -> None:
    """Replace session data, preserving the version when ``version`` is None.

    Pass ``project_id`` when switching projects; otherwise source loading uses
    the active project.
    ``keep_widgets`` names session keys that survive ``reset_widgets``.
    """
    data = migrate(data)
    previous = st.session_state.get(_KEY, {})
    pid = project_id if project_id is not None else active_id()
    source_pending = previous.get(_SOURCE_KEY, False)
    inline_records = data.get("original_records") or []
    if inline_records:
        # Move legacy inline spreadsheets to their own row on the next save.
        df = pd.DataFrame(inline_records, columns=data.get("original_columns") or None)
        source_pending = True
    elif source is not None:
        columns, records = source
        df = pd.DataFrame(records, columns=columns or None)
        source_pending = False
    else:
        columns, records = [], []
        if pid:
            # Propagate read failures to avoid silently exporting empty columns.
            columns, records = db.load_project_source(auth.current_user(), pid)
        df = pd.DataFrame(records, columns=columns or None)
        source_pending = False
    replacement = {
        _PROJECT_ID_KEY: pid,
        "mode": data["mode"],
        "config": data["config"],
        "papers": data["papers"],
        "original_df": df,
        "cursors": data["cursors"],
        "pending_pdf_deletions": data.get("pending_pdf_deletions", []),
        _VERSION_KEY: _as_version(version) if version is not None else previous.get(_VERSION_KEY, 0),
        _SOURCE_KEY: source_pending,
        _CONFLICT_KEY: None if version is not None else previous.get(_CONFLICT_KEY),
        _UNSAVED_KEY: False if version is not None else previous.get(_UNSAVED_KEY, False),
        _RUNS_MIGRATED_KEY: bool(data.get(_RUNS_MIGRATED_KEY)),
    }
    if reset_widgets:
        # Finish interruptible cleanup before publishing the loaded document.
        prefixes = (f"extraction:{pid}:", f"abstract_prompt_{pid}", f"fulltext_prompt_{pid}",
                    f"jump_abstract_{pid}", f"jump_fulltext_{pid}", "abstract:", "fulltext:",
                    f"runs:index:{pid}")
        for key in list(st.session_state):
            if str(key).startswith(prefixes) and key not in keep_widgets:
                del st.session_state[key]
    st.session_state[_KEY] = replacement


def reload_project(pid: str, *, keep_setup_draft: bool = False) -> None:
    """Read review work and its spreadsheet from one database snapshot."""
    data, version, source = db.load_project_bundle(auth.current_user(), pid)
    if not version:
        raise db.DatabaseError("Project no longer exists or is not accessible.")
    # A discarded form must not autosave its old widget values into the reload.
    keep = (f"extraction:{pid}:setup_draft",) if keep_setup_draft else ()
    load_into_session(data, version, project_id=pid, source=source, reset_widgets=True,
                      keep_widgets=keep)


def clear_session() -> None:
    st.session_state.pop(_KEY, None)
    st.session_state.pop(_SOURCE_KEY, None)
    st.session_state.pop(_VERSION_KEY, None)
    st.session_state.pop(_CONFLICT_KEY, None)


def _store() -> dict:
    if _KEY not in st.session_state:
        load_into_session({})
    store = st.session_state[_KEY]
    if _PROJECT_ID_KEY not in store:
        pid = active_id()
        if pid is not None:
            store[_PROJECT_ID_KEY] = pid
    if _VERSION_KEY not in store:
        # Adopt bookkeeping still held in top-level session keys.
        store.update({key: st.session_state.get(key, default) for key, default in (
            (_VERSION_KEY, 0), (_SOURCE_KEY, False), (_CONFLICT_KEY, None),
        )})
    return store


def loaded() -> bool:
    store = st.session_state.get(_KEY)
    pid = active_id()
    return store is not None and store.get(_PROJECT_ID_KEY, pid) == pid


def snapshot() -> dict:
    """Return project data without the separately stored imported spreadsheet."""
    return _snapshot(_store())


def _snapshot(store: dict) -> dict:
    data = {
        "schema_version": SCHEMA_VERSION,
        "mode": store["mode"],
        "config": store["config"],
        "papers": store["papers"],
        "cursors": store["cursors"],
        "pending_pdf_deletions": store.get("pending_pdf_deletions", []),
    }
    if store.get(_RUNS_MIGRATED_KEY):
        data[_RUNS_MIGRATED_KEY] = True
    return data


def _paper_uids(data: dict) -> set[str]:
    return {paper["uid"] for paper in data.get("papers") or []}


def _pending_source(store: dict | None = None) -> tuple[list, list] | None:
    """Return unsaved source data for the same transaction as the project."""
    store = _store() if store is None else store
    if not store.get(_SOURCE_KEY):
        return None
    df: pd.DataFrame = store["original_df"]
    records = json.loads(df.to_json(orient="records")) if len(df.columns) else []
    return list(df.columns), records


class PreparedSave:
    """Capture session references before I/O; commit without Streamlit yield points."""

    def __init__(self, user_email: str, pid: str, store: dict):
        self.user_email, self.pid, self.store = user_email, pid, store
        self.version = _as_version(store.get(_VERSION_KEY))
        self.data = deepcopy(_snapshot(store))
        self.source = _pending_source(store)

    def commit(self, data: dict, *, source: tuple[list, list] | None = None,
               fulltext: dict | None = None, remove_fulltexts: bool = False,
               remove_fulltext_uids: list[str] | None = None) -> int:
        source = self.source if source is None else source
        migrated = bool(self.store.get(_RUNS_MIGRATED_KEY))
        # The first save imports earlier AI snapshots; later saves skip that scan.
        data = {**data, _RUNS_MIGRATED_KEY: True}
        replacement = {**self.store, **data}
        if source is not None:
            replacement["original_df"] = pd.DataFrame(source[1], columns=source[0] or None)
        options = {}
        if fulltext is not None:
            options["fulltext"] = fulltext
        if remove_fulltexts:
            options["remove_fulltexts"] = True
        if remove_fulltext_uids:
            options["remove_fulltext_uids"] = remove_fulltext_uids
        if migrated:
            options["import_legacy_runs"] = False
            removed = sorted(_paper_uids(self.data) - _paper_uids(data))
            if removed:
                options["removed_paper_uids"] = removed
        version = db.save_project(self.user_email, self.pid, data,
                                  expected_version=self.version or None, source=source, **options)
        # No session/UI access may interrupt version acknowledgement after commit.
        replacement.update({_VERSION_KEY: _as_version(version), _SOURCE_KEY: False,
                            _CONFLICT_KEY: None, _UNSAVED_KEY: False})
        self.store.clear()
        self.store.update(replacement)
        self.version, self.source, self.data = _as_version(version), None, data
        return self.version


def prepare_save() -> PreparedSave:
    pid, user_email, store = active_id(), auth.current_user(), _store()
    if not pid:
        raise db.DatabaseError("No project is open.")
    if store.get(_PROJECT_ID_KEY) != pid:
        raise db.DatabaseError(
            "The project switch was interrupted. Open the project again from My Projects "
            "before saving; no changes were saved."
        )
    return PreparedSave(user_email, pid, store)


def save_active(*, fulltext: dict | None = None) -> bool:
    """Save this version, optionally updating PDF metadata in the same transaction."""
    if not active_id():
        return True
    try:
        prepared = prepare_save()
    except db.DatabaseError as exc:
        st.error(str(exc))
        return False
    try:
        prepared.commit(_snapshot(prepared.store), fulltext=fulltext)
    except db.ProjectConflictError as exc:
        prepared.store[_CONFLICT_KEY] = str(exc)
        st.error(str(exc))
        return False
    except db.DatabaseError as exc:
        st.error(str(exc))
        return False
    return True


def commit(change) -> None:
    """Restore interrupted edits unless their database transaction already committed."""
    store = _store()
    before = deepcopy(store)

    def rollback():
        if store.get(_VERSION_KEY) != before.get(_VERSION_KEY):
            return
        conflict = store.get(_CONFLICT_KEY)
        store.clear()
        store.update(before)
        store[_CONFLICT_KEY] = conflict

    try:
        change()
        saved = save_active()
    except ValueError as exc:
        rollback()
        st.error(str(exc))
        st.stop()
        return
    except BaseException:
        rollback()
        raise
    if not saved:
        rollback()
        st.error("Changes were not saved. Your previous saved data is unchanged; try again.")
        st.stop()


def has_unsaved_results() -> bool:
    """Whether a paid result or a draft exists only in this session."""
    return bool(st.session_state.get(_KEY, {}).get(_UNSAVED_KEY))


def begin_result() -> None:
    """Arm recovery before paid work or a draft mutation can be interrupted."""
    _store()[_UNSAVED_KEY] = True


def clear_unsaved_results() -> None:
    _store()[_UNSAVED_KEY] = False


def save_result() -> bool:
    """Retain an AI result or human draft when its checkpoint fails."""
    saved = save_active()
    if saved:
        # Covers a save that had nothing to commit, such as a closed project.
        st.session_state.get(_KEY, {})[_UNSAVED_KEY] = False
    else:
        begin_result()
    return saved


def require_saved_results() -> None:
    if st.session_state.get(_KEY, {}).get("pending_ai_completion"):
        from features.workflow.run_controls import recover_pending
        recover_pending()
    if not has_unsaved_results():
        return
    conflict = _store().get(_CONFLICT_KEY)
    st.error("Results or drafts are waiting to be saved. " + (
        "Another session changed this project; download your unsaved work before reloading."
        if conflict else "Keep this session open and retry saving, or download your work below."
    ))
    if st.button("Retry saving results", type="primary", disabled=bool(conflict)) and save_result():
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
            st.rerun()
    if st.button("Close project without saving", disabled=not discard):
        clear_session()
        set_active(None, None)
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


def pending_pdf_deletions() -> list[dict]:
    return _store().setdefault("pending_pdf_deletions", [])


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
    base[f"{prefix}: Source run ID"] = [
        cell(i, lambda p, s: s.get("source_run_id", "")) for i in n
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


def decision_history_dataframe() -> pd.DataFrame:
    """Every archived screening outcome with its decision, oldest first per paper."""
    columns = ["Paper", "Title", "DOI", "Stage", "Archived at", "AI verdict", "AI reason or error",
               "Decision", "Human verdict", "Provider", "Model", "Criteria hash", "Source run ID"]
    rows = []
    for number, paper in enumerate(papers(), start=1):
        for stage in (STAGE_ABSTRACT, STAGE_FULLTEXT):
            for entry in (paper.get("stages") or {}).get(stage, {}).get("history", []):
                rows.append({
                    "Paper": number, "Title": paper.get("title", ""), "DOI": paper.get("doi", ""),
                    "Stage": STAGE_TITLES[stage], "Archived at": entry.get("archived_at") or "",
                    "AI verdict": VERDICT_LABELS.get(entry.get("ai_verdict"), ""),
                    "AI reason or error": entry.get("ai_reason") or entry.get("ai_error") or "",
                    "Decision": entry.get("decision") or "",
                    "Human verdict": VERDICT_LABELS.get(entry.get("human_verdict"), ""),
                    "Provider": entry.get("provider") or "", "Model": entry.get("model") or "",
                    "Criteria hash": entry.get("criteria_hash") or entry.get("review_criteria_hash") or "",
                    "Source run ID": entry.get("source_run_id") or "",
                })
    return pd.DataFrame(rows, columns=columns)


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
