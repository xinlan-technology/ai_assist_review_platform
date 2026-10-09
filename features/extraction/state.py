"""Pure per-paper extraction state with source and specification provenance."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json

from .schema import spec_hash, validate_response


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get(paper: dict) -> dict:
    return paper.setdefault("stages", {}).setdefault("extraction", {})


def form_nonce(record: dict) -> int:
    """Form revision changes with AI results, errors, or archives, but not drafts."""
    return int(record.get("form_nonce", 0))


def _bump_form(record: dict) -> None:
    record["form_nonce"] = form_nonce(record) + 1


_META_KEYS = {"history", "form_nonce"}


def _has_current(record: dict) -> bool:
    return any(key not in _META_KEYS for key in record)


def archive(paper: dict) -> None:
    record = get(paper)
    history = list(record.get("history", []))
    if _has_current(record):
        snapshot = deepcopy({key: value for key, value in record.items()
                             if key not in _META_KEYS})
        snapshot["archived_at"] = _now()
        history.append(snapshot)
    nonce = form_nonce(record) + 1
    record.clear()
    record["form_nonce"] = nonce
    if history:
        kept = history[-5:]
        if not any(entry.get("review_state") == "confirmed" for entry in kept):
            confirmed = [entry for entry in history if entry.get("review_state") == "confirmed"]
            if confirmed:
                kept = [confirmed[-1]] + kept[-4:]
        record["history"] = kept


def is_stale(paper: dict, spec: dict, source_hash: str) -> bool:
    record = get(paper)
    return _has_current(record) and (
        record.get("spec_hash") != spec_hash(spec)
        or not source_hash
        or record.get("source_sha256") != source_hash
    )


def _stamp(spec: dict, source_hash: str) -> dict:
    if not isinstance(source_hash, str) or not source_hash.strip():
        raise ValueError("Attach a PDF before extracting or reviewing answers.")
    return {
        "spec": deepcopy(spec), "spec_hash": spec_hash(spec),
        "source_sha256": source_hash, "prompt_version": spec["prompt_version"],
    }


def _safe_raw(value: object) -> object:
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, default=str)
    except (TypeError, ValueError, RecursionError):
        return str(value)[:12000]
    return json.loads(encoded) if len(encoded) <= 12000 else encoded[:12000]


def _draft_answers(answers: dict) -> dict:
    return {
        qid: {key: _safe_raw(value) for key, value in answer.items()}
        if isinstance(answer, dict) else _safe_raw(answer)
        for qid, answer in answers.items()
    }


def set_ai_result(
    paper: dict, spec: dict, source_hash: str, result: dict,
    provider: str, model: str, page_count: int | None = None,
) -> None:
    stamp = _stamp(spec, source_hash)
    answers, errors = validate_response(result, spec, page_count)
    archive(paper)
    _bump_form(get(paper))
    get(paper).update(stamp | {
        "ai_answers": answers, "field_errors": errors,
        "invalid_answers": {qid: _safe_raw(result["answers"].get(qid)) for qid in errors},
        "provider": provider, "model": model, "completed_at": _now(),
        "page_count": page_count,
    })


def set_error(paper: dict, spec: dict, source_hash: str, message: str, kind: str) -> None:
    stamp = _stamp(spec, source_hash)
    if kind not in {"call_failed", "invalid_response"}:
        raise ValueError("Unknown extraction error kind.")
    archive(paper)
    _bump_form(get(paper))
    get(paper).update(stamp | {
        "ai_error": str(message)[:4000], "ai_error_kind": kind, "completed_at": _now(),
    })


def _same_answer(question: dict, left: dict, right: dict) -> bool:
    a, b = left.get("values", []), right.get("values", [])
    if question["type"] == "multiple_choice":
        a, b = sorted(a), sorted(b)
    return a == b and left.get("other_text", "") == right.get("other_text", "")


def save_review(
    paper: dict, spec: dict, source_hash: str, answers: dict,
    page_count: int | None = None, confirm: bool = False,
) -> None:
    stamp = _stamp(spec, source_hash)
    if is_stale(paper, spec, source_hash):
        raise ValueError("Archive outdated extraction before reviewing it.")
    valid, errors = validate_response({"answers": answers}, spec, page_count)
    if confirm and errors:
        raise ValueError("Complete all answers before confirming: " + "; ".join(errors.values()))
    record = get(paper)
    if record.get("review_state") == "confirmed":
        retained = deepcopy({key: value for key, value in record.items()
                             if key not in _META_KEYS})
        archive(paper)
        record.update(retained)
    record.update(stamp | {
        "final_answers": valid if confirm else _draft_answers(answers),
        "review_errors": errors,
        "review_state": "confirmed" if confirm else "draft",
        "decisions": {
            q["id"]: (
                "human" if q["id"] not in record.get("ai_answers", {})
                else "accepted" if _same_answer(q, answer, record["ai_answers"][q["id"]])
                else "edited"
            )
            for q in spec["questions"] if (answer := valid.get(q["id"])) is not None
        },
        "reviewed_at": _now() if confirm else None,
        "draft_saved_at": None if confirm else _now(),
        "page_count": page_count,
    })


def confirmed_answers(paper: dict, spec: dict, source_hash: str) -> dict:
    record = get(paper)
    if record.get("review_state") != "confirmed" or is_stale(paper, spec, source_hash):
        return {}
    try:
        answers, errors = validate_response({"answers": record.get("final_answers")}, spec, record.get("page_count"))
    except ValueError:
        return {}
    return {} if errors else deepcopy(answers)


def status(paper: dict, spec: dict, source_hash: str) -> str:
    record = get(paper)
    if is_stale(paper, spec, source_hash):
        return "Outdated"
    if record.get("review_state") == "confirmed" and confirmed_answers(paper, spec, source_hash):
        return "Confirmed"
    if record.get("review_state") == "draft":
        return "Draft"
    if record.get("field_errors") or record.get("ai_error_kind") == "invalid_response":
        return "Invalid"
    if record.get("ai_error_kind") or record.get("ai_error"):
        return "Error"
    if record.get("ai_answers"):
        return "Needs review"
    return "Pending"


def pending(paper: dict, spec: dict, source_hash: str) -> bool:
    record = get(paper)
    if is_stale(paper, spec, source_hash) or record.get("review_state"):
        return False
    if record.get("field_errors") or "ai_answers" in record:
        return False
    return not record.get("ai_error_kind") or record["ai_error_kind"] == "call_failed"
