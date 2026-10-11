"""Pure per-paper extraction state with source and specification provenance."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json

from .schema import spec_hash, validate_answer, validate_response


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get(paper: dict) -> dict:
    return paper.setdefault("stages", {}).setdefault("extraction", {})


def form_nonce(record: dict) -> int:
    """Form revision changes with AI results, errors, or archives, but not drafts."""
    return int(record.get("form_nonce", 0))


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
        record["history"] = history


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
    get(paper).update(stamp | {
        "ai_error": str(message)[:4000], "ai_error_kind": kind, "completed_at": _now(),
    })


def _same_answer(question: dict, left: dict, right: dict) -> bool:
    a, b = left.get("values", []), right.get("values", [])
    if question["type"] == "multiple_choice":
        a, b = sorted(a), sorted(b)
    return a == b and left.get("other_text", "") == right.get("other_text", "")


def candidate_answer(
    run: dict, spec: dict, source_hash: str, question_id: str,
    page_count: int | None = None,
) -> dict:
    """Read one valid, current proposal without changing the review or run."""
    result = run.get("result", {})
    if not isinstance(result, dict) or not result:
        raise ValueError("This run has no completed extraction answer.")
    if (not source_hash or result.get("spec_hash") != spec_hash(spec)
            or result.get("source_sha256") != source_hash
            or result.get("prompt_version") != spec["prompt_version"]):
        raise ValueError("This run uses a different extraction setup or PDF.")
    question = next((q for q in spec["questions"] if q["id"] == question_id), None)
    if question is None:
        raise ValueError("Unknown extraction question.")
    if result.get("ai_error") or question_id in result.get("field_errors", {}):
        raise ValueError("This run has no valid proposal for this question.")
    return validate_answer(question, result.get("ai_answers", {}).get(question_id), page_count)


def save_review(
    paper: dict, spec: dict, source_hash: str, answers: dict,
    page_count: int | None = None, confirm: bool = False,
    source_run_ids: dict | None = None, source_answers: dict | None = None,
    source_metadata: dict | None = None,
) -> None:
    stamp = _stamp(spec, source_hash)
    if is_stale(paper, spec, source_hash):
        raise ValueError("Archive outdated extraction before reviewing it.")
    valid, errors = validate_response({"answers": answers}, spec, page_count)
    if confirm and errors:
        raise ValueError("Complete all answers before confirming: " + "; ".join(errors.values()))
    record = get(paper)
    question_ids = {q["id"] for q in spec["questions"]}
    sources = source_run_ids if source_run_ids is not None else record.get("source_run_ids", {})
    if (not isinstance(sources, dict) or set(sources) - question_ids
            or any(not isinstance(value, str) or not value.strip() or len(value) > 200
                   for value in sources.values())):
        raise ValueError("Answer source runs must identify configured questions and valid run IDs.")
    proposed = (record.get("source_answers", record.get("ai_answers", {}))
                if source_answers is None else source_answers)
    if not isinstance(proposed, dict) or set(proposed) - question_ids:
        raise ValueError("Answer sources must identify configured questions.")
    proposed = {q["id"]: validate_answer(q, proposed[q["id"]], page_count)
                for q in spec["questions"] if q["id"] in proposed}
    if source_run_ids is None and not sources and record.get("source_run_id"):
        sources = {qid: record["source_run_id"] for qid in answers if qid in proposed}
    provenance = (record.get("source_metadata", {})
                  if source_metadata is None else source_metadata)
    if not isinstance(provenance, dict) or set(provenance) - question_ids:
        raise ValueError("Source metadata must identify configured questions.")
    if source_metadata is None:
        provenance = {qid: value for qid, value in provenance.items()
                      if isinstance(value, dict) and value.get("id") == sources.get(qid)}
        for qid, source_id in sources.items():
            if qid not in provenance and source_id == record.get("source_run_id"):
                provenance[qid] = {
                    "id": source_id, "provider": record.get("provider", ""),
                    "model": record.get("model", ""), "completed_at": record.get("completed_at"),
                }
    for qid, item in provenance.items():
        if (not isinstance(item, dict)
                or set(item) != {"id", "provider", "model", "completed_at"}
                or qid not in sources or item["id"] != sources[qid]
                or any(not isinstance(item[field], str) or len(item[field]) > 200
                       for field in ("id", "provider", "model"))
                or (item["completed_at"] is not None and (
                    not isinstance(item["completed_at"], str) or len(item["completed_at"]) > 100))):
            raise ValueError("Source metadata must match its answer run and contain only model audit fields.")
    if record.get("review_state") == "confirmed":
        retained = deepcopy({key: value for key, value in record.items()
                             if key not in _META_KEYS})
        archive(paper)
        record.update(retained)
    record.update(stamp | {
        "final_answers": valid if confirm else _draft_answers(answers),
        "review_errors": errors,
        "review_state": "confirmed" if confirm else "draft",
        "source_run_ids": deepcopy(sources),
        "source_answers": deepcopy(proposed),
        "source_metadata": deepcopy(provenance),
        "decisions": {
            q["id"]: (
                "human" if q["id"] not in proposed
                else "accepted" if _same_answer(q, answer, proposed[q["id"]])
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
