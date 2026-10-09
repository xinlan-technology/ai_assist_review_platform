from __future__ import annotations

from core.llm import InvalidModelResponse, call_pdf_structured, call_structured
from features.screening import prompts
from features.workflow.state import STAGE_ABSTRACT, STAGE_FULLTEXT, STAGE_VERDICTS


def _parse_verdict(result: dict, allowed: list[str]) -> dict:
    verdict = str(result.get("verdict", "") or "").strip().lower()
    if verdict not in allowed:
        raise InvalidModelResponse(
            f"Model returned an unrecognized verdict: {result!r}"
        )
    raw_reason = result.get("reason")
    if not isinstance(raw_reason, str) or not raw_reason.strip():
        raise InvalidModelResponse(
            f"Model returned a missing or invalid reason: {result!r}"
        )
    reason = raw_reason.strip()
    return {"verdict": verdict, "reason": reason}


def judge_abstract(
    provider: str,
    model: str,
    api_key: str,
    criteria: str,
    title: str,
    abstract: str,
) -> dict:
    """Return {verdict, reason} for one title and abstract.

    InvalidModelResponse requires explicit retry or human review.
    """
    user = prompts.build_abstract_prompt(criteria, title or "", abstract or "")
    result = call_structured(provider, model, api_key, prompts.SYSTEM_PROMPT, user)
    return _parse_verdict(result, STAGE_VERDICTS[STAGE_ABSTRACT])


def judge_fulltext(
    provider: str,
    model: str,
    api_key: str,
    criteria: str,
    title: str,
    pdf_bytes: bytes,
    filename: str,
) -> dict:
    """Screen one original PDF. Full text has only include/exclude outcomes."""
    user = prompts.build_fulltext_prompt(criteria, title or "")
    result = call_pdf_structured(
        provider,
        model,
        api_key,
        prompts.SYSTEM_PROMPT,
        user,
        pdf_bytes,
        filename,
    )
    return _parse_verdict(result, STAGE_VERDICTS[STAGE_FULLTEXT])
