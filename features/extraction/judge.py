"""Extract all configured fields in one PDF request."""
from __future__ import annotations

from core.llm import InvalidModelResponse, call_pdf_structured
from features.extraction import prompts
from features.extraction.schema import build_spec, provider_issue, response_schema


def extract_pdf(
    provider: str,
    model: str,
    api_key: str,
    spec: dict,
    pdf_bytes: bytes,
    filename: str,
) -> dict:
    spec = build_spec(spec.get("instructions", ""), spec.get("questions", []))
    issue = provider_issue(provider, spec)
    if issue:
        raise ValueError(issue)
    result = call_pdf_structured(
        provider,
        model,
        api_key,
        prompts.SYSTEM_PROMPT,
        prompts.build_extraction_prompt(spec),
        pdf_bytes,
        filename,
        schema=response_schema(spec),
        schema_name="data_extraction_result",
        max_output_tokens=min(12000, 1200 + 300 * len(spec["questions"])),
    )
    if not isinstance(result, dict) or not isinstance(result.get("answers"), dict):
        raise InvalidModelResponse("The extraction response must contain an answers object")
    return result
