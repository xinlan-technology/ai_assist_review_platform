from __future__ import annotations

import json
import re
import base64

PROVIDERS: dict[str, list[str]] = {
    "OpenAI": ["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini", "gpt-4o"],
    "Anthropic": ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"],
    "Google": ["gemini-2.5-pro", "gemini-2.5-flash"],
}

PROVIDER_KEY_HELP: dict[str, str] = {
    "OpenAI": "Create an API key at platform.openai.com (starts with sk-)",
    "Anthropic": "Create an API key at console.anthropic.com (starts with sk-ant-)",
    "Google": "Create an API key at aistudio.google.com",
}

MAX_INLINE_PDF_BYTES = 19 * 1024 * 1024

# Reserve output room for Gemini Pro reasoning; Flash can disable reasoning.
GOOGLE_PRO_THINKING_BUDGET = 128
GOOGLE_PRO_HEADROOM = 2048


def _google_limits(model: str, max_output_tokens: int) -> tuple[int, int]:
    """Return (output cap, thinking budget) for a Gemini request."""
    if "pro" in (model or "").lower():
        return (
            max(int(max_output_tokens) + GOOGLE_PRO_HEADROOM, 4096),
            GOOGLE_PRO_THINKING_BUDGET,
        )
    return int(max_output_tokens), 0

PDF_SCREENING_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["include", "exclude"]},
        "reason": {"type": "string", "minLength": 1},
    },
    "required": ["verdict", "reason"],
    "additionalProperties": False,
}


class InvalidModelResponse(ValueError):
    """A nonconforming reply requiring explicit retry or manual review."""


def error_message(error: Exception, api_key: str) -> str:
    message = str(error)
    return (message.replace(api_key, "[redacted]") if api_key else message)[:4000]


def _unique_json_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise InvalidModelResponse(f"The model repeated JSON key {key!r}")
        result[key] = value
    return result


def _invalid_json_constant(value: str):
    raise InvalidModelResponse(f"The model returned a non-JSON value: {value}")


def _extract_json(text: str) -> dict:
    """Parse a JSON object; classify invalid output separately from API failures."""
    if not text:
        raise InvalidModelResponse("The model returned no content")
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()
    decoder = json.JSONDecoder(object_pairs_hook=_unique_json_object,
                               parse_constant=_invalid_json_constant)
    try:
        parsed = decoder.decode(cleaned)
    except json.JSONDecodeError:
        pass  # Fall back to extracting an object from prose.
    else:
        if isinstance(parsed, dict):
            return parsed
        # Never salvage a nested object from valid non-object JSON.
        raise InvalidModelResponse("The model's response is not a JSON object")
    if cleaned.startswith("["):
        try:
            _, end = decoder.raw_decode(cleaned)
        except json.JSONDecodeError as exc:
            raise InvalidModelResponse("The model returned an incomplete JSON array") from exc
        cleaned = cleaned[end:]
    start = cleaned.find("{")
    if start != -1:
        try:
            parsed, end = decoder.raw_decode(cleaned[start:])
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(parsed, dict):
                if cleaned[start + end:].lstrip().startswith(("{", "[")):
                    raise InvalidModelResponse("The model returned multiple JSON values")
                return parsed
    raise InvalidModelResponse("The model's response is not a JSON object")


_TRUNCATION_REASONS = {"length", "max_tokens", "incomplete"}


def _check_finished(provider: str, reason: object, expected: str, limit: int | None = None) -> None:
    reason = getattr(reason, "value", reason)
    if not reason or reason == expected:
        return
    if str(reason).strip().lower() in _TRUNCATION_REASONS:
        cap = f" of {limit} tokens" if limit else ""
        raise InvalidModelResponse(
            f"{provider} reached its output limit{cap} before completing the JSON "
            "answer. Raise the output limit or choose another model."
        )
    raise InvalidModelResponse(f"{provider} did not finish normally ({reason})")


def _anthropic_result(response, limit: int | None = None) -> dict:
    _check_finished("Anthropic", getattr(response, "stop_reason", None), "end_turn", limit)
    text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
    return _extract_json(text)


def _google_result(response, limit: int | None = None) -> dict:
    for candidate in getattr(response, "candidates", None) or []:
        _check_finished("Google", getattr(candidate, "finish_reason", None), "STOP", limit)
    feedback = getattr(response, "prompt_feedback", None)
    reason = getattr(feedback, "block_reason", None)
    reason = getattr(reason, "value", reason)
    if reason and reason != "BLOCKED_REASON_UNSPECIFIED":
        raise InvalidModelResponse("Google blocked the request")
    return _extract_json(response.text or "")


def _call_openai(model: str, api_key: str, system: str, user: str) -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    if not resp.choices:
        raise InvalidModelResponse("OpenAI returned no choices")
    choice = resp.choices[0]
    _check_finished("OpenAI", getattr(choice, "finish_reason", None), "stop")
    if getattr(choice.message, "refusal", None):
        raise InvalidModelResponse("OpenAI declined the screening request")
    return _extract_json(choice.message.content or "")


def _call_anthropic(model: str, api_key: str, system: str, user: str) -> dict:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return _anthropic_result(resp)


def _call_google(model: str, api_key: str, system: str, user: str) -> dict:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    # Leave reasoning uncapped here; the PDF path sets an explicit budget.
    resp = client.models.generate_content(
        model=model,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
        ),
    )
    return _google_result(resp)


def _pdf_data_uri(pdf_bytes: bytes) -> str:
    encoded = base64.b64encode(pdf_bytes).decode("ascii")
    return f"data:application/pdf;base64,{encoded}"


def _call_openai_pdf(
    model: str,
    api_key: str,
    system: str,
    user: str,
    pdf_bytes: bytes,
    filename: str,
    *,
    schema: dict | None = None,
    schema_name: str = "fulltext_screening_result",
    max_output_tokens: int = 512,
) -> dict:
    """Send the original PDF through the multimodal Responses API."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    resp = client.responses.create(
        model=model,
        max_output_tokens=max_output_tokens,
        store=False,
        text={
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "schema": schema if schema is not None else PDF_SCREENING_SCHEMA,
                "strict": True,
            }
        },
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "filename": filename or "paper.pdf",
                        "file_data": _pdf_data_uri(pdf_bytes),
                    },
                    {"type": "input_text", "text": user},
                ],
            },
        ],
    )
    _check_finished("OpenAI", getattr(resp, "status", None), "completed", max_output_tokens)
    for item in getattr(resp, "output", None) or []:
        for block in getattr(item, "content", None) or []:
            if getattr(block, "type", None) == "refusal":
                raise InvalidModelResponse("OpenAI declined the extraction or screening request")
    return _extract_json(resp.output_text or "")


def _call_anthropic_pdf(
    model: str,
    api_key: str,
    system: str,
    user: str,
    pdf_bytes: bytes,
    filename: str,
    *,
    schema: dict | None = None,
    schema_name: str = "fulltext_screening_result",
    max_output_tokens: int = 512,
) -> dict:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=max_output_tokens,
        system=system,
        output_config={
            "format": {
                "type": "json_schema",
                "schema": anthropic.transform_schema(
                    schema if schema is not None else PDF_SCREENING_SCHEMA
                ),
            }
        },
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": base64.b64encode(pdf_bytes).decode("ascii"),
                        },
                    },
                    {"type": "text", "text": user},
                ],
            }
        ],
    )
    return _anthropic_result(resp, max_output_tokens)


def _call_google_pdf(
    model: str,
    api_key: str,
    system: str,
    user: str,
    pdf_bytes: bytes,
    filename: str,
    *,
    schema: dict | None = None,
    schema_name: str = "fulltext_screening_result",
    max_output_tokens: int = 512,
) -> dict:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    pdf_part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
    cap, thinking_budget = _google_limits(model, max_output_tokens)
    resp = client.models.generate_content(
        model=model,
        contents=[pdf_part, user],
        config=types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=schema if schema is not None else PDF_SCREENING_SCHEMA,
            max_output_tokens=cap,
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
        ),
    )
    return _google_result(resp, cap)


_DISPATCH = {
    "OpenAI": _call_openai,
    "Anthropic": _call_anthropic,
    "Google": _call_google,
}

_PDF_DISPATCH = {
    "OpenAI": _call_openai_pdf,
    "Anthropic": _call_anthropic_pdf,
    "Google": _call_google_pdf,
}


def call_structured(provider: str, model: str, api_key: str, system: str, user: str) -> dict:
    if provider not in _DISPATCH:
        raise ValueError(f"Unknown provider: {provider}")
    return _DISPATCH[provider](model, api_key, system, user)


def call_pdf_structured(
    provider: str,
    model: str,
    api_key: str,
    system: str,
    user: str,
    pdf_bytes: bytes,
    filename: str,
    *,
    schema: dict | None = None,
    schema_name: str = "fulltext_screening_result",
    max_output_tokens: int = 512,
) -> dict:
    """Call a provider with the original PDF and return one JSON object."""
    if provider not in _PDF_DISPATCH:
        raise ValueError(f"Unknown provider: {provider}")
    if not isinstance(pdf_bytes, (bytes, bytearray)) or not pdf_bytes:
        raise ValueError("The PDF is empty")
    if len(pdf_bytes) > MAX_INLINE_PDF_BYTES:
        raise ValueError("The PDF exceeds the 19 MB inline-processing limit")
    if type(max_output_tokens) is not int or max_output_tokens < 1:
        raise ValueError("The output token limit must be a positive integer")
    options = {}
    if schema is not None:
        options["schema"] = schema
    if schema_name != "fulltext_screening_result":
        options["schema_name"] = schema_name
    if max_output_tokens != 512:
        options["max_output_tokens"] = max_output_tokens
    return _PDF_DISPATCH[provider](
        model, api_key, system, user, bytes(pdf_bytes), filename, **options
    )


def pdf_input_issue(
    provider: str,
    model: str,
    file_size: int | None,
    page_count: int | None,
) -> str | None:
    """Return why a stored PDF should not be sent to the selected model."""
    if int(file_size or 0) > MAX_INLINE_PDF_BYTES:
        return "larger than the 19 MB AI limit"
    if provider == "Anthropic" and page_count:
        page_limit = 100 if "haiku" in model.lower() else 600
        if int(page_count) > page_limit:
            return f"over the {page_limit}-page limit for {model}"
    if provider == "Google" and page_count and int(page_count) > 1000:
        return "over the 1000-page limit for Google"
    return None
