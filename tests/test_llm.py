"""Regression tests for core/llm.py response parsing."""
from __future__ import annotations


from core import llm
from core.llm import InvalidModelResponse, _extract_json


def test_error_messages_do_not_persist_the_api_key_and_are_bounded():
    key = "test-private-key"
    error = RuntimeError(f"Failed request using {key}: " + "x" * 5000)
    message = llm.error_message(error, key)
    assert key not in message
    assert "[redacted]" in message
    assert len(message) == 4000


def test_plain_object():
    assert _extract_json('{"verdict": "include", "reason": "r"}') == {
        "verdict": "include", "reason": "r"
    }


def test_fenced_object():
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_object_embedded_in_prose():
    assert _extract_json('Sure! Here you go: {"a": {"b": 2}} hope that helps') == {
        "a": {"b": 2}
    }


def test_non_object_json_is_invalid_response():
    # Do not salvage nested objects or classify invalid JSON as retryable.
    for bad in ("[1, 2]", '[{"a": 1}]', '```json\n[{"a": 1}]\n```',
                '"just a string"', "null", "42", "", "no json here",
                "{broken: json}"):
        try:
            _extract_json(bad)
        except InvalidModelResponse:
            pass
        else:
            raise AssertionError(f"{bad!r} must raise InvalidModelResponse")


def test_array_before_object_still_finds_object():
    assert _extract_json('[1,2] then {"a": 1}') == {"a": 1}


def test_evidence_with_braces_and_escaped_quotes_in_prose():
    text = 'Result: {"quote": "Use } for category \\"A\\"; { for B", "page": 2} done'
    assert _extract_json(text) == {"quote": 'Use } for category "A"; { for B', "page": 2}


def test_malformed_containers_duplicate_keys_and_nonfinite_values_are_invalid():
    for text in (
        '[{"verdict":"include","reason":"r"}',
        '{"verdict":"include","verdict":"exclude","reason":"r"}',
        'Result: {"a": 1, "a": 2}', '{"a": NaN}', '{"a": Infinity}',
        '{"a": -Infinity}', '{"a": 1}{"a": 2}',
    ):
        try:
            _extract_json(text)
        except InvalidModelResponse:
            pass
        else:
            raise AssertionError(f"{text!r} must be rejected")


def test_object_in_a_leading_array_is_not_used_as_the_prose_answer():
    assert _extract_json('[{"unrelated":1}] then {"a":2}') == {"a": 2}
    try:
        _extract_json('[{"a":1}] some prose but no answer')
    except InvalidModelResponse:
        pass
    else:
        raise AssertionError("Do not salvage an object from a leading array")


def test_pdf_dispatch_preserves_original_bytes():
    seen = {}

    def fake(model, api_key, system, user, pdf_bytes, filename):
        seen.update(
            model=model,
            api_key=api_key,
            system=system,
            user=user,
            pdf_bytes=pdf_bytes,
            filename=filename,
        )
        return {"verdict": "include", "reason": "r"}

    old = llm._PDF_DISPATCH.get("Test")
    llm._PDF_DISPATCH["Test"] = fake
    try:
        result = llm.call_pdf_structured(
            "Test", "m", "secret", "system", "user", b"%PDF-original", "paper.pdf"
        )
    finally:
        if old is None:
            llm._PDF_DISPATCH.pop("Test", None)
        else:
            llm._PDF_DISPATCH["Test"] = old
    assert result["verdict"] == "include"
    assert seen["pdf_bytes"] == b"%PDF-original"
    assert seen["filename"] == "paper.pdf"


def test_pdf_dispatch_rejects_empty_and_oversized_before_provider_call():
    for bad in (b"", "not bytes", b"x" * (llm.MAX_INLINE_PDF_BYTES + 1)):
        try:
            llm.call_pdf_structured("OpenAI", "m", "k", "s", "u", bad, "p.pdf")
        except ValueError:
            pass
        else:
            raise AssertionError("invalid PDF payload must fail before calling the provider")
