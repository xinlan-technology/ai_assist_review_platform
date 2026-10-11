"""Regression tests for core/llm.py response parsing."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from core import llm
from core.llm import InvalidModelResponse, _extract_json


def test_error_messages_do_not_persist_the_api_key_and_are_bounded():
    key = "test-private-key"
    error = RuntimeError(f"Failed request using {key}: " + "x" * 5000)
    message = llm.error_message(error, key)
    assert key not in message
    assert "[redacted]" in message
    assert len(message) == 4000


@pytest.mark.parametrize("padding", [" ", "\t", "\r\n", "\u00a0"])
def test_error_messages_redact_raw_and_sdk_normalized_keys(padding):
    key = "test-private-key"
    entered = padding + key + padding
    error = RuntimeError(f"Raw: {entered}; SDK: {key}; JSON: {json.dumps(entered)}")
    message = llm.error_message(error, entered)
    assert key not in message
    assert message.count("[redacted]") == 3


@pytest.mark.parametrize("key", ["", " ", " \t\n"])
def test_error_messages_with_no_usable_key_keep_the_message_bounded(key):
    assert llm.error_message(RuntimeError("Request failed" + "x" * 5000), key) == (
        "Request failed" + "x" * 5000
    )[:4000]


@pytest.mark.parametrize("provider", ["OpenAI", "Anthropic"])
@pytest.mark.parametrize("pdf", [False, True])
@pytest.mark.parametrize("failure", ["server_error", "timeout"])
def test_paid_calls_never_retry_an_ambiguous_sdk_failure(monkeypatch, provider, pdf, failure):
    import anthropic
    import httpx
    import openai

    module = openai if provider == "OpenAI" else anthropic
    name = provider
    client_type = getattr(module, name)
    requests = []

    def respond(request):
        requests.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("Synthetic timeout", request=request)
        return httpx.Response(500, json={"error": {
            "type": "api_error", "message": "Synthetic server failure",
        }}, headers={"retry-after-ms": "1"})

    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        def make_client(**kwargs):
            assert kwargs["max_retries"] == 0
            return client_type(base_url="https://example.invalid", http_client=http_client, **kwargs)

        monkeypatch.setattr(module, name, make_client)
        expected_error = module.APITimeoutError if failure == "timeout" else module.InternalServerError
        with pytest.raises(expected_error):
            args = (provider, "synthetic-model", "synthetic-key", "system", "user")
            if pdf:
                llm.call_pdf_structured(*args, b"%PDF-synthetic", "paper.pdf")
            else:
                llm.call_structured(*args)
    assert len(requests) == 1


@pytest.mark.parametrize("pdf", [False, True])
def test_dispatch_normalizes_keys_before_provider_calls(monkeypatch, pdf):
    seen = []

    def fake(model, api_key, *args, **kwargs):
        seen.append(api_key)
        return {"verdict": "include", "reason": "r"}

    dispatch = llm._PDF_DISPATCH if pdf else llm._DISPATCH
    monkeypatch.setitem(dispatch, "Test", fake)
    args = ("Test", "m", " \t test-private-key \r\n", "system", "user")
    if pdf:
        llm.call_pdf_structured(*args, b"%PDF-original", "paper.pdf")
    else:
        llm.call_structured(*args)
    assert seen == ["test-private-key"]


@pytest.mark.parametrize("key", [" \t test-private-key \n", " \t\n", "test-private\t-key\r\n"])
def test_model_controls_returns_a_normalized_key(monkeypatch, key):
    from core import ui

    fake_st = SimpleNamespace(
        subheader=lambda *args, **kwargs: None,
        selectbox=lambda label, options, **kwargs: options[0],
        text_input=lambda *args, **kwargs: key,
        caption=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(ui, "st", fake_st)
    # Whitespace anywhere in a pasted key is noise, including inside it.
    assert ui.model_controls()[2] == "".join(key.split())


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


@pytest.mark.parametrize("separator", [" Correction: ", ", ", "\nFinal answer:\n"])
@pytest.mark.parametrize("second", ['{"verdict":"exclude","reason":"final"}', '[{"a":2}]'])
def test_prose_cannot_hide_multiple_json_answers(separator, second):
    with pytest.raises(InvalidModelResponse, match="multiple JSON"):
        _extract_json('{"verdict":"include","reason":"draft"}' + separator + second)


@pytest.mark.parametrize("tail", [
    ' Page 2 of 10.', ' A closing brace } is fine.',
    ' The symbols "{}" and "[]" are examples.',
    ' Quoted example "braces \\"{}\\" remain quoted".',
])
def test_prose_tail_does_not_mistake_quoted_braces_or_numbers_for_answers(tail):
    assert _extract_json('Result: {"a":{"reason":"Use { or } in [text]"}}' + tail) == {
        "a": {"reason": "Use { or } in [text]"},
    }


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
