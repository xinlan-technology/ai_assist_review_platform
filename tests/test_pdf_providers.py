"""Provider payload tests. SDKs are mocked; these tests never call the network."""
from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest

from core import llm


PDF = b"%PDF-1.7\noriginal bytes"


@pytest.mark.parametrize("provider,reason", [
    ("OpenAI", "length"), ("OpenAI", "content_filter"),
    ("Anthropic", "max_tokens"), ("Anthropic", "refusal"),
    ("Google", "MAX_TOKENS"), ("Google", "SAFETY"),
])
def test_text_providers_reject_unfinished_json(monkeypatch, provider, reason):
    import openai
    import anthropic
    from google import genai

    text = '{"verdict":"include","reason":"r"}'
    response = SimpleNamespace(
        choices=[SimpleNamespace(finish_reason=reason, message=SimpleNamespace(content=text, refusal=None))],
        stop_reason=reason, content=[SimpleNamespace(type="text", text=text)],
        candidates=[SimpleNamespace(finish_reason=reason)], text=text,
    )
    monkeypatch.setattr(openai, "OpenAI", lambda api_key, **kwargs: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response))))
    monkeypatch.setattr(anthropic, "Anthropic", lambda api_key, **kwargs: SimpleNamespace(
        messages=SimpleNamespace(create=lambda **kwargs: response)))
    monkeypatch.setattr(genai, "Client", lambda api_key: SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **kwargs: response)))
    with pytest.raises(llm.InvalidModelResponse):
        llm.call_structured(provider, "m", "k", "s", "u")


@pytest.mark.parametrize("choices", [[], [SimpleNamespace(
    finish_reason="stop",
    message=SimpleNamespace(content='{"verdict":"include","reason":"r"}', refusal="Declined"),
)]])
def test_openai_text_rejects_no_choice_or_refusal(monkeypatch, choices):
    import openai

    monkeypatch.setattr(openai, "OpenAI", lambda api_key, **kwargs: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(choices=choices)))))
    with pytest.raises(llm.InvalidModelResponse):
        llm.call_structured("OpenAI", "m", "k", "s", "u")


def test_google_unspecified_block_reason_does_not_reject_completed_answer():
    response = SimpleNamespace(
        candidates=[SimpleNamespace(finish_reason="STOP")],
        prompt_feedback=SimpleNamespace(block_reason="BLOCKED_REASON_UNSPECIFIED"),
        text='{"verdict":"include","reason":"r"}',
    )
    assert llm._google_result(response)["verdict"] == "include"


def test_openai_pdf_payload(monkeypatch):
    import openai

    seen = {}

    class Responses:
        def create(self, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(output_text='{"verdict":"include","reason":"r"}')

    fake_client = SimpleNamespace(responses=Responses())
    monkeypatch.setattr(openai, "OpenAI", lambda api_key, **kwargs: fake_client)

    result = llm._call_openai_pdf("gpt", "key", "system", "prompt", PDF, "paper.pdf")
    assert result["verdict"] == "include"
    assert seen["input"][0]["role"] == "system"
    assert seen["text"]["format"]["type"] == "json_schema"
    assert seen["text"]["format"]["strict"] is True
    content = seen["input"][1]["content"]
    assert content[0]["type"] == "input_file"
    assert content[1] == {"type": "input_text", "text": "prompt"}
    encoded = content[0]["file_data"].split(",", 1)[1]
    assert base64.b64decode(encoded) == PDF


def test_anthropic_pdf_payload(monkeypatch):
    import anthropic

    seen = {}

    class Messages:
        def create(self, **kwargs):
            seen.update(kwargs)
            block = SimpleNamespace(type="text", text='{"verdict":"exclude","reason":"r"}')
            return SimpleNamespace(content=[block])

    fake_client = SimpleNamespace(messages=Messages())
    monkeypatch.setattr(anthropic, "Anthropic", lambda api_key, **kwargs: fake_client)

    result = llm._call_anthropic_pdf("claude", "key", "system", "prompt", PDF, "p.pdf")
    assert result["verdict"] == "exclude"
    assert seen["system"] == "system"
    assert seen["output_config"]["format"]["type"] == "json_schema"
    content = seen["messages"][0]["content"]
    assert content[0]["type"] == "document"
    assert content[0]["source"]["media_type"] == "application/pdf"
    assert base64.b64decode(content[0]["source"]["data"]) == PDF
    assert content[1] == {"type": "text", "text": "prompt"}


def test_google_pdf_payload(monkeypatch):
    from google import genai

    seen = {}

    class Models:
        def generate_content(self, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(text='{"verdict":"include","reason":"r"}')

    monkeypatch.setattr(genai, "Client", lambda api_key: SimpleNamespace(models=Models()))

    result = llm._call_google_pdf("gemini", "key", "system", "prompt", PDF, "p.pdf")
    assert result["verdict"] == "include"
    assert seen["contents"][1] == "prompt"
    part = seen["contents"][0]
    assert part.inline_data.mime_type == "application/pdf"
    assert part.inline_data.data == PDF
    assert seen["config"].system_instruction == "system"
    assert seen["config"].response_mime_type == "application/json"
    assert seen["config"].response_json_schema == llm.PDF_SCREENING_SCHEMA


@pytest.mark.parametrize("provider", ["OpenAI", "Anthropic", "Google"])
def test_pdf_providers_receive_dynamic_schema_and_token_limit(monkeypatch, provider):
    import openai
    import anthropic
    from google import genai

    seen = {}

    def fake_response(**kwargs):
        seen.update(kwargs)
        text = '{"answers": {}}'
        return SimpleNamespace(
            output_text=text, status="completed", output=[], stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text=text)], text=text,
            candidates=[SimpleNamespace(finish_reason="STOP")],
        )

    monkeypatch.setattr(openai, "OpenAI", lambda api_key, **kwargs: SimpleNamespace(
        responses=SimpleNamespace(create=fake_response)))
    monkeypatch.setattr(anthropic, "Anthropic", lambda api_key, **kwargs: SimpleNamespace(
        messages=SimpleNamespace(create=fake_response)))
    monkeypatch.setattr(genai, "Client", lambda api_key: SimpleNamespace(
        models=SimpleNamespace(generate_content=fake_response)))
    schema = {
        "type": "object", "properties": {"answers": {
            "type": "object", "properties": {}, "additionalProperties": False,
        }},
        "required": ["answers"], "additionalProperties": False,
    }
    assert llm.call_pdf_structured(
        provider, "model", "key", "system", "prompt", PDF, "study.pdf",
        schema=schema, schema_name="custom_extraction", max_output_tokens=4200,
    ) == {"answers": {}}
    if provider == "OpenAI":
        assert seen["text"]["format"]["schema"] == schema
        assert seen["text"]["format"]["name"] == "custom_extraction"
        assert seen["max_output_tokens"] == 4200
        assert seen["store"] is False
    elif provider == "Anthropic":
        assert seen["output_config"]["format"]["schema"] == schema
        assert seen["max_tokens"] == 4200
    else:
        assert seen["config"].response_json_schema == schema
        assert seen["config"].max_output_tokens == 4200


def test_anthropic_removes_unsupported_schema_constraints(monkeypatch):
    import anthropic
    from features.extraction.schema import build_spec, response_schema

    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(stop_reason="end_turn", content=[
            SimpleNamespace(type="text", text='{"answers": {}}')])

    monkeypatch.setattr(anthropic, "Anthropic", lambda api_key, **kwargs: SimpleNamespace(
        messages=SimpleNamespace(create=fake)))
    spec = build_spec("", [{
        "id": "q1", "text": "Ecosystem?", "type": "single_choice", "options": ["Forest"],
    }])
    schema = response_schema(spec)
    llm._call_anthropic_pdf("m", "k", "s", "u", PDF, "p.pdf", schema=schema)
    sent = seen["output_config"]["format"]["schema"]
    values = sent["properties"]["answers"]["properties"]["q1"]["properties"]["values"]
    assert "maxItems" not in values
    assert values["items"]["enum"] == ["Forest", "Other", "Not reported"]
    assert "maxItems" in schema["properties"]["answers"]["properties"]["q1"]["properties"]["values"]


@pytest.mark.parametrize("status", ["incomplete", "failed"])
def test_openai_rejects_unfinished_valid_json(monkeypatch, status):
    import openai

    response = SimpleNamespace(status=status, output_text='{"answers": {}}')
    monkeypatch.setattr(openai, "OpenAI", lambda api_key, **kwargs: SimpleNamespace(
        responses=SimpleNamespace(create=lambda **kwargs: response)))
    with pytest.raises(llm.InvalidModelResponse):
        llm._call_openai_pdf("m", "k", "s", "u", PDF, "p.pdf")


def test_openai_rejects_refusal_even_with_json(monkeypatch):
    import openai

    response = SimpleNamespace(
        status="completed", output_text='{"answers": {}}',
        output=[SimpleNamespace(content=[SimpleNamespace(type="refusal")])],
    )
    monkeypatch.setattr(openai, "OpenAI", lambda api_key, **kwargs: SimpleNamespace(
        responses=SimpleNamespace(create=lambda **kwargs: response)))
    with pytest.raises(llm.InvalidModelResponse):
        llm._call_openai_pdf("m", "k", "s", "u", PDF, "p.pdf")


@pytest.mark.parametrize("reason", ["max_tokens", "refusal", "model_context_window_exceeded"])
def test_anthropic_rejects_partial_or_refused_json(monkeypatch, reason):
    import anthropic

    response = SimpleNamespace(
        stop_reason=reason,
        content=[SimpleNamespace(type="text", text='{"answers": {}}')],
    )
    monkeypatch.setattr(anthropic, "Anthropic", lambda api_key, **kwargs: SimpleNamespace(
        messages=SimpleNamespace(create=lambda **kwargs: response)))
    with pytest.raises(llm.InvalidModelResponse):
        llm._call_anthropic_pdf("m", "k", "s", "u", PDF, "p.pdf")


@pytest.mark.parametrize("reason", ["MAX_TOKENS", "SAFETY", "RECITATION"])
def test_google_rejects_partial_or_blocked_json(monkeypatch, reason):
    from google import genai
    from google.genai import types

    response = SimpleNamespace(
        text='{"answers": {}}',
        candidates=[SimpleNamespace(finish_reason=types.FinishReason(reason))],
    )
    monkeypatch.setattr(genai, "Client", lambda api_key: SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **kwargs: response)))
    with pytest.raises(llm.InvalidModelResponse):
        llm._call_google_pdf("m", "k", "s", "u", PDF, "p.pdf")


def test_google_rejects_prompt_block(monkeypatch):
    from google import genai

    response = SimpleNamespace(
        text='{"answers": {}}', candidates=[],
        prompt_feedback=SimpleNamespace(block_reason="SAFETY"),
    )
    monkeypatch.setattr(genai, "Client", lambda api_key: SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **kwargs: response)))
    with pytest.raises(llm.InvalidModelResponse):
        llm._call_google_pdf("m", "k", "s", "u", PDF, "p.pdf")


def test_pdf_input_issue_enforces_size_and_claude_page_limits():
    assert llm.pdf_input_issue("OpenAI", "gpt-4.1", 1, 999) is None
    assert "19 MB" in llm.pdf_input_issue(
        "Google", "gemini-2.5-flash", llm.MAX_INLINE_PDF_BYTES + 1, 1
    )
    assert "100-page" in llm.pdf_input_issue(
        "Anthropic", "claude-haiku-4-5", 1, 101
    )
    assert "600-page" in llm.pdf_input_issue(
        "Anthropic", "claude-sonnet-4-6", 1, 601
    )
    assert llm.pdf_input_issue("Google", "gemini-2.5-flash", 1, 1000) is None
    assert "1000-page" in llm.pdf_input_issue("Google", "gemini-2.5-flash", 1, 1001)
