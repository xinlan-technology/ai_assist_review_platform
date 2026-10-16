"""Every provider gets the same low reasoning level and room for it under the output cap."""
from types import SimpleNamespace

import pytest

from core import llm

PDF = b"%PDF-1.4 fake"
TEXT = '{"verdict":"include","reason":"r"}'


@pytest.fixture
def sent(monkeypatch):
    import anthropic
    import openai
    from google import genai

    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            output_text=TEXT, status="completed", output=[],
            choices=[SimpleNamespace(finish_reason="stop",
                                     message=SimpleNamespace(content=TEXT, refusal=None))],
            stop_reason="end_turn", content=[SimpleNamespace(type="text", text=TEXT)],
            candidates=[SimpleNamespace(finish_reason="STOP")], text=TEXT,
        )

    monkeypatch.setattr(openai, "OpenAI", lambda api_key, **kwargs: SimpleNamespace(
        responses=SimpleNamespace(create=fake),
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake))))
    monkeypatch.setattr(anthropic, "Anthropic", lambda api_key, **kwargs: SimpleNamespace(
        messages=SimpleNamespace(create=fake)))
    monkeypatch.setattr(genai, "Client", lambda api_key: SimpleNamespace(
        models=SimpleNamespace(generate_content=fake)))
    return seen


@pytest.mark.parametrize("provider", list(llm.PROVIDERS))
def test_pdf_calls_send_low_reasoning_with_headroom(sent, provider):
    from google.genai import types

    llm.call_pdf_structured(provider, llm.PROVIDERS[provider][0], "key", "system", "prompt", PDF, "p.pdf")
    cap = 512 + llm.REASONING_HEADROOM
    if provider == "OpenAI":
        assert sent["reasoning"] == {"effort": "low"}
        assert sent["max_output_tokens"] == cap
    elif provider == "Anthropic":
        assert sent["output_config"]["effort"] == "low"
        assert sent["max_tokens"] == cap
    else:
        assert sent["config"].thinking_config.thinking_level == types.ThinkingLevel.LOW
        assert sent["config"].thinking_config.thinking_budget is None
        assert sent["config"].max_output_tokens == cap


@pytest.mark.parametrize("provider", list(llm.PROVIDERS))
def test_text_calls_send_low_reasoning(sent, provider):
    from google.genai import types

    llm.call_structured(provider, llm.PROVIDERS[provider][0], "key", "system", "prompt")
    if provider == "OpenAI":
        assert sent["reasoning_effort"] == "low"
    elif provider == "Anthropic":
        assert sent["output_config"] == {"effort": "low"}
        assert sent["max_tokens"] == 1024 + llm.REASONING_HEADROOM
    else:
        assert sent["config"].thinking_config.thinking_level == types.ThinkingLevel.LOW
        assert sent["config"].max_output_tokens is None


def test_largest_extraction_cap_fits_anthropic_non_streaming_limit():
    # The SDK refuses non-streaming calls it expects to exceed 10 minutes.
    assert llm._output_cap(12000) * 3600 / 128_000 <= 600


@pytest.mark.parametrize("provider,reason", [
    ("Google", "MAX_TOKENS"), ("OpenAI", "length"), ("Anthropic", "max_tokens"),
    ("OpenAI", "incomplete"),
])
def test_truncation_is_reported_as_an_output_limit(provider, reason):
    with pytest.raises(llm.InvalidModelResponse) as exc:
        llm._check_finished(provider, reason, "never", 512)
    assert "output limit" in str(exc.value)
    assert "512" in str(exc.value)


def test_other_abnormal_finishes_keep_their_own_message():
    with pytest.raises(llm.InvalidModelResponse) as exc:
        llm._check_finished("Google", "SAFETY", "STOP", 512)
    assert "SAFETY" in str(exc.value)
