"""Gemini keeps output budget for the answer rather than spending it on reasoning."""
from types import SimpleNamespace

import pytest

from core import llm

PDF = b"%PDF-1.4 fake"


def test_flash_disables_thinking_and_keeps_the_requested_cap():
    assert llm._google_limits("gemini-2.5-flash", 512) == (512, 0)


def test_pro_cannot_disable_thinking_so_it_gets_headroom():
    cap, budget = llm._google_limits("gemini-2.5-pro", 512)
    assert budget == llm.GOOGLE_PRO_THINKING_BUDGET
    assert cap >= 4096


@pytest.mark.parametrize("model,expected_budget", [
    ("gemini-2.5-flash", 0),
    ("gemini-2.5-pro", llm.GOOGLE_PRO_THINKING_BUDGET),
])
def test_pdf_call_sends_a_thinking_budget(monkeypatch, model, expected_budget):
    from google import genai

    seen = {}

    monkeypatch.setattr(genai, "Client", lambda api_key: SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **kwargs: (
            seen.update(kwargs),
            SimpleNamespace(text='{"verdict":"include","reason":"r"}'),
        )[1])))

    llm._call_google_pdf(model, "key", "system", "prompt", PDF, "p.pdf")
    config = seen["config"]
    assert config.thinking_config.thinking_budget == expected_budget
    assert config.max_output_tokens >= 512


def test_text_call_leaves_reasoning_alone(monkeypatch):
    """Abstract screening retains the provider's default reasoning budget."""
    from google import genai

    seen = {}

    monkeypatch.setattr(genai, "Client", lambda api_key: SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **kwargs: (
            seen.update(kwargs),
            SimpleNamespace(text='{"verdict":"include","reason":"r"}'),
        )[1])))

    llm._call_google("gemini-2.5-flash", "key", "system", "prompt")
    assert seen["config"].thinking_config is None
    assert seen["config"].max_output_tokens is None


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
