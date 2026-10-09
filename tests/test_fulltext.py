"""Pure contract tests for full-text prompts and verdict parsing."""
from __future__ import annotations


from features.screening import judge, prompts
from core.llm import InvalidModelResponse


def test_fulltext_prompt_contains_user_instructions_and_fixed_contract():
    text = prompts.build_fulltext_prompt("Include randomized trials only", "Paper A")
    assert "Include randomized trials only" in text
    assert "Paper A" in text
    assert '"include" | "exclude"' in text
    assert "untrusted" in text
    assert "unsure outcome" in text


def test_fulltext_judge_accepts_only_fulltext_verdicts():
    old = judge.call_pdf_structured
    try:
        judge.call_pdf_structured = lambda *args, **kwargs: {
            "verdict": "include", "reason": "eligible"
        }
        result = judge.judge_fulltext(
            "OpenAI", "m", "k", "criteria", "title", b"%PDF-x", "x.pdf"
        )
        assert result == {"verdict": "include", "reason": "eligible"}

        judge.call_pdf_structured = lambda *args, **kwargs: {
            "verdict": "unsure", "reason": "unclear"
        }
        try:
            judge.judge_fulltext(
                "OpenAI", "m", "k", "criteria", "title", b"%PDF-x", "x.pdf"
            )
        except InvalidModelResponse:
            pass
        else:
            raise AssertionError("full-text unsure must be rejected")

        for bad_reason in (None, "", {"detail": "not text"}):
            judge.call_pdf_structured = lambda *args, _reason=bad_reason, **kwargs: {
                "verdict": "include",
                "reason": _reason,
            }
            try:
                judge.judge_fulltext(
                    "OpenAI", "m", "k", "criteria", "title", b"%PDF-x", "x.pdf"
                )
            except InvalidModelResponse:
                pass
            else:
                raise AssertionError("a missing/non-text reason must be rejected")
    finally:
        judge.call_pdf_structured = old
