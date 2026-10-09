"""Extraction question and field validation without network calls."""
from copy import deepcopy

import pytest


from core.llm import InvalidModelResponse
from features.extraction.schema import (
    build_spec, normalize_questions, provider_issue, response_schema, spec_hash,
    validate_answer, validate_response,
)


def question(kind="single_choice", **changes):
    return {"id": "q1", "text": "Ecosystem?", "type": kind, "options": ["Forest", "Wetland"], **changes}


def answer(values=None, **changes):
    return {
        "values": values if values is not None else ["Forest"],
        "other_text": "", "page": 2, "quote": "The study examined forests.", "issue": "", **changes,
    }


def test_question_normalization_is_stable_and_adds_reserved_options():
    raw = question(options=[" Forest ", "forest", "other", "NOT REPORTED", "Wetland", "", "  "])
    normalized = normalize_questions([raw])
    assert normalized[0]["options"] == ["Forest", "Wetland", "Other", "Not reported"]
    assert normalize_questions(normalized) == normalized
    assert raw["options"][0] == " Forest "
    many = normalize_questions([question(options=[str(i) for i in range(50)])])
    assert normalize_questions(many) == many


@pytest.mark.parametrize("questions", [
    [], [question(id="")], [question(), question()], [question(type="number")],
    [question(type=[])], [question(options=[str(i) for i in range(51)])],
])
def test_invalid_question_definitions_are_rejected(questions):
    with pytest.raises(ValueError):
        normalize_questions(questions)


def test_hash_tracks_question_options_instructions_order_and_prompt_version():
    spec = build_spec("Extract this study only", [question(), question("open_text", id="q2")])
    assert spec_hash(spec) == spec_hash(deepcopy(spec))
    variants = [deepcopy(spec) for _ in range(4)]
    variants[0]["questions"][0]["options"].append("Marine")
    variants[1]["instructions"] += " carefully"
    variants[2]["questions"].reverse()
    variants[3]["prompt_version"] = "next-version"
    assert all(spec_hash(item) != spec_hash(spec) for item in variants)


def test_schema_uses_stable_ids_strict_envelopes_and_shared_page_union():
    schema = response_schema(build_spec("", [question(), question("open_text", id="q2")]))
    fields = schema["properties"]["answers"]
    assert fields["required"] == ["q1", "q2"]
    assert fields["additionalProperties"] is False
    q1 = fields["properties"]["q1"]
    assert q1["properties"]["values"]["items"]["enum"] == ["Forest", "Wetland", "Other", "Not reported"]
    assert q1["properties"]["page"] == {"$ref": "#/$defs/pdf_page"}
    assert {"type": "null"} in schema["$defs"]["pdf_page"]["anyOf"]


def test_openai_total_enum_boundary_includes_reserved_options():
    questions = [question(id=f"q{i}", options=[f"option{j}" for j in range(48)]) for i in range(20)]
    spec = build_spec("", questions)
    before = deepcopy(spec)
    assert provider_issue("OpenAI", spec) is None
    assert spec == before
    questions[0]["options"].append("additional option")
    oversized = build_spec("", questions)
    assert "1,001" in provider_issue("OpenAI", oversized)
    assert provider_issue("Google", oversized) is None
    assert provider_issue("Anthropic", oversized) is None


def test_openai_schema_keywords_are_legal_question_ids_at_enum_boundary():
    ids = ["enum", "properties", "definitions", "defs", "const"]
    ids.extend(f"q{i}" for i in range(15))
    spec = build_spec("", [question(id=qid, options=[f"option{j}" for j in range(48)]) for qid in ids])
    assert provider_issue("OpenAI", spec) is None


def test_openai_schema_string_budget_counts_long_options_without_mutation():
    options = [f"{j:03d}" + "x" * 197 for j in range(40)]
    short = build_spec("x" * 12000, [question(id=f"q{i}", options=options[:-1]) for i in range(15)])
    assert provider_issue("OpenAI", short) is None
    long = build_spec("", [question(id=f"q{i}", options=options) for i in range(15)])
    before = deepcopy(long)
    assert "120,000-character" in provider_issue("OpenAI", long)
    assert long == before


@pytest.mark.parametrize("changes", [
    {"values": ["forest"]}, {"values": ["Forest", "Wetland"]}, {"values": []},
    {"values": ["Other"]}, {"other_text": "Unexpected explanation"},
    {"page": 0}, {"page": True}, {"page": 4}, {"page": None}, {"quote": ""},
    {"issue": "The PDF is unreadable"}, {"extra": "unexpected"},
])
def test_invalid_single_answers_are_not_silently_coerced(changes):
    q = normalize_questions([question()])[0]
    with pytest.raises(ValueError):
        validate_answer(q, answer(**changes), page_count=3)


def test_other_not_reported_and_multiselect_are_distinct():
    q = normalize_questions([question("multiple_choice")])[0]
    assert validate_answer(q, answer(["Other"], other_text="Marine"))["other_text"] == "Marine"
    assert validate_answer(q, answer(["Not reported"], page=None, quote=""))["page"] is None
    assert validate_answer(q, answer(["Forest", "Wetland"]))["values"] == ["Forest", "Wetland"]
    for values in (["Forest", "Not reported"], ["Forest", "Forest"]):
        with pytest.raises(ValueError):
            validate_answer(q, answer(values))


def test_open_text_still_requires_evidence_or_explicit_not_reported():
    q = normalize_questions([question("open_text")])[0]
    assert q["options"] == []
    assert validate_answer(q, answer(["Colombia"]))["values"] == ["Colombia"]
    assert validate_answer(q, answer(["Not reported"], page=None, quote=""))
    with pytest.raises(ValueError):
        validate_answer(q, answer(["Colombia"], page=None, quote=""))


def test_partial_invalid_response_keeps_valid_answers():
    spec = build_spec("", [question(), question(id="q2"), question(id="q3")])
    valid, errors = validate_response({"answers": {"q1": answer(), "q2": answer(["Savanna"])}}, spec, 3)
    assert list(valid) == ["q1"]
    assert set(errors) == {"q2", "q3"}


@pytest.mark.parametrize("result", [[], {}, {"answers": []}, {"answers": {}, "extra": 1}, {"answers": {"unknown": {}}}])
def test_bad_envelopes_are_invalid_model_responses(result):
    with pytest.raises(InvalidModelResponse):
        validate_response(result, build_spec("", [question()]))
