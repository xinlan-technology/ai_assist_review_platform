"""Question definitions, model response schemas, and strict answer validation."""
from __future__ import annotations

import hashlib
import json
import re

from core.llm import InvalidModelResponse

PROMPT_VERSION = "extraction-v1"
TYPES = {
    "single_choice": "Single choice",
    "multiple_choice": "Multiple choice",
    "open_text": "Open text",
}
OTHER = "Other"
NOT_REPORTED = "Not reported"
ANSWER_FIELDS = {"values", "other_text", "page", "quote", "issue"}


def _text(value: object, label: str, limit: int, required: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text.")
    value = value.strip()
    if len(value) > limit or (required and not value):
        raise ValueError(f"{label} must contain {'1' if required else '0'}–{limit} characters.")
    return value


def normalize_questions(questions: list, allow_empty: bool = False) -> list[dict]:
    if not isinstance(questions, list) or len(questions) > 30:
        raise ValueError("Provide a list of at most 30 questions.")
    if not questions and not allow_empty:
        raise ValueError("Add at least one extraction question.")
    normalized, ids = [], set()
    for question in questions:
        if not isinstance(question, dict):
            raise ValueError("Each question must be an object.")
        qid = _text(question.get("id"), "Question ID", 80, required=True)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", qid) or qid in ids:
            raise ValueError("Question IDs must be unique and use letters, numbers, underscores, or hyphens.")
        ids.add(qid)
        kind = question.get("type")
        if not isinstance(kind, str) or kind not in TYPES:
            raise ValueError("Choose Single choice, Multiple choice, or Open text.")
        options = []
        if kind != "open_text":
            supplied = question.get("options", [])
            if not isinstance(supplied, list):
                raise ValueError("Each question may contain at most 50 options.")
            seen = {OTHER.casefold(), NOT_REPORTED.casefold()}
            for option in supplied:
                option = _text(option, "Option", 200)
                if option and option.casefold() not in seen:
                    options.append(option)
                    seen.add(option.casefold())
            if len(options) > 50:
                raise ValueError("Each question may contain at most 50 user-defined options.")
            options.extend([OTHER, NOT_REPORTED])
        normalized.append({
            "id": qid,
            "text": _text(question.get("text"), "Question", 1000, required=True),
            "type": kind,
            "options": options,
            "guidance": _text(question.get("guidance", ""), "Question guidance", 2000),
        })
    return normalized


def build_spec(instructions: str, questions: list) -> dict:
    return {
        "instructions": _text(instructions, "Extraction instructions", 12000),
        "questions": normalize_questions(questions),
        "prompt_version": PROMPT_VERSION,
    }


def spec_hash(spec: dict) -> str:
    payload = json.dumps(spec, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def response_schema(spec: dict) -> dict:
    fields = {}
    for question in spec["questions"]:
        choice = question["type"] != "open_text"
        item = {"type": "string"}
        if choice:
            item["enum"] = question["options"]
        fields[question["id"]] = {
            "type": "object",
            "properties": {
                "values": {
                    "type": "array", "items": item,
                    "maxItems": len(question["options"]) if question["type"] == "multiple_choice" else 1,
                },
                "other_text": {"type": "string"},
                "page": {"$ref": "#/$defs/pdf_page"},
                "quote": {"type": "string"},
                "issue": {"type": "string"},
            },
            "required": ["values", "other_text", "page", "quote", "issue"],
            "additionalProperties": False,
        }
    return {
        "type": "object",
        "$defs": {"pdf_page": {"anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}]}},
        "properties": {"answers": {
            "type": "object", "properties": fields,
            "required": list(fields), "additionalProperties": False,
        }},
        "required": ["answers"],
        "additionalProperties": False,
    }


def provider_issue(provider: str, spec: dict) -> str | None:
    if provider != "OpenAI":
        return None
    enum_count = string_size = 0

    def count(node):
        nonlocal enum_count, string_size
        if isinstance(node, dict):
            enum_count += len(node.get("enum", []))
            string_size += sum(len(value) for value in node.get("enum", []) if isinstance(value, str))
            string_size += len(node["const"]) if isinstance(node.get("const"), str) else 0
            for field in ("properties", "$defs", "definitions"):
                for name, child in node.get(field, {}).items():
                    string_size += len(name)
                    count(child)
            for key, value in node.items():
                if key not in ("properties", "$defs", "definitions"):
                    count(value)
        elif isinstance(node, list):
            for value in node:
                count(value)

    count(response_schema(spec))
    if enum_count > 1000:
        return (f"OpenAI allows at most 1,000 choice options across all questions, including "
                f"Other and Not reported; this setup has {enum_count:,}. Reduce options or select another provider.")
    if string_size > 120000:
        return ("The combined option text and field names exceed OpenAI's 120,000-character "
                f"limit ({string_size:,}). Shorten options or select another provider.")
    return None


def validate_answer(question: dict, answer: dict, page_count: int | None = None) -> dict:
    if not isinstance(answer, dict) or set(answer) != ANSWER_FIELDS:
        raise ValueError("Answer must contain only values, other_text, page, quote, and issue.")
    issue = _text(answer["issue"], "Issue", 1000)
    if issue:
        raise ValueError(f"Manual review required: {issue}")
    values = answer["values"]
    if not isinstance(values, list) or not values:
        raise ValueError("Provide an answer or select Not reported.")
    values = [_text(value, "Answer", 4000, required=True) for value in values]
    if len(values) != len(set(values)):
        raise ValueError("An answer cannot contain duplicate values.")
    kind = question["type"]
    if kind != "multiple_choice" and len(values) != 1:
        raise ValueError("This question requires exactly one answer.")
    if kind != "open_text" and any(value not in question["options"] for value in values):
        raise ValueError("Every selected answer must match a configured option exactly.")
    if NOT_REPORTED in values and len(values) != 1:
        raise ValueError("Not reported cannot be combined with another answer.")
    other_text = _text(answer["other_text"], "Other explanation", 1000)
    uses_other = kind != "open_text" and OTHER in values
    if uses_other and not other_text:
        raise ValueError("Explain the answer when selecting Other.")
    if not uses_other and other_text:
        raise ValueError("Other explanation is only allowed when Other is selected.")
    page = answer["page"]
    if page is not None:
        if isinstance(page, bool) or not isinstance(page, int) or page < 1:
            raise ValueError("Evidence page must be a positive PDF page number.")
        if page_count is not None and page > page_count:
            raise ValueError(f"Evidence page exceeds the PDF's {page_count} pages.")
    quote = _text(answer["quote"], "Evidence quote", 1200)
    if values != [NOT_REPORTED] and (page is None or not quote):
        raise ValueError("Provide a PDF page number and a short supporting quote.")
    return {"values": values, "other_text": other_text, "page": page, "quote": quote, "issue": ""}


def validate_response(result: dict, spec: dict, page_count: int | None = None) -> tuple[dict, dict]:
    if not isinstance(result, dict) or set(result) != {"answers"} or not isinstance(result["answers"], dict):
        raise InvalidModelResponse("Extraction response must be an object containing an answers object.")
    supplied = result["answers"]
    ids = {question["id"] for question in spec["questions"]}
    if set(supplied) - ids:
        raise InvalidModelResponse("Extraction response contains unknown question IDs.")
    answers, errors = {}, {}
    for question in spec["questions"]:
        qid = question["id"]
        try:
            answers[qid] = validate_answer(question, supplied.get(qid), page_count)
        except ValueError as exc:
            errors[qid] = str(exc)
    return answers, errors
