"""Offline checks for extraction migration and screening-stage integration."""
from copy import deepcopy

import pytest


from features.extraction import state as extraction
from features.extraction.schema import build_spec
from features.workflow import state as workflow

A = workflow.STAGE_ABSTRACT
F = workflow.STAGE_FULLTEXT
E = workflow.STAGE_EXTRACTION
HASHES = {A: workflow.criteria_hash("abstract criteria"), F: workflow.criteria_hash("full-text criteria")}
SPEC = build_spec("Extract the study location", [{"id": "region", "text": "Where?", "type": "open_text"}])
ANSWERS = {"region": {"values": ["Chile"], "other_text": "", "page": 1, "quote": "The study took place in Chile.", "issue": ""}}


def paper(abstract=None, fulltext="include"):
    result = workflow.new_paper("10.1/example", "Example study", "Example abstract")
    if abstract:
        workflow.set_human_verdict(result, A, abstract, HASHES[A])
    if fulltext:
        workflow.set_ai_result(result, F, fulltext, "Eligible study", "Provider", "Model", HASHES[F], "fulltext-v1")
        workflow.record_agree(result, F)
    return result


def eligible(papers, mode=workflow.MODE_PRISMA, hashes=None):
    return workflow.eligible_papers(papers, mode, E, HASHES if hashes is None else hashes)


def test_v2_migration_preserves_existing_project_identity_configuration_and_results():
    old_paper = paper("include")
    config = {
        "abstract_criteria": "Original abstract prompt", "fulltext_criteria": "Original full-text prompt",
        "extraction_instructions": "Draft extraction instructions", "extraction_questions": SPEC["questions"],
        "fulltext_exclusion_reasons": ["Legacy reason"],
    }
    original = {
        "schema_version": 2, "mode": workflow.MODE_PRISMA, "config": config,
        "papers": [old_paper], "original_columns": ["Title"],
        "original_records": [{"Title": "Example study"}], "cursors": {A: 3, F: 1},
    }
    result = workflow.migrate(deepcopy(original))
    assert result["schema_version"] == workflow.SCHEMA_VERSION
    assert result["papers"] == original["papers"]
    assert result["papers"][0]["uid"] == old_paper["uid"]
    assert all(result["config"][key] == value for key, value in config.items())
    assert result["cursors"] == original["cursors"]
    assert result["original_records"] == original["original_records"]
    assert result["original_columns"] == original["original_columns"]
    assert E in workflow.MODE_STAGES[result["mode"]]


def test_future_project_schema_is_rejected_without_mutation():
    data = {"schema_version": workflow.SCHEMA_VERSION + 1, "papers": [paper("include")]}
    before = deepcopy(data)
    with pytest.raises(workflow.UnsupportedSchemaError):
        workflow.migrate(data)
    assert data == before


def test_prisma_extraction_requires_both_screening_stages_to_qualify():
    papers = [
        paper("include"), paper("unsure"), paper("exclude"), paper(None),
        paper("include", "exclude"), paper("include", None),
    ]
    assert [index for index, _ in eligible(papers)] == [0, 1]


@pytest.mark.parametrize("stage", [A, F])
def test_outdated_upstream_screening_prevents_extraction(stage):
    papers = [paper("include")]
    assert len(eligible(papers)) == 1
    changed = HASHES | {stage: workflow.criteria_hash("New eligibility criteria")}
    assert eligible(papers, hashes=changed) == []


def test_direct_mode_does_not_require_an_abstract_verdict():
    papers = [paper(None), paper("exclude"), paper(None, "exclude"), paper(None, None)]
    assert [index for index, _ in eligible(papers, workflow.MODE_DIRECT, {F: HASHES[F]})] == [0, 1]


def test_unconfirmed_fulltext_disagreement_prevents_extraction():
    result = paper("include")
    assert eligible([result])
    workflow.record_disagree(result, F)
    assert eligible([result]) == []
    workflow.set_human_verdict(result, F, "include")
    assert eligible([result])


def test_replaced_pdf_archives_fulltext_and_extraction_with_original_provenance():
    result = paper("include")
    extraction.set_ai_result(result, SPEC, "old-pdf", {"answers": ANSWERS}, "Provider", "Model", 4)
    extraction.save_review(result, SPEC, "old-pdf", ANSWERS, page_count=4, confirm=True)
    abstract_before = deepcopy(result["stages"][A])
    assert workflow.archive_document_results(result, "old-pdf", "new-pdf")
    fulltext = workflow.stage_state(result, F)
    assert workflow.current_final_verdict(result, F, HASHES[F]) is None
    assert fulltext["history"][-1]["ai_verdict"] == "include"
    assert result["stages"][A] == abstract_before
    record = extraction.get(result)
    assert set(record) - {"form_nonce"} == {"history"}
    assert record["history"][-1]["spec"] == SPEC
    assert record["history"][-1]["source_sha256"] == "old-pdf"
    assert record["history"][-1]["final_answers"] == ANSWERS
    assert not extraction.confirmed_answers(result, SPEC, "new-pdf")
    assert eligible([result]) == []


def test_identical_pdf_attachment_does_not_invalidate_screening_or_extraction():
    result = paper("include")
    extraction.save_review(result, SPEC, "same-pdf", ANSWERS, page_count=4, confirm=True)
    before = deepcopy(result)
    assert not workflow.archive_document_results(result, "same-pdf", "same-pdf")
    assert result == before
    assert extraction.confirmed_answers(result, SPEC, "same-pdf") == ANSWERS
    assert eligible([result])
