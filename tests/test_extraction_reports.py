"""Extraction report checks without API, database, or storage calls."""
from copy import deepcopy

import pytest

from core.fulltext_storage import sha256
from features.extraction import reports, schema, state


SPEC = schema.build_spec("Extract the study setting.", [{
    "id": "ecosystem", "text": "Ecosystem?", "type": "single_choice",
    "options": ["Forest", "Wetland"], "guidance": "Use the study site, not the background.",
}])
PDF = b"%PDF-1.7\nmock-document"
SOURCE = sha256(PDF)
ANSWERS = {"ecosystem": {
    "values": ["Forest"], "other_text": "", "page": 1,
    "quote": "The site was a forest.", "issue": "",
}}


@pytest.fixture
def setup():
    papers = [{"uid": f"p{i}", "title": f"Study {i}", "doi": ""} for i in range(3)]
    metadata = {p["uid"]: {"storage_key": p["uid"], "sha256": SOURCE,
                          "filename": "study.pdf", "page_count": 2} for p in papers}
    return papers, metadata


def test_final_and_frequency_exports_use_only_current_confirmed_answers(setup):
    papers, metadata = setup
    for p in papers:
        state.set_ai_result(p, SPEC, SOURCE, {"answers": ANSWERS}, "P", "M", 2)
    state.save_review(papers[0], SPEC, SOURCE, ANSWERS, page_count=2, confirm=True)
    state.save_review(papers[1], SPEC, SOURCE, ANSWERS, page_count=2)
    state.save_review(papers[2], SPEC, SOURCE, ANSWERS, page_count=2, confirm=True)
    metadata["p2"]["sha256"] = "replaced"
    frame = reports.final_dataframe(papers, SPEC, metadata)
    assert list(frame["Extraction status"]) == ["Confirmed", "Draft", "Outdated"]
    assert list(frame["Ecosystem? [ecosystem]"]) == ["Forest", "", ""]
    counts = reports.summary(papers, SPEC, metadata)
    assert counts["confirmed"] == counts["outdated"] == 1
    assert counts["ai_done"] == 2
    frequency = reports.choice_counts(papers, SPEC, metadata)
    assert frequency["Papers"].sum() == 1
    assert set(frequency["Confirmed papers"]) == {1}


def test_audit_retains_old_question_setup_but_not_as_current_final(setup):
    papers, metadata = setup
    paper = papers[0]
    state.save_review(paper, SPEC, SOURCE, ANSWERS, page_count=2, confirm=True)
    state.archive(paper)
    updated = deepcopy(SPEC)
    updated["questions"][0]["text"] = "Main ecosystem?"
    updated["questions"][0]["guidance"] = "Updated guidance"
    state.save_review(paper, updated, SOURCE, ANSWERS, page_count=2, confirm=True)
    audit = reports.audit_dataframe([paper], {"p0"}, updated, metadata)
    assert list(audit["Version status"]) == ["Archived", "Confirmed"]
    assert list(audit["Question"]) == ["Ecosystem?", "Main ecosystem?"]
    assert list(audit["Question guidance"]) == [SPEC["questions"][0]["guidance"], "Updated guidance"]
    assert list(audit["Current final"]) == ["", "Forest"]
    ineligible = reports.audit_dataframe([paper], set(), updated, metadata)
    assert set(ineligible["Current final"]) == {""}
    assert ineligible.iloc[-1]["Version status"] == "Ineligible"
    assert ineligible.iloc[-1]["Recorded final"]


def test_draft_invalid_fields_are_counted(setup):
    papers, metadata = setup
    state.save_review(papers[0], SPEC, SOURCE, {"ecosystem": {"values": []}})
    assert reports.summary(papers, SPEC, metadata)["invalid_fields"] == 1


def test_audit_distinguishes_ai_and_review_errors_and_retains_draft_timestamp(setup):
    papers, metadata = setup
    paper = papers[0]
    invalid = deepcopy(ANSWERS)
    invalid["ecosystem"]["values"] = ["Unknown"]
    state.set_ai_result(paper, SPEC, SOURCE, {"answers": invalid}, "P", "M", 2)
    state.save_review(paper, SPEC, SOURCE, {"ecosystem": {"values": []}})
    saved = deepcopy(state.get(paper))
    for expected_status in ("Draft", "Archived"):
        audit = reports.audit_dataframe([paper], {paper["uid"]}, SPEC, metadata)
        row = audit.loc[audit["Version status"] == expected_status].iloc[0]
        assert row["Field error"] == saved["field_errors"]["ecosystem"]
        assert row["Review error"] == saved["review_errors"]["ecosystem"]
        assert row["Draft saved at"] == saved["draft_saved_at"]
        assert row["Current final"] == ""
        state.archive(paper)


def test_other_explanation_is_exported_without_losing_choice():
    assert reports.answer_text({"values": ["Other"], "other_text": "Grassland"}) == "Other (Grassland)"
