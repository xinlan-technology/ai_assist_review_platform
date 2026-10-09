"""Checkpoint and export integration checks without API or storage calls."""
from copy import deepcopy
from unittest.mock import Mock

import pytest

from core.llm import InvalidModelResponse
from features.extraction import reports, schema, service, state


SPEC = schema.build_spec("Extract the study setting.", [{
    "id": "ecosystem", "text": "Ecosystem?", "type": "single_choice",
    "options": ["Forest", "Wetland"], "guidance": "Use the study site, not the background.",
}])
PDF = b"%PDF-1.7\nmock-document"
SOURCE = service.fulltext_storage.sha256(PDF)
ANSWERS = {"ecosystem": {
    "values": ["Forest"], "other_text": "", "page": 1,
    "quote": "The site was a forest.", "issue": "",
}}


@pytest.fixture
def setup(monkeypatch):
    papers = [{"uid": f"p{i}", "title": f"Study {i}", "doi": ""} for i in range(3)]
    metadata = {p["uid"]: {"storage_key": p["uid"], "sha256": SOURCE,
                          "filename": "study.pdf", "page_count": 2} for p in papers}
    model = Mock(return_value={"answers": ANSWERS})
    monkeypatch.setattr(service.fulltext_storage, "load_pdf", lambda key: PDF)
    monkeypatch.setattr(service.judge, "extract_pdf", model)
    return papers, metadata, model


def test_failed_initial_save_prevents_all_calls(setup):
    papers, metadata, model = setup
    assert not service.run(papers, SPEC, metadata, "P", "M", "k", lambda: False)
    model.assert_not_called()


def test_failed_checkpoint_stops_after_one_call_and_retains_answer(setup):
    papers, metadata, model = setup
    assert not service.run(papers, SPEC, metadata, "P", "M", "k", Mock(side_effect=[True, False]))
    assert model.call_count == 1
    assert state.get(papers[0])["ai_answers"] == ANSWERS
    remaining = [p for p in papers if state.pending(p, SPEC, SOURCE)]
    assert [p["uid"] for p in remaining] == ["p1", "p2"]
    progress = Mock()
    assert service.run(remaining, SPEC, metadata, "P", "M", "k", lambda: True, progress)
    assert model.call_count == 3
    assert progress.call_args.args == (2, 2)


@pytest.mark.parametrize("error,kind,retry", [
    (RuntimeError("Timed out"), "call_failed", True),
    (InvalidModelResponse("Invalid JSON"), "invalid_response", False),
])
def test_errors_are_persisted_and_retries_are_classified(setup, error, kind, retry):
    papers, metadata, model = setup
    model.side_effect = error
    assert service.run(papers[:1], SPEC, metadata, "P", "M", "k", lambda: True)
    assert state.get(papers[0])["ai_error_kind"] == kind
    assert state.pending(papers[0], SPEC, SOURCE) is retry


def test_pdf_hash_mismatch_never_reaches_provider(setup):
    papers, metadata, model = setup
    metadata["p0"]["sha256"] = "different"
    assert service.run(papers[:1], SPEC, metadata, "P", "M", "k", lambda: True)
    model.assert_not_called()
    assert "Reattach" in state.get(papers[0])["ai_error"]


def test_partial_invalid_result_is_not_automatically_retried(setup):
    papers, metadata, model = setup
    bad = deepcopy(ANSWERS)
    bad["ecosystem"]["values"] = ["Unknown"]
    model.return_value = {"answers": bad}
    assert service.run(papers[:1], SPEC, metadata, "P", "M", "k", lambda: True)
    assert state.status(papers[0], SPEC, SOURCE) == "Invalid"
    assert not state.pending(papers[0], SPEC, SOURCE)


def test_final_and_frequency_exports_use_only_current_confirmed_answers(setup):
    papers, metadata, _ = setup
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
    papers, metadata, _ = setup
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
    papers, metadata, _ = setup
    state.save_review(papers[0], SPEC, SOURCE, {"ecosystem": {"values": []}})
    assert reports.summary(papers, SPEC, metadata)["invalid_fields"] == 1


def test_audit_distinguishes_ai_and_review_errors_and_retains_draft_timestamp(setup):
    papers, metadata, _ = setup
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
