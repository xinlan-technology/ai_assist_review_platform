"""Exports attribute chosen answers and diagnostics to their actual AI run."""
from copy import deepcopy
import json

import pandas as pd
import pytest

from features.extraction import reports, schema, state as extraction
from features.workflow import state


SPEC = schema.build_spec("Use the paper's evidence.", [
    {"id": "setting", "text": "Setting?", "type": "single_choice", "options": ["Forest", "Wetland"]},
    {"id": "region", "text": "Region?", "type": "open_text"},
])
DIGEST = "offline-pdf-digest"
ANSWER_A = {
    "setting": {"values": ["Forest"], "other_text": "", "page": 1,
                "quote": "A forest site.", "issue": ""},
    "region": {"values": ["Brazil"], "other_text": "", "page": 1,
               "quote": "The site was in Brazil.", "issue": ""},
}
ANSWER_B = {
    "setting": dict(ANSWER_A["setting"], values=["Wetland"], page=2, quote="A wetland site."),
    "region": dict(ANSWER_A["region"], values=["Canada"], page=2, quote="The site was in Canada."),
}
SOURCE_A = {"id": "run-a", "provider": "OpenAI", "model": "model-a",
            "completed_at": "2026-01-01T01:00:00+00:00"}
SOURCE_B = {"id": "run-b", "provider": "Anthropic", "model": "model-b",
            "completed_at": "2026-01-01T02:00:00+00:00"}


def paper_with_baseline(answers=None):
    paper = state.new_paper("", "Offline study", "Offline abstract")
    extraction.set_ai_result(paper, SPEC, DIGEST, {"answers": answers or ANSWER_A},
                             SOURCE_A["provider"], SOURCE_A["model"], 2)
    extraction.get(paper).update(source_run_id=SOURCE_A["id"], completed_at=SOURCE_A["completed_at"])
    return paper


def audit(paper):
    return reports.audit_dataframe([paper], {paper["uid"]}, SPEC, {
        paper["uid"]: {"sha256": DIGEST, "page_count": 2},
    }).set_index("Question ID")


def test_mixed_final_export_uses_each_questions_selected_answer_and_model():
    paper = paper_with_baseline()
    selected = {"setting": ANSWER_B["setting"], "region": ANSWER_A["region"]}
    sources = {"setting": SOURCE_B, "region": SOURCE_A}
    extraction.save_review(
        paper, SPEC, DIGEST, selected, 2, confirm=True,
        source_run_ids={qid: source["id"] for qid, source in sources.items()},
        source_answers=selected, source_metadata=sources,
    )
    before = deepcopy(paper)
    frame = audit(paper)

    for qid, source in sources.items():
        row = frame.loc[qid]
        assert row["Source run ID"] == source["id"]
        assert row["Provider"] == source["provider"]
        assert row["Model"] == source["model"]
        assert row["AI completed at"] == source["completed_at"]
        assert json.loads(row["AI answer"]) == selected[qid]
        assert row["AI page"] == selected[qid]["page"]
        assert row["AI quote"] == selected[qid]["quote"]
        assert row["Decision"] == "accepted"
        assert row["Current final"] == selected[qid]["values"][0]
    assert paper == before


@pytest.mark.parametrize("baseline_failure", ["invalid_field", "call_failed"])
def test_selected_valid_run_never_inherits_another_runs_failure(baseline_failure):
    invalid = deepcopy(ANSWER_A)
    invalid["setting"]["values"] = ["Not a configured option"]
    paper = paper_with_baseline(invalid)
    if baseline_failure == "call_failed":
        extraction.set_error(paper, SPEC, DIGEST, "Run A provider failed.", "call_failed")
        extraction.get(paper).update(source_run_id=SOURCE_A["id"], provider=SOURCE_A["provider"],
                                     model=SOURCE_A["model"], completed_at=SOURCE_A["completed_at"])
    extraction.save_review(
        paper, SPEC, DIGEST, ANSWER_B, 2, confirm=True,
        source_run_ids={qid: SOURCE_B["id"] for qid in ANSWER_B},
        source_answers=ANSWER_B, source_metadata={qid: SOURCE_B for qid in ANSWER_B},
    )

    frame = audit(paper)
    # A call failure archives the preceding invalid baseline; check the current
    # confirmed version rather than deliberately retained historical errors.
    current = frame[frame["Version status"] == "Confirmed"]
    assert set(current["Source run ID"]) == {SOURCE_B["id"]}
    assert set(current["Model"]) == {SOURCE_B["model"]}
    assert set(current["Field error"]) == {""}
    assert set(current["AI error"]) == {""}


def test_baseline_failure_is_retained_when_no_other_run_was_selected():
    invalid = deepcopy(ANSWER_A)
    invalid["setting"]["values"] = ["Not a configured option"]
    paper = paper_with_baseline(invalid)
    row = audit(paper).loc["setting"]
    assert row["Source run ID"] == SOURCE_A["id"]
    assert row["Model"] == SOURCE_A["model"]
    assert row["Field error"]


def test_unknown_legacy_source_does_not_borrow_the_initial_models_metadata():
    paper = paper_with_baseline()
    extraction.save_review(
        paper, SPEC, DIGEST, ANSWER_B, 2, confirm=True,
        source_run_ids={qid: "older-source-without-metadata" for qid in ANSWER_B},
        source_answers=ANSWER_B,
    )
    frame = audit(paper)
    assert set(frame["Source run ID"]) == {"older-source-without-metadata"}
    assert set(frame["Provider"]) == {""}
    assert set(frame["Model"]) == {""}
    assert set(frame["AI completed at"]) == {""}
    assert json.loads(frame.loc["setting", "AI answer"]) == ANSWER_B["setting"]


@pytest.mark.parametrize("legacy", [False, True])
def test_screening_results_forward_exact_source_run_ids_or_leave_legacy_blank(monkeypatch, legacy):
    project = state.new_project_data(state.MODE_PRISMA)
    paper = state.new_paper("", "Offline study", "Offline abstract")
    for stage in (state.STAGE_ABSTRACT, state.STAGE_FULLTEXT):
        project["config"][f"{stage}_criteria"] = f"Criteria for {stage}."
        state.set_ai_result(paper, stage, "include", "Eligible.", "Provider", "Model",
                            state.criteria_hash(project["config"][f"{stage}_criteria"]), "prompt-v1")
        state.record_agree(paper, stage)
        if not legacy:
            state.stage_state(paper, stage)["source_run_id"] = f"selected-{stage}-run"
    project["papers"] = [paper]
    project["original_df"] = pd.DataFrame()
    monkeypatch.setattr(state, "_store", lambda: project)

    frame = state.results_dataframe()

    for stage, label in ((state.STAGE_ABSTRACT, "Abstract"), (state.STAGE_FULLTEXT, "Full-text")):
        assert frame.loc[0, f"{label}: Source run ID"] == ("" if legacy else f"selected-{stage}-run")
        assert frame.loc[0, f"{label}: Final"] == "Include"
