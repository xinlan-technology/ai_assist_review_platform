"""One persisted PRISMA workflow with two models at all three stages."""
from copy import deepcopy
from unittest.mock import Mock

from core import db
from features.extraction import state as extraction
from features.screening import prompts
from features.workflow import runs, state
from test_multi_runs import ANSWER, EXTRACTION_SPEC, MODELS, SPEC, project


def test_three_stage_comparison_and_human_confirmation_survive_reopen(project, monkeypatch):
    paper = state.papers()[0]
    state.config()["fulltext_criteria"] = SPEC["criteria"]
    for stage, version in (("abstract", prompts.ABSTRACT_PROMPT_VERSION),
                           ("fulltext", prompts.FULLTEXT_PROMPT_VERSION)):
        spec = {"criteria": SPEC["criteria"], "prompt_version": version}
        assert paper in [p for _, p in state.stage_papers(stage)]
        assert runs.execute([paper], stage, spec, project.metadata, MODELS)
        records = runs.load(stage, paper["uid"])
        assert len(records) == 2
        assert state.final_verdict(paper, stage) is None
        runs.confirm_screening(paper, stage, spec, records[1], "include", project.metadata)
        assert state.save_active()
    assert paper in [p for _, p in state.stage_papers("extraction")]
    model = Mock(return_value={"answers": deepcopy(ANSWER)})
    monkeypatch.setattr(runs.extraction_judge, "extract_pdf", model)
    assert runs.execute([paper], "extraction", EXTRACTION_SPEC, project.metadata, MODELS)
    proposals = runs.load("extraction", paper["uid"])
    assert len(proposals) == 2
    assert extraction.get(paper).get("review_state") is None
    selected = proposals[1]
    digest = project.metadata[paper["uid"]]["sha256"]
    answers = {"setting": extraction.candidate_answer(selected, EXTRACTION_SPEC, digest, "setting", 2)}
    extraction.save_review(paper, EXTRACTION_SPEC, digest, answers, 2, confirm=True,
                           source_run_ids={"setting": selected["id"]}, source_answers=answers)
    assert state.save_active()
    before = deepcopy(paper)
    assert runs.execute([paper], "extraction", EXTRACTION_SPEC, project.metadata, MODELS, repeat=True)
    assert paper == before
    saved, version = db.load_project_versioned(project.user, project.pid)
    state.load_into_session(saved, version=version, source=([], []))
    reopened = state.papers()[0]
    assert reopened == before
    assert extraction.confirmed_answers(reopened, EXTRACTION_SPEC, digest) == answers
    assert len(db.load_ai_runs(project.user, project.pid)) == 8
    assert project.model.call_count == 4
    assert model.call_count == 4
