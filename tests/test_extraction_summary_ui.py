"""Render extraction summaries using in-memory projects and mocked persistence."""
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest


from core import auth, csv_io, db
from features.extraction import state as extraction
from features.extraction.schema import build_spec
from features.workflow import state as workflow

ROOT = Path(__file__).resolve().parents[1]


def render_summary(monkeypatch, scenario="confirmed", mode=workflow.MODE_PRISMA):
    config = {
        "abstract_criteria": "Include relevant studies",
        "fulltext_criteria": "Include studies reporting ecosystem outcomes",
        "extraction_instructions": "Extract the studied ecosystem",
        "extraction_questions": [{
            "id": "habitat", "text": "Which ecosystem?", "type": "single_choice",
            "options": ["Forest", "Wetland"], "guidance": "Use the focal study only.",
        }],
    }
    paper = workflow.new_paper("10.1/example", "Example environmental study", "Example abstract")
    for stage in (workflow.STAGE_ABSTRACT, workflow.STAGE_FULLTEXT):
        criteria = config[f"{stage}_criteria"]
        workflow.set_ai_result(paper, stage, "include", "Eligible", "Provider", "Model",
                               workflow.criteria_hash(criteria), "screening-v1")
        workflow.record_agree(paper, stage)
    spec = build_spec(config["extraction_instructions"], config["extraction_questions"])
    answers = {"habitat": {
        "values": ["Forest"], "other_text": "", "page": 2,
        "quote": "We sampled the forest ecosystem.", "issue": "",
    }}
    extraction.set_ai_result(paper, spec, "pdf-digest", {"answers": answers}, "Provider", "Model", 3)
    extraction.save_review(paper, spec, "pdf-digest", answers, page_count=3, confirm=True)
    if scenario == "outdated":
        config["extraction_instructions"] = "Extract the comparison ecosystem instead"
    elif scenario == "ineligible":
        config["abstract_criteria"] = "Updated abstract eligibility criteria"
    metadata = {paper["uid"]: {"sha256": "pdf-digest", "page_count": 3}}
    monkeypatch.setattr(auth, "sidebar_user", lambda: None)
    monkeypatch.setattr(auth, "current_user", lambda: "test@example.invalid")
    monkeypatch.setattr(db, "load_fulltexts", lambda user, project: metadata)
    # These tests target extraction tables and controls, not screening chart rendering.
    monkeypatch.setattr(st, "bar_chart", lambda *args, **kwargs: None)

    def forbidden_database_access():
        raise AssertionError("Summary rendering must not access a database in this test.")

    monkeypatch.setattr(db, "_get_engine", forbidden_database_access)
    exported = []
    to_csv_bytes = csv_io.to_csv_bytes

    def capture_export(frame):
        exported.append(frame.copy(deep=True))
        return to_csv_bytes(frame)

    monkeypatch.setattr(csv_io, "to_csv_bytes", capture_export)
    app = AppTest.from_file(str(ROOT / "views" / "review_summary.py"), default_timeout=10)
    app.session_state["active_project_id"] = "offline-summary-project"
    app.session_state["active_project_name"] = "Offline summary test"
    app.session_state["project_store"] = {
        "mode": mode, "config": deepcopy(config), "papers": [paper],
        "original_df": pd.DataFrame(), "cursors": {},
    }
    app.run()
    assert not app.exception
    assert not app.error
    return app, exported


def extraction_metrics(app):
    labels = {"Eligible", "AI extracted", "Human confirmed", "Invalid fields"}
    return {metric.label: metric.value for metric in app.metric if metric.label in labels}


def extraction_frame(frames):
    return next(frame for frame in frames if "Extraction status" in frame.columns)


@pytest.mark.parametrize("mode", [workflow.MODE_PRISMA, workflow.MODE_DIRECT])
def test_summary_displays_current_confirmed_answers_metrics_and_downloads(monkeypatch, mode):
    app, exported = render_summary(monkeypatch, mode=mode)
    assert extraction_metrics(app) == {
        "Eligible": "1", "AI extracted": "1", "Human confirmed": "1", "Invalid fields": "0",
    }
    displayed = extraction_frame([element.value for element in app.dataframe])
    assert displayed.loc[0, "Extraction status"] == "Confirmed"
    assert displayed.loc[0, "Which ecosystem? [habitat]"] == "Forest"
    assert extraction_frame(exported).equals(displayed)
    choices = next(element.value for element in app.dataframe if "Option" in element.value.columns)
    assert choices.loc[choices["Option"] == "Forest", "Papers"].item() == 1
    labels = {button.label for button in app.download_button}
    assert {"Download extraction results CSV", "Download extraction audit CSV"} <= labels
    assert all(not button.disabled for button in app.download_button)
    audit = next(frame for frame in exported if "Version status" in frame.columns)
    assert audit.loc[0, "Current final"] == "Forest"
    assert audit.loc[0, "Decision"] == "accepted"


def test_summary_excludes_outdated_answers_from_metrics_and_final_export(monkeypatch):
    app, exported = render_summary(monkeypatch, "outdated")
    assert extraction_metrics(app) == {
        "Eligible": "1", "AI extracted": "0", "Human confirmed": "0", "Invalid fields": "0",
    }
    final = extraction_frame(exported)
    assert final.loc[0, "Extraction status"] == "Outdated"
    assert final.loc[0, "Which ecosystem? [habitat]"] == ""
    assert any("1 outdated" in caption.value for caption in app.caption)
    audit = next(frame for frame in exported if "Version status" in frame.columns)
    assert audit.loc[0, "Version status"] == "Outdated"
    assert "Forest" in audit.loc[0, "Recorded final"]
    assert audit.loc[0, "Current final"] == ""


def test_summary_keeps_ineligible_records_in_audit_only(monkeypatch):
    app, exported = render_summary(monkeypatch, "ineligible")
    assert extraction_metrics(app) == {
        "Eligible": "0", "AI extracted": "0", "Human confirmed": "0", "Invalid fields": "0",
    }
    assert extraction_frame(exported).empty
    displayed = extraction_frame([element.value for element in app.dataframe])
    assert displayed.empty
    audit = next(frame for frame in exported if "Version status" in frame.columns)
    assert audit.loc[0, "Version status"] == "Ineligible"
    assert "Forest" in audit.loc[0, "Recorded final"]
    assert audit.loc[0, "Current final"] == ""
    assert len(app.download_button) == 3
