"""Edits on the review form survive leaving it; confirmed answers change only on request."""
from copy import deepcopy

import pytest
import streamlit as st
from streamlit.runtime.scriptrunner_utils.exceptions import StopException

from core import db
from features.extraction import drafts, schema, state as extraction
from features.workflow import state
from test_extraction_ui import ANSWER, DIGEST, PDF, QUESTION, element, ui  # noqa: F401

RUN = "▶ Run AI extraction (2 papers · 2 calls)"
PREFIX = "extraction:test-project"


def _papers(app):
    return app.session_state["project_store"]["papers"]


def _record(app, index=0):
    return extraction.get(_papers(app)[index])


@pytest.fixture
def two_papers(ui, monkeypatch):
    """Two included papers, both answered by the model through the normal run."""
    create, save, model = ui
    app = create()
    store = app.session_state["project_store"]
    second = state.new_paper("", "Second study", "")
    second["uid"] = "paper2"
    state.set_human_verdict(second, state.STAGE_FULLTEXT, state.VERDICT_INCLUDE,
                            review_hash=state.criteria_hash(store["config"]["fulltext_criteria"]))
    store["papers"].append(second)
    monkeypatch.setattr(db, "load_fulltexts", lambda *args: {
        uid: {"storage_key": f"mock/{uid}.pdf", "sha256": DIGEST, "status": "ok",
              "filename": "study.pdf", "file_size": len(PDF), "page_count": 2}
        for uid in ("paper1", "paper2")})
    app.run()
    element(app, "text_input", "API key").input("mock-key").run()
    element(app, "button", RUN).click().run()
    assert not app.exception and model.call_count == 2
    assert _record(app)["source_run_id"] and "review_state" not in _record(app)
    return app, save, model


def _confirm_first(app):
    for box in app.checkbox:
        if box.label == "Reviewed this question":
            box.check()
    app.run()
    element(app, "button", "Confirm extraction").click().run()
    assert _record(app)["review_state"] == "confirmed"
    # Confirming moves on; go back to the confirmed paper.
    element(app, "button", "‹ Prev").click().run()
    assert not app.exception


def test_edit_on_an_ai_answered_paper_is_kept_when_moving_to_the_next_paper(two_papers):
    app, save, model = two_papers
    element(app, "text_area", "Supporting quote").set_value("Evidence I typed.")
    element(app, "button", "Next ›").click().run()

    assert not app.exception
    saved = _record(app)
    assert saved["review_state"] == "draft"
    assert saved["final_answers"]["q1"]["quote"] == "Evidence I typed."
    assert saved["source_run_ids"] == {"q1": saved["source_run_id"]}
    element(app, "button", "‹ Prev").click().run()
    assert element(app, "text_area", "Supporting quote").value == "Evidence I typed."
    assert model.call_count == 2


def test_leaving_a_confirmed_paper_with_edits_asks_before_anything_changes(two_papers):
    app, save, model = two_papers
    _confirm_first(app)
    confirmed = deepcopy(_record(app))
    element(app, "text_area", "Supporting quote").set_value("A changed quote.")
    element(app, "button", "Next ›").click().run()

    assert not app.exception
    assert _record(app) == confirmed
    assert any("left the paper without saving" in warning.value for warning in app.warning)
    assert {button.label for button in app.button if not button.disabled} >= {
        "Save as a draft", "Discard my changes"}
    assert not [area for area in app.text_area if area.label == "Supporting quote"]
    element(app, "button", "Discard my changes").click().run()

    assert not app.exception
    assert _record(app) == confirmed
    assert f"{PREFIX}:held_edits" not in app.session_state
    element(app, "button", "‹ Prev").click().run()
    assert element(app, "text_area", "Supporting quote").value == confirmed["final_answers"]["q1"]["quote"]


@pytest.mark.parametrize("choice, state_after", [("Save and keep confirmed", "confirmed"),
                                                 ("Save as a draft", "draft")])
def test_held_edits_are_saved_only_by_an_explicit_choice(two_papers, choice, state_after):
    app, save, model = two_papers
    _confirm_first(app)
    element(app, "text_area", "Supporting quote").set_value("A changed quote.")
    element(app, "button", "Next ›").click().run()
    element(app, "button", choice).click().run()

    assert not app.exception
    saved = _record(app)
    assert saved["review_state"] == state_after
    assert saved["final_answers"]["q1"]["quote"] == "A changed quote."
    assert f"{PREFIX}:held_edits" not in app.session_state
    assert model.call_count == 2


def test_editing_a_confirmed_paper_in_place_is_not_interrupted(two_papers):
    app, save, model = two_papers
    _confirm_first(app)
    confirmed = deepcopy(_record(app))
    element(app, "text_area", "Supporting quote").input("Still typing.").run()
    element(app, "text_area", "Extraction instructions").input("Changed instructions.").run()

    assert not app.exception
    assert _record(app) == confirmed
    assert not any("left the paper without saving" in warning.value for warning in app.warning)
    assert element(app, "text_area", "Supporting quote").value == "Still typing."


def test_form_interrupted_while_first_drawn_cannot_overwrite_the_saved_draft(two_papers, monkeypatch):
    app, save, model = two_papers
    element(app, "button", "Next ›").click().run()
    element(app, "text_area", "Supporting quote").input("Second paper evidence.").run()
    element(app, "button", "Save draft").click().run()
    element(app, "button", "‹ Prev").click().run()
    draft = deepcopy(_record(app, 1))
    assert draft["review_state"] == "draft"

    real, armed = st.number_input, {"on": True}

    def interrupted_once(label, *args, **kwargs):
        if armed["on"] and label == "Evidence PDF page":
            armed["on"] = False
            raise StopException()
        return real(label, *args, **kwargs)

    monkeypatch.setattr(st, "number_input", interrupted_once)
    # The second paper's form stops after its answer widget, before page and quote.
    element(app, "button", "Next ›").click().run()
    assert not armed["on"]
    app.run()

    assert not app.exception
    assert _record(app, 1) == draft
    assert element(app, "text_area", "Supporting quote").value == "Second paper evidence."


def test_incomplete_widget_set_is_never_read_as_answers(monkeypatch):
    spec = schema.build_spec("", [QUESTION])
    session = {"k:single": "Forest"}
    monkeypatch.setattr(drafts, "st", type("S", (), {"session_state": session}))
    assert drafts._widget_answers(spec, {"q1": "k"}) == (None, [])
    session.update({"k:page": 1, "k:quote": "Evidence."})
    # The review box is drawn last; without it the form is still incomplete.
    assert drafts._widget_answers(spec, {"q1": "k"}) == (None, [])
    session["k:checked"] = True
    answers, checked = drafts._widget_answers(spec, {"q1": "k"})
    assert answers["q1"]["values"] == ["Forest"] and answers["q1"]["quote"] == "Evidence."
    assert checked == ["q1"]


@pytest.fixture
def whole_app(ui, monkeypatch):
    """The real multi-page application, entered through app.py."""
    create, save, model = ui
    app = create(ai=True, filename="app.py")
    app.run()
    app.switch_page("views/extraction.py").run()
    assert not app.exception
    return app, save, model


def test_edit_is_kept_as_a_draft_when_another_page_is_opened(whole_app):
    app, save, model = whole_app
    element(app, "text_area", "Supporting quote").set_value("Typed, then I opened another page.")
    app.switch_page("views/review_summary.py").run()

    assert not app.exception
    saved = _record(app)
    assert saved["review_state"] == "draft"
    assert saved["final_answers"]["q1"]["quote"] == "Typed, then I opened another page."
    assert f"{PREFIX}:open_form" not in app.session_state
    app.switch_page("views/extraction.py").run()
    assert element(app, "text_area", "Supporting quote").value == "Typed, then I opened another page."
    model.assert_not_called()


def test_confirmed_edits_block_the_next_page_until_resolved(whole_app):
    app, save, model = whole_app
    for box in app.checkbox:
        if box.label == "Reviewed this question":
            box.check()
    app.run()
    element(app, "button", "Confirm extraction").click().run()
    confirmed = deepcopy(_record(app))
    assert confirmed["review_state"] == "confirmed"
    element(app, "text_area", "Supporting quote").set_value("Changed after confirming.")
    app.switch_page("views/review_summary.py").run()

    assert not app.exception
    assert _record(app) == confirmed
    assert any("left the paper without saving" in warning.value for warning in app.warning)
    assert not app.get("download_button")
    element(app, "button", "Discard my changes").click().run()

    assert not app.exception
    assert _record(app) == confirmed
    assert not any("left the paper without saving" in warning.value for warning in app.warning)


def test_ticked_review_boxes_are_kept_when_moving_between_papers(two_papers):
    app, save, model = two_papers
    element(app, "checkbox", "Reviewed this question").check().run()
    element(app, "button", "Next ›").click().run()
    element(app, "button", "‹ Prev").click().run()

    assert not app.exception
    assert _record(app)["checked_questions"] == ["q1"]
    assert _record(app)["review_state"] == "draft"
    assert element(app, "checkbox", "Reviewed this question").value is True
    # The answers themselves were not edited, so they still count as accepted.
    assert _record(app)["decisions"] == {"q1": "accepted"}


def test_saving_a_new_setup_first_settles_edits_to_confirmed_answers(two_papers):
    app, save, model = two_papers
    _confirm_first(app)
    confirmed = deepcopy(_record(app))
    setup = deepcopy(app.session_state["project_store"]["config"])
    element(app, "text_area", "Supporting quote").input("Changed, then I edited the setup.").run()
    element(app, "text_area", "Extraction instructions").input("New instructions.").run()
    element(app, "checkbox",
            "I understand that changing the setup makes existing extraction results outdated.").check().run()
    element(app, "button", "Save extraction setup").click().run()

    assert not app.exception
    assert any("left the paper without saving" in warning.value for warning in app.warning)
    assert app.session_state["project_store"]["config"] == setup
    assert _record(app) == confirmed
    element(app, "button", "Save as a draft").click().run()

    assert not app.exception
    assert _record(app)["final_answers"]["q1"]["quote"] == "Changed, then I edited the setup."
    assert app.session_state["project_store"]["config"] == setup
    # The setup draft and its acknowledgement are still there: one more click saves it.
    assert element(app, "text_area", "Extraction instructions").value == "New instructions."
    assert not element(app, "button", "Save extraction setup").disabled
    element(app, "button", "Save extraction setup").click().run()
    assert not app.exception
    assert app.session_state["project_store"]["config"]["extraction_instructions"] == "New instructions."


def test_held_edit_prompt_shows_the_title_as_plain_text(two_papers):
    app, save, model = two_papers
    _papers(app)[0]["title"] = "![tracker](https://example.invalid/pixel.png)"
    app.run()
    _confirm_first(app)
    element(app, "text_area", "Supporting quote").set_value("A changed quote.")
    element(app, "button", "Next ›").click().run()

    prompt = next(warning.value for warning in app.warning if "left the paper without saving" in warning.value)
    assert "![tracker](" not in prompt
    assert "\\!\\[tracker\\]\\(" in prompt


def test_redraw_interrupted_before_the_review_box_keeps_the_saved_ticks(two_papers, monkeypatch):
    app, save, model = two_papers
    element(app, "checkbox", "Reviewed this question").check().run()
    element(app, "text_area", "Extraction instructions").input("Trigger one more complete draw.").run()
    saved = deepcopy(_record(app))
    assert saved["review_state"] == "draft" and saved["checked_questions"] == ["q1"]

    real, armed = st.checkbox, {"on": True}

    def interrupted_once(label, *args, **kwargs):
        if armed["on"] and label == "Reviewed this question":
            armed["on"] = False
            raise StopException()
        return real(label, *args, **kwargs)

    monkeypatch.setattr(st, "checkbox", interrupted_once)
    # The same form is drawn again and stops just before its review box.
    app.run()
    assert not armed["on"]
    app.run()

    assert not app.exception
    assert _record(app) == saved
    assert element(app, "checkbox", "Reviewed this question").value is True


@pytest.mark.parametrize("withdraw", ["untick", "edit"])
def test_remembered_setup_acknowledgement_is_withdrawn_by_unticking_or_editing(two_papers, withdraw):
    app, save, model = two_papers
    ack = "I understand that changing the setup makes existing extraction results outdated."
    _confirm_first(app)
    element(app, "text_area", "Supporting quote").input("Changed, then I edited the setup.").run()
    element(app, "text_area", "Extraction instructions").input("Setup A.").run()
    element(app, "checkbox", ack).check().run()
    element(app, "button", "Save extraction setup").click().run()
    element(app, "button", "Discard my changes").click().run()
    assert element(app, "checkbox", ack).value is True

    if withdraw == "untick":
        element(app, "checkbox", ack).uncheck().run()
    element(app, "text_area", "Extraction instructions").input("Setup B.").run()
    assert element(app, "checkbox", ack).value is False
    element(app, "text_area", "Extraction instructions").input("Setup A.").run()

    assert not app.exception
    assert element(app, "checkbox", ack).value is False
    assert element(app, "button", "Save extraction setup").disabled
    assert "extraction:test-project:acknowledged" not in app.session_state
