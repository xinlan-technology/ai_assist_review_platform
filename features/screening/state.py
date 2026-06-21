from __future__ import annotations

import json
import math

import pandas as pd
import streamlit as st

from core import auth, db
from features.screening.rubric import DEFAULT_RUBRIC, DEFAULT_TOPIC

RUBRIC_SCORE = "Score"
RUBRIC_DEF = "Definition"

LABEL_RELEVANT = "Relevant"
LABEL_IRRELEVANT = "Not relevant"
LABEL_PENDING = "Pending"
LABEL_ERROR = "Error"

DECISION_LABELS = [LABEL_RELEVANT, LABEL_IRRELEVANT, LABEL_PENDING, LABEL_ERROR]
SCORE_OPTIONS = list(range(10, 101, 10))
DEFAULT_THRESHOLD = 80

_KEY = "screening_review"


def threshold() -> int:
    return int(st.session_state.get("threshold", DEFAULT_THRESHOLD))


def init(papers: list[dict], original_df: pd.DataFrame) -> None:
    st.session_state[_KEY] = {"papers": papers, "original_df": original_df, "idx": 0}


def has_results() -> bool:
    return _KEY in st.session_state and bool(st.session_state[_KEY]["papers"])


def papers() -> list[dict]:
    return st.session_state[_KEY]["papers"]


def count() -> int:
    return len(papers())


def current_index() -> int:
    return st.session_state[_KEY]["idx"]


def goto(i: int) -> None:
    st.session_state[_KEY]["idx"] = max(0, min(count() - 1, i))


def record(i: int, decision: str) -> None:
    papers()[i]["decision"] = decision


def set_human_score(i: int, score: int) -> None:
    papers()[i]["human_score"] = int(score)


def _is_number(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def label(score, thr: int) -> str:
    if not _is_number(score):
        return LABEL_ERROR
    return LABEL_RELEVANT if score >= thr else LABEL_IRRELEVANT


def ai_suggestion(p: dict, thr: int) -> str:
    return label(p.get("ai_score"), thr)


def final_decision(p: dict, thr: int) -> str:
    decision = p.get("decision")
    if decision is None:
        return LABEL_PENDING
    if decision == "agree":
        return ai_suggestion(p, thr)
    return label(p.get("human_score"), thr)


def is_reviewed(p: dict) -> bool:
    return p.get("decision") is not None


def summary(thr: int) -> dict:
    ps = papers()
    total = len(ps)
    reviewed = sum(1 for p in ps if is_reviewed(p))
    agreed = sum(1 for p in ps if p.get("decision") == "agree")
    disagreed = sum(1 for p in ps if p.get("decision") == "disagree")
    decided = agreed + disagreed
    counts = {lbl: 0 for lbl in DECISION_LABELS}
    for p in ps:
        counts[final_decision(p, thr)] += 1
    return {
        "total": total,
        "reviewed": reviewed,
        "remaining": total - reviewed,
        "agreed": agreed,
        "disagreed": disagreed,
        "agreement_rate": (agreed / decided) if decided else None,
        "counts": counts,
    }


def results_dataframe(thr: int) -> pd.DataFrame:
    store = st.session_state[_KEY]
    base = store["original_df"].copy()
    ps = store["papers"]
    agreed_text = {"agree": "Yes", "disagree": "No"}
    base["AI score"] = [p.get("ai_score") for p in ps]
    base["AI reason"] = [p.get("ai_reason") for p in ps]
    base["AI suggestion"] = [ai_suggestion(p, thr) for p in ps]
    base["Agreed?"] = [agreed_text.get(p.get("decision"), "") for p in ps]
    base["Your score"] = [
        p.get("human_score") if p.get("decision") == "disagree" else "" for p in ps
    ]
    base["Final decision"] = [final_decision(p, thr) for p in ps]
    return base


def active_id() -> str | None:
    return st.session_state.get("active_project_id")


def active_name() -> str | None:
    return st.session_state.get("active_project_name")


def set_active(pid: str | None, name: str | None) -> None:
    st.session_state["active_project_id"] = pid
    st.session_state["active_project_name"] = name


def _current_rubric() -> list[dict]:
    df = st.session_state.get("rubric_df")
    if df is not None:
        return [{"score": int(r[RUBRIC_SCORE]), "definition": str(r[RUBRIC_DEF])}
                for _, r in df.iterrows()]
    return [{"score": s, "definition": DEFAULT_RUBRIC.get(s, "")} for s in range(100, 0, -10)]


def snapshot() -> dict:
    data = {
        "topic": st.session_state.get("topic", DEFAULT_TOPIC),
        "threshold": threshold(),
        "rubric": _current_rubric(),
    }
    if _KEY in st.session_state:
        store = st.session_state[_KEY]
        df = store["original_df"]
        data["papers"] = store["papers"]
        data["original_columns"] = list(df.columns)
        data["original_records"] = json.loads(df.to_json(orient="records"))
        data["idx"] = store["idx"]
    return data


def restore(data: dict) -> None:
    st.session_state["topic"] = data.get("topic", DEFAULT_TOPIC)
    st.session_state["threshold"] = int(data.get("threshold", DEFAULT_THRESHOLD))

    rubric = data.get("rubric") or [
        {"score": s, "definition": DEFAULT_RUBRIC.get(s, "")} for s in range(100, 0, -10)
    ]
    st.session_state["rubric_df"] = pd.DataFrame(
        {RUBRIC_SCORE: [r["score"] for r in rubric],
         RUBRIC_DEF: [r["definition"] for r in rubric]}
    )
    st.session_state.pop("rubric_editor", None)

    if data.get("papers") is not None:
        df = pd.DataFrame(data.get("original_records", []), columns=data.get("original_columns"))
        st.session_state[_KEY] = {
            "papers": data["papers"],
            "original_df": df,
            "idx": int(data.get("idx", 0)),
        }
    else:
        st.session_state.pop(_KEY, None)


def save_active() -> bool:
    pid = active_id()
    if pid:
        try:
            db.save_project(auth.current_user(), pid, snapshot())
        except db.DatabaseError as exc:
            st.error(str(exc))
            return False
    return True
