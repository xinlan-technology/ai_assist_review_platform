"""Shared controls for accepting or overriding screening verdicts."""
from __future__ import annotations

import streamlit as st

from features.workflow import state


def decision_controls(
    paper: dict,
    stage: str,
    chash: str,
    *,
    position: int,
    total: int,
    stale_note: str,
    disabled: bool = False,
) -> None:
    """Collect and confirm a verdict before advancing.

    Outdated results and disabled evidence cannot receive decisions.
    """
    stg = state.stage_state(paper, stage)
    verdicts = state.STAGE_VERDICTS[stage]
    options = [state.VERDICT_LABELS[verdict] for verdict in verdicts]
    keys = f"{stage}:{paper['uid']}"

    def pick(label: str) -> str | None:
        current = stg.get("human_verdict")
        return st.radio(
            label, options,
            index=verdicts.index(current) if current else None,
            horizontal=True,
            key=f"{keys}:verdict:{stg.get('hv_nonce', 0)}",
        )

    def confirm(label: str, chosen: str | None, key: str) -> None:
        if st.button(label, type="primary", disabled=chosen is None or disabled, key=key):
            state.commit(lambda: state.set_human_verdict(
                paper, stage, verdicts[options.index(chosen)], review_hash=chash,
            ))
            state.goto(stage, position + 1, total)
            st.rerun()

    if state.is_stale(paper, stage, chash):
        st.warning(stale_note)
        return

    ai_verdict = stg.get("ai_verdict")
    if not ai_verdict:
        confirm("Confirm verdict", pick("Your verdict (independent of the AI)"),
                f"{keys}:confirm")
        return

    decision = stg.get("decision")
    st.caption("Do you agree with the AI?")
    agree_col, disagree_col = st.columns(2)
    if agree_col.button(
        "✓ Agree", width="stretch", key=f"{keys}:agree", disabled=disabled,
        type="primary" if decision == state.DECISION_AGREE else "secondary",
    ):
        state.commit(lambda: state.record_agree(paper, stage))
        state.goto(stage, position + 1, total)
        st.rerun()
    if disagree_col.button(
        "✗ Disagree", width="stretch", key=f"{keys}:disagree", disabled=disabled,
        type="primary" if decision == state.DECISION_DISAGREE else "secondary",
    ):
        state.commit(lambda: state.record_disagree(paper, stage))
        st.rerun()

    if stg.get("decision") == state.DECISION_DISAGREE:
        chosen = pick("Your verdict")
        if chosen is None:
            st.caption("Pick a verdict, then confirm — the paper stays unreviewed until you do.")
        confirm("Confirm your verdict", chosen, f"{keys}:confirm-override")
