"""Opt-in in-memory AI-run persistence for UI tests, never live database tests."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

from core import db


def install_run_store(monkeypatch):
    """Mock only the run API, leaving the global live-I/O guard in force."""
    rows = []
    owners = {}

    def start(user_email, pid, *, paper_uid, stage, batch_id, provider, model,
              config_hash, source_hash, prompt_version, prompt_snapshot,
              expected_version=None, run_id=None):
        identifier = run_id or f"offline-run-{len(rows) + 1}"
        row = {
            "id": identifier, "project_id": pid, "user_email": user_email,
            "paper_uid": paper_uid, "stage": stage,
            "batch_id": batch_id, "provider": provider, "model": model,
            "config_hash": config_hash, "source_hash": source_hash,
            "prompt_version": prompt_version,
            "prompt_snapshot": deepcopy(prompt_snapshot),
            "started_at": "2026-01-01T00:00:00+00:00", "completed_at": None,
            "status": "running", "result": None, "duration_seconds": None,
        }
        if identifier in owners:
            previous = next(item for item in rows if item["id"] == identifier)
            assert all(previous[key] == value for key, value in row.items()
                       if key not in {"started_at", "completed_at", "status", "result", "duration_seconds"})
            return deepcopy(previous)
        owners[identifier] = (user_email, pid)
        rows.append(row)
        return deepcopy(row)

    def finish(user_email, pid, run_id, *, status, result, duration_seconds, verify_paper=True):
        assert owners.get(run_id) == (user_email, pid), "Run ownership mismatch."
        row = next(item for item in rows if item["id"] == run_id)
        if row["status"] != "running":
            assert (row["status"], row["result"], row["duration_seconds"]) == (status, result, duration_seconds), (
                "Terminal run records are immutable."
            )
            return deepcopy(row)
        row.update(status=status, result=deepcopy(result), duration_seconds=duration_seconds,
                   completed_at="2026-01-01T00:00:01+00:00")
        return deepcopy(row)

    def load(user_email, pid, stage=None, paper_uid=None):
        return deepcopy([
            row for row in rows
            if owners[row["id"]] == (user_email, pid)
            and (stage is None or row["stage"] == stage)
            and (paper_uid is None or row["paper_uid"] == paper_uid)
        ])

    def index(user_email, pid, stage=None):
        return [{name: row[name] for name in db._RUN_INDEX_COLUMNS}
                for row in load(user_email, pid, stage)]

    fixture = SimpleNamespace(rows=rows, start=Mock(side_effect=start),
                              finish=Mock(side_effect=finish), load=Mock(side_effect=load),
                              index=Mock(side_effect=index))
    monkeypatch.setattr(db, "start_ai_run", fixture.start)
    monkeypatch.setattr(db, "finish_ai_run", fixture.finish)
    monkeypatch.setattr(db, "load_ai_runs", fixture.load)
    monkeypatch.setattr(db, "load_ai_run_index", fixture.index)
    return fixture
