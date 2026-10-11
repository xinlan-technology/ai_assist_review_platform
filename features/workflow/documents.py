"""Detached PDF and CSV changes, with guarded storage cleanup."""
from __future__ import annotations

from copy import deepcopy
from uuid import uuid4

from core import db, fulltext_storage
from features.workflow import state


def cleanup_keys(user_email: str, keys: list[str]) -> list[str]:
    """Return failed keys; never delete an object referenced by PDF metadata."""
    failed = []
    for key in dict.fromkeys(key for key in keys if key):
        try:
            if not db.fulltext_key_in_use(user_email, key):
                fulltext_storage.delete_pdf(key)
        except Exception:
            failed.append(key)
    return failed


def _queue(data: dict, keys) -> None:
    pending = data.setdefault("pending_pdf_deletions", [])
    known = {target.get("storage_key") for target in pending}
    for key in keys:
        if key and key not in known:
            pending.append({"paper_uid": None, "storage_key": key})
            known.add(key)


def attach(save, paper: dict, data: bytes, filename: str,
           page_count: int | None, digest: str, failed_cleanup: list[str],
           *, new_paper: bool = False) -> tuple[dict, bool]:
    candidate = deepcopy(save.data)
    uid = paper["uid"]
    target = next((item for item in candidate["papers"] if item["uid"] == uid), None)
    if new_paper:
        if target is not None:
            raise ValueError("This paper already exists in the project.")
        target = deepcopy(paper)
        candidate["papers"].append(target)
    elif target is None:
        raise ValueError("This paper is no longer in the project.")

    metadata = db.load_fulltexts(save.user_email, save.pid)
    old = metadata.get(uid) or {}
    same_source = bool(old.get("sha256") == digest and old.get("storage_key"))
    key = fulltext_storage.object_key(save.user_email, save.pid, uid,
                                     f"{digest}-{uuid4().hex}")
    entry = {
        "paper_uid": uid, "filename": filename, "storage_key": key, "sha256": digest,
        "file_size": len(data), "page_count": page_count, "status": "ok", "error": None, "text": None,
    }
    state.archive_document_results(
        target, old.get("sha256") or (f"unknown:{old['storage_key']}" if old.get("storage_key") else None),
        digest,
    )
    _queue(candidate, [old.get("storage_key")])
    try:
        fulltext_storage.save_pdf(key, data)
        save.commit(candidate, fulltext=entry)
    except BaseException:
        # A failed response can follow a successful commit; check before rollback.
        failed_cleanup.extend(cleanup_keys(save.user_email, [key]))
        raise
    return entry, not same_source


def remove(save, paper_uid: str) -> None:
    candidate = deepcopy(save.data)
    if not any(paper["uid"] == paper_uid for paper in candidate["papers"]):
        raise ValueError("This paper is no longer in the project.")
    metadata = db.load_fulltexts(save.user_email, save.pid)
    candidate["papers"] = [paper for paper in candidate["papers"] if paper["uid"] != paper_uid]
    candidate["cursors"][state.STAGE_FULLTEXT] = 0
    _queue(candidate, [(metadata.get(paper_uid) or {}).get("storage_key")])
    save.commit(candidate, remove_fulltext_uids=[paper_uid])


def replace_papers(save, rows: list[dict], source: tuple[list, list]) -> None:
    candidate = deepcopy(save.data)
    metadata = db.load_fulltexts(save.user_email, save.pid)
    candidate.update(papers=deepcopy(rows), cursors={})
    _queue(candidate, [item.get("storage_key") for item in metadata.values()])
    save.commit(candidate, source=source, remove_fulltexts=True)


def cleanup(save) -> bool:
    candidate = deepcopy(save.data)
    pending = candidate.get("pending_pdf_deletions", [])
    if not pending:
        return True
    active = {paper["uid"] for paper in candidate["papers"]}
    legacy_uids = [target["paper_uid"] for target in pending
                   if target.get("paper_uid") and target["paper_uid"] not in active]
    if legacy_uids:
        for target in pending:
            target["paper_uid"] = None
        save.commit(candidate, remove_fulltext_uids=legacy_uids)
    failed = cleanup_keys(save.user_email, [target.get("storage_key") for target in pending])
    remaining = [target for target in pending if target.get("storage_key") in failed]
    if remaining != pending:
        candidate["pending_pdf_deletions"] = remaining
        save.commit(candidate)
    return not remaining
