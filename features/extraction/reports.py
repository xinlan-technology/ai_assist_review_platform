"""Current extraction summaries and versioned audit exports."""
import json

import pandas as pd

from . import state


def answer_text(answer):
    if not isinstance(answer, dict):
        return ""
    values = answer.get("values", [])
    text = values[0] if len(values) == 1 else json.dumps(values, ensure_ascii=False) if values else ""
    return f"{text} ({answer['other_text']})" if answer.get("other_text") else text


def summary(papers, spec, metadata):
    counts = {"total": len(papers), "ai_done": 0, "confirmed": 0,
              "invalid_fields": 0, "outdated": 0, "failed": 0}
    for paper in papers:
        record = state.get(paper)
        digest = metadata.get(paper["uid"], {}).get("sha256", "")
        status = state.status(paper, spec, digest)
        counts["outdated"] += status == "Outdated"
        if status == "Outdated":
            continue
        counts["ai_done"] += "ai_answers" in record
        counts["confirmed"] += status == "Confirmed"
        counts["failed"] += bool(record.get("ai_error")) and status != "Confirmed"
        if status != "Confirmed":
            errors = "review_errors" if status == "Draft" else "field_errors"
            counts["invalid_fields"] += len(record.get(errors, {}))
    return counts


def final_dataframe(papers, spec, metadata):
    columns = ["Paper UID", "Title", "DOI", "Extraction status"]
    questions = spec["questions"]
    labels = {q["id"]: f"{q['text']} [{q['id']}]" for q in questions}
    rows = []
    for paper in papers:
        digest = metadata.get(paper["uid"], {}).get("sha256", "")
        answers = state.confirmed_answers(paper, spec, digest)
        row = {"Paper UID": paper["uid"], "Title": paper.get("title", ""),
               "DOI": paper.get("doi", ""), "Extraction status": state.status(paper, spec, digest)}
        row.update({labels[q["id"]]: answer_text(answers.get(q["id"])) for q in questions})
        rows.append(row)
    return pd.DataFrame(rows, columns=columns + list(labels.values()))


def audit_dataframe(papers, eligible_uids, spec, metadata):
    columns = ["Paper UID", "Title", "DOI", "Version status", "Question ID", "Question",
               "Type", "Options", "Question guidance", "AI answer", "Recorded final", "Current final", "Decision",
               "AI page", "AI quote", "Final page", "Final quote", "Field error", "Review error", "AI error",
               "Source run ID", "Provider", "Model", "Prompt version", "Spec hash", "PDF SHA256", "Instructions",
               "AI completed at", "Draft saved at", "Reviewed at", "Archived at"]
    rows = []
    for paper in papers:
        current = state.get(paper)
        digest = metadata.get(paper["uid"], {}).get("sha256", "")
        eligible = paper["uid"] in eligible_uids
        final = state.confirmed_answers(paper, spec, digest) if eligible else {}
        versions = [(r, "Archived") for r in current.get("history", [])]
        versions.append((current, state.status(paper, spec, digest) if eligible else "Ineligible"))
        for record, status in versions:
            snapshot = record.get("spec", spec)
            for q in snapshot["questions"]:
                qid = q["id"]
                ai = record.get("source_answers", {}).get(qid, record.get("ai_answers", {}).get(qid, {}))
                source_id = record.get("source_run_ids", {}).get(qid, record.get("source_run_id", ""))
                source = record.get("source_metadata", {}).get(qid, {})
                baseline = not source_id or source_id == record.get("source_run_id")
                raw = record.get("invalid_answers", {}).get(qid) if baseline else None
                human = record.get("final_answers", {}).get(qid, {})
                human = human if isinstance(human, dict) else {}
                rows.append({
                    "Paper UID": paper["uid"], "Title": paper.get("title", ""), "DOI": paper.get("doi", ""),
                    "Version status": status, "Question ID": qid, "Question": q["text"], "Type": q["type"],
                    "Options": json.dumps(q["options"], ensure_ascii=False),
                    "Question guidance": q["guidance"],
                    "AI answer": json.dumps(ai or raw, ensure_ascii=False) if ai or raw is not None else "",
                    "Recorded final": json.dumps(human, ensure_ascii=False) if human else "",
                    "Current final": answer_text(final.get(qid)) if status == "Confirmed" else "",
                    "Decision": record.get("decisions", {}).get(qid, ""),
                    "AI page": ai.get("page"), "AI quote": ai.get("quote", ""),
                    "Final page": human.get("page"), "Final quote": human.get("quote", ""),
                    "Field error": record.get("field_errors", {}).get(qid, "") if baseline else "",
                    "Review error": record.get("review_errors", {}).get(qid, ""),
                    "AI error": record.get("ai_error", "") if baseline else "", "Source run ID": source_id,
                    "Provider": source.get("provider", record.get("provider", "") if baseline else ""),
                    "Model": source.get("model", record.get("model", "") if baseline else ""),
                    "Prompt version": record.get("prompt_version", ""),
                    "Spec hash": record.get("spec_hash", ""), "PDF SHA256": record.get("source_sha256", ""),
                    "Instructions": snapshot["instructions"],
                    "AI completed at": source.get("completed_at", record.get("completed_at", "") if baseline else ""),
                    "Draft saved at": record.get("draft_saved_at", ""),
                    "Reviewed at": record.get("reviewed_at", ""), "Archived at": record.get("archived_at", ""),
                })
    return pd.DataFrame(rows, columns=columns)


def choice_counts(papers, spec, metadata):
    rows = []
    confirmed = [state.confirmed_answers(p, spec, metadata.get(p["uid"], {}).get("sha256", ""))
                 for p in papers]
    confirmed = [answers for answers in confirmed if answers]
    for q in spec["questions"]:
        if q["type"] == "open_text":
            continue
        for option in q["options"]:
            count = sum(option in answers[q["id"]]["values"] for answers in confirmed)
            rows.append({"Question": q["text"], "Question ID": q["id"], "Option": option,
                         "Papers": count, "Confirmed papers": len(confirmed)})
    return pd.DataFrame(rows)
