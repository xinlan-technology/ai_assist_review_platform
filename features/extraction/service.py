"""Sequential PDF extraction with durable checkpoints."""
from time import perf_counter

from core import fulltext_storage
from core.llm import InvalidModelResponse, error_message
from features.extraction import judge, state


def run(papers, spec, metadata, provider, model, api_key, save, progress=None):
    if not save():
        return False
    for index, paper in enumerate(papers):
        meta = metadata[paper["uid"]]
        started = perf_counter()
        try:
            pdf = fulltext_storage.load_pdf(meta["storage_key"])
            if fulltext_storage.sha256(pdf) != meta["sha256"]:
                raise ValueError("The stored PDF differs from its metadata. Reattach it first.")
            result = judge.extract_pdf(provider, model, api_key, spec, pdf, meta["filename"])
            state.set_ai_result(
                paper, spec, meta["sha256"], result, provider, model, meta.get("page_count")
            )
        except InvalidModelResponse as exc:
            state.set_error(paper, spec, meta["sha256"], error_message(exc, api_key), "invalid_response")
        except Exception as exc:
            state.set_error(paper, spec, meta["sha256"], error_message(exc, api_key), "call_failed")
        state.get(paper).update(provider=provider, model=model,
                                duration_seconds=round(perf_counter() - started, 3))
        if not save():
            return False
        if progress:
            progress(index + 1, len(papers))
    return True
