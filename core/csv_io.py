from __future__ import annotations

import io

import pandas as pd

DOI_CANDIDATES = ["doi", "doi link", "doi url", "article doi"]
TITLE_CANDIDATES = ["title", "article title", "document title", "paper title"]
ABSTRACT_CANDIDATES = ["abstract", "summary", "abstract note", "abst"]


def read_csv(uploaded_file) -> pd.DataFrame:
    raw = (
        uploaded_file.getvalue()
        if hasattr(uploaded_file, "getvalue")
        else uploaded_file.read()
    )
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except Exception as exc:
            last_error = exc
    raise ValueError(f"Could not read the CSV file; check its format or encoding: {last_error}")


def guess_column(columns: list[str], candidates: list[str]) -> str | None:
    if not columns:
        return None
    by_lower = {str(c).lower().strip(): c for c in columns}
    for cand in candidates:
        if cand in by_lower:
            return by_lower[cand]
    for col in columns:
        col_lower = str(col).lower()
        if any(cand in col_lower for cand in candidates):
            return col
    return columns[0]


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")
