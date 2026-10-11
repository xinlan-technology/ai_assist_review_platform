from __future__ import annotations

import csv
import io

import pandas as pd

DOI_CANDIDATES = ["doi", "doi link", "doi url", "article doi"]
TITLE_CANDIDATES = ["title", "article title", "document title", "paper title"]
ABSTRACT_CANDIDATES = ["abstract", "summary", "abstract note", "abst"]

MAX_CSV_BYTES = 20 * 1024 * 1024
MAX_CSV_ROWS = 20_000
MAX_CSV_COLUMNS = 100
MAX_CSV_CELL_CHARS = 100_000
MAX_CSV_CELLS = 1_000_000


def _validate_csv(text: str) -> None:
    """Check resource budgets before pandas allocates a full table."""
    rows = columns = 0
    try:
        for row in csv.reader(io.StringIO(text, newline=""), strict=True):
            if not row:
                continue
            columns = max(columns, len(row))
            if columns > MAX_CSV_COLUMNS:
                raise ValueError(f"The CSV exceeds the {MAX_CSV_COLUMNS:,}-column limit.")
            if any(len(value) > MAX_CSV_CELL_CHARS for value in row):
                raise ValueError(f"A CSV cell exceeds {MAX_CSV_CELL_CHARS:,} characters.")
            rows += 1
            if rows - 1 > MAX_CSV_ROWS:
                raise ValueError(f"The CSV exceeds the {MAX_CSV_ROWS:,}-row limit.")
            # Include padded cells when input rows have different lengths.
            if (rows - 1) * columns > MAX_CSV_CELLS:
                raise ValueError(f"The CSV exceeds the {MAX_CSV_CELLS:,}-cell limit.")
    except csv.Error:
        raise ValueError("The CSV has invalid quoting or an excessively large cell.") from None


def read_csv(uploaded_file) -> pd.DataFrame:
    raw = (
        uploaded_file.getvalue()
        if hasattr(uploaded_file, "getvalue")
        else uploaded_file.read(MAX_CSV_BYTES + 1)
    )
    if len(raw) > MAX_CSV_BYTES:
        raise ValueError(f"The CSV exceeds the {MAX_CSV_BYTES // (1024 * 1024)} MB upload limit.")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    _validate_csv(text)
    try:
        return pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError):
        raise ValueError("Could not read the CSV file; check its columns and quoting.") from None


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
    # Require manual mapping when no column matches.
    return None


def _excel_safe(value):
    if isinstance(value, str) and (
        value.startswith(("\t", "\r", "\n")) or value.lstrip().startswith(("=", "+", "-", "@"))
    ):
        return "'" + value
    return value


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Escape formula-like cells and headers without changing source data."""
    safe = df.apply(lambda column: column.map(_excel_safe))
    safe.columns = [_excel_safe(column) for column in df.columns]
    return safe.to_csv(index=False).encode("utf-8-sig")
