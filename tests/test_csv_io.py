"""Text-preserving CSV imports and spreadsheet-safe exports."""
from io import BytesIO

import pandas as pd
from pandas.testing import assert_frame_equal
import pytest


from core import csv_io


def test_import_preserves_identifiers_missing_markers_empty_values_and_unicode():
    data = 'ID,Title,Abstract,Region,Year\n00123,NA,"Line one\nLine two",NA,2024\n00002,Caf\u00e9,,NULL,\n'
    frame = csv_io.read_csv(BytesIO(data.encode("utf-8-sig")))
    assert frame.to_dict("records") == [
        {"ID": "00123", "Title": "NA", "Abstract": "Line one\nLine two", "Region": "NA", "Year": "2024"},
        {"ID": "00002", "Title": "Caf\u00e9", "Abstract": "", "Region": "NULL", "Year": ""},
    ]


def test_import_supports_latin1_without_numeric_coercion():
    frame = csv_io.read_csv(BytesIO("ID,Title\n001,Caf\u00e9\n".encode("latin-1")))
    assert frame.loc[0, "ID"] == "001"
    assert frame.loc[0, "Title"] == "Caf\u00e9"


@pytest.mark.parametrize("limit,value,data,message", [
    ("MAX_CSV_BYTES", 8, b"Title\nlong value\n", "upload limit"),
    ("MAX_CSV_ROWS", 1, b"Title\none\ntwo\n", "row limit"),
    ("MAX_CSV_COLUMNS", 2, b"A,B,C\n1,2,3\n", "column limit"),
    ("MAX_CSV_COLUMNS", 2, b"A,B\n1,2,3\n", "column limit"),
    ("MAX_CSV_CELL_CHARS", 5, b"Title\n123456\n", "characters"),
    ("MAX_CSV_CELL_CHARS", 5, b"123456\none\n", "characters"),
    ("MAX_CSV_CELLS", 3, b"A,B\n1,2\n3,4\n", "cell limit"),
    ("MAX_CSV_CELLS", 3, b"A,B\n1\n2\n", "cell limit"),
])
def test_import_rejects_resource_limits_before_pandas(monkeypatch, limit, value, data, message):
    monkeypatch.setattr(csv_io, limit, value)
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: pytest.fail("must not allocate table"))
    with pytest.raises(ValueError, match=message):
        csv_io.read_csv(BytesIO(data))


def test_stream_input_read_is_bounded_before_parsing(monkeypatch):
    monkeypatch.setattr(csv_io, "MAX_CSV_BYTES", 8)

    class Stream:
        def read(self, size):
            assert size == 9
            return b"x" * size

    with pytest.raises(ValueError, match="upload limit"):
        csv_io.read_csv(Stream())


def test_import_accepts_exact_limits_and_preserves_pandas_headers(monkeypatch):
    raw = b'Title,Title\n"12345",abcde\n'
    monkeypatch.setattr(csv_io, "MAX_CSV_BYTES", len(raw))
    monkeypatch.setattr(csv_io, "MAX_CSV_ROWS", 1)
    monkeypatch.setattr(csv_io, "MAX_CSV_COLUMNS", 2)
    monkeypatch.setattr(csv_io, "MAX_CSV_CELL_CHARS", 5)
    monkeypatch.setattr(csv_io, "MAX_CSV_CELLS", 2)
    frame = csv_io.read_csv(BytesIO(raw))
    assert frame.columns.tolist() == ["Title", "Title.1"]
    assert frame.iloc[0].tolist() == ["12345", "abcde"]


def test_multiline_quoted_cells_count_as_one_row(monkeypatch):
    monkeypatch.setattr(csv_io, "MAX_CSV_ROWS", 1)
    frame = csv_io.read_csv(BytesIO(b'Title,Abstract\nStudy,"line 1\nline 2"\n'))
    assert frame["Abstract"].tolist() == ["line 1\nline 2"]


def test_invalid_quoting_is_not_retried_as_another_encoding(monkeypatch):
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: pytest.fail("invalid CSV must not parse"))
    with pytest.raises(ValueError, match="invalid quoting"):
        csv_io.read_csv(BytesIO(b'Title\n"unterminated\n'))


def test_pandas_parser_errors_are_not_retried(monkeypatch):
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise pd.errors.ParserError("Synthetic parser failure")

    monkeypatch.setattr(pd, "read_csv", fail)
    with pytest.raises(ValueError, match="columns and quoting"):
        csv_io.read_csv(BytesIO(b"Title\nStudy\n"))
    assert calls == [1]


def test_export_neutralizes_formula_strings_and_column_headers_without_mutation():
    text = ["=1+1", "+1", "-1", "@SUM(1,2)", "  =1+1", "\tplain", "\nplain", "\rplain", "ordinary", "'already safe"]
    frame = pd.DataFrame({"=Formula header": text, "Value": [-2.5] * len(text)})
    original = frame.copy(deep=True)
    exported = csv_io.to_csv_bytes(frame)
    assert exported.startswith(b"\xef\xbb\xbf")
    reread = pd.read_csv(BytesIO(exported), dtype=str, keep_default_na=False)
    assert reread.columns.tolist() == ["'=Formula header", "Value"]
    assert reread.iloc[:, 0].tolist() == ["'" + value for value in text[:8]] + text[8:]
    assert reread["Value"].tolist() == ["-2.5"] * len(text)
    assert_frame_equal(frame, original)


def test_empty_export_keeps_safe_headers_and_does_not_mutate_frame():
    frame = pd.DataFrame(columns=["@Header", "Title"])
    assert csv_io.to_csv_bytes(frame).decode("utf-8-sig") == "'@Header,Title\n"
    assert frame.columns.tolist() == ["@Header", "Title"]


def test_export_preserves_empty_cells_and_multiline_quotes():
    frame = pd.DataFrame({"Text": [None, "", 'A "quoted"\nobservation']})
    result = csv_io.read_csv(BytesIO(csv_io.to_csv_bytes(frame)))
    assert result["Text"].tolist() == ["", "", 'A "quoted"\nobservation']
