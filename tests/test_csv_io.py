"""Text-preserving CSV imports and spreadsheet-safe exports."""
from io import BytesIO

import pandas as pd
from pandas.testing import assert_frame_equal


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
