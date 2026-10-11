"""Untrusted display text must not become remote images or arbitrary DOI links."""
from pathlib import Path
import tomllib

import pytest

from core import ui


@pytest.mark.parametrize("value", [
    "![image](https://example.invalid/track)",
    "<img src='https://example.invalid/track'>",
    "[link](javascript:alert(1))",
    "\\![image](https://example.invalid/track)",
])
def test_markdown_control_characters_are_escaped(value):
    escaped = ui.escape_markdown(value)
    for index, character in enumerate(escaped):
        if character in "[]<>!":
            preceding = len(escaped[:index]) - len(escaped[:index].rstrip("\\"))
            assert preceding % 2 == 1


@pytest.mark.parametrize("value", [
    "https://example.invalid/track", "javascript:alert(1)", "not a DOI", "10.1234/has space",
])
def test_invalid_doi_never_becomes_an_arbitrary_link(value):
    assert ui.doi_url(value) == ""


@pytest.mark.parametrize("value", ["10.1234/study(1)", "https://doi.org/10.1234/study(1)",
                                    "DOI: 10.1234/study(1)"])
def test_doi_links_use_the_resolver_and_encoded_path(value):
    assert ui.doi_url(value) == "https://doi.org/10.1234/study%281%29"


def test_browser_hides_uncaught_exception_details():
    config = Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml"
    assert tomllib.loads(config.read_text())["client"]["showErrorDetails"] == "none"


@pytest.mark.parametrize("value, expected", [
    ("https://doi.org/10.1002/%28SICI%291097", "https://doi.org/10.1002/%28SICI%291097"),
    ("https://www.doi.org/10.1000/abc", "https://doi.org/10.1000/abc"),
    ("10.1000.10/abc", "https://doi.org/10.1000.10/abc"),
    ("http://dx.doi.org/10.1000/ABC-1", "https://doi.org/10.1000/ABC-1"),
])
def test_common_doi_spellings_resolve_to_the_same_record(value, expected):
    assert ui.doi_url(value) == expected


@pytest.mark.parametrize("value", [
    "See https://platform.example.invalid/docs/error-codes.",
    "Cost of $5 and $10", ":smile: 10:30", "a_b-c.d", "Heading\n===",
])
def test_every_punctuation_character_is_escaped_so_text_stays_literal(value):
    escaped = ui.escape_markdown(value)
    for index, character in enumerate(escaped):
        if character != "\\" and not character.isalnum() and not character.isspace():
            assert escaped[index - 1] == "\\", (character, escaped)
    assert escaped.replace("\\", "") == value.replace("\\", "")
