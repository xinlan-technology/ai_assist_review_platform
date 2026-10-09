"""Exercise the real chart and PDF components without files or live services."""
from streamlit.testing.v1 import AppTest


def test_pdf_and_chart_components_load_and_rerun():
    app = AppTest.from_string('''
import io
import streamlit as st
from pypdf import PdfWriter

output = io.BytesIO()
writer = PdfWriter()
writer.add_blank_page(width=200, height=200)
writer.write(output)
st.pdf(output.getvalue(), height=300)
st.bar_chart({"Papers": [1, 2]})
''', default_timeout=20)
    app.run()
    assert not app.exception
    assert len(app.get("bidi_component")) == 1
    app.run()
    assert not app.exception
