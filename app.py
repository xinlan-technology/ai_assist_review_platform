import streamlit as st

st.set_page_config(page_title="AI Literature Review Platform", page_icon="🔎", layout="wide")

from core import auth
from features.extraction import drafts
from features.workflow import state

auth.require_login()

projects_page = st.Page("views/projects.py", title="My Projects", default=True)
abstract_page = st.Page("views/abstract_screening.py", title="Abstract Screening")
fulltext_page = st.Page("views/fulltext_screening.py", title="Full-text Screening")
extraction_page = st.Page("views/extraction.py", title="Data Extraction")
summary_page = st.Page("views/review_summary.py", title="Review Summary")

state.require_saved_results()

pages = [projects_page]
if state.active_id() and state.loaded():
    if state.mode() == state.MODE_PRISMA:
        pages += [abstract_page]
    pages += [fulltext_page, extraction_page, summary_page]

navigation = st.navigation(pages)
if state.active_id() and state.loaded() and st.session_state.pop("_go_workflow", False):
    first = ("views/abstract_screening.py" if state.mode() == state.MODE_PRISMA
             else "views/fulltext_screening.py")
    st.switch_page(first)
if state.active_id() and state.loaded():
    review_form = f"extraction:{state.active_id()}"
    if navigation.title != extraction_page.title:
        # Edits typed on the extraction page are kept before its form disappears.
        drafts.leave(review_form)
    drafts.resolve(review_form)
navigation.run()
