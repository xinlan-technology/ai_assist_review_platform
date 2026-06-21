import streamlit as st

st.set_page_config(page_title="AI Literature Review Platform", page_icon="🔎", layout="wide")

from core import auth
from features.screening import state

auth.require_login()

projects = st.Page("views/projects.py", title="My Projects", default=True)
screening = st.Page("views/relevance_screening.py", title="Relevance Screening")
summary = st.Page("views/review_summary.py", title="Review Summary")

pages = [projects]
if state.active_id():
    pages += [screening, summary]

navigation = st.navigation(pages)
if state.active_id() and st.session_state.pop("_go_screening", False):
    st.switch_page("views/relevance_screening.py")
navigation.run()
