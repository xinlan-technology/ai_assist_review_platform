from __future__ import annotations

import re
from urllib.parse import quote, unquote

import streamlit as st

from core.llm import PROVIDERS, PROVIDER_KEY_HELP

ACCENT = "#0E6E55"

_BASE_CSS = """
<style>
[data-testid="stAppDeployButton"] {display: none !important;}
#MainMenu {display: none !important;}
[data-testid="stDecoration"] {display: none !important;}
footer {display: none !important;}
[data-testid="stHeader"] {background: transparent;}

[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {
    display: flex !important; visibility: visible !important; opacity: 1 !important;
}

.block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1040px;}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF;
    border: 1px solid #E7E9ED;
    border-radius: 10px;
}

h3 {font-size: 1.05rem; font-weight: 600; color: #1C2127;}

.stButton > button[kind="primary"] {
    background: #0E6E55; border: 1px solid #0E6E55; color: #FFFFFF; border-radius: 8px;
}
.stButton > button[kind="primary"]:hover {background: #0C5E48; border-color: #0C5E48;}
.stDownloadButton > button {border-radius: 8px;}
</style>
"""

_HEADER_TEMPLATE = """
<div style="display:flex; align-items:center; gap:11px; margin:0 0 0.6rem;">
  <div style="width:30px; height:30px; border-radius:8px; background:{accent};
              display:flex; align-items:center; justify-content:center;">
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#fff"
         stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"></polygon>
    </svg>
  </div>
  <div>
    <div style="font-size:19px; font-weight:600; color:#1C2127; line-height:1.15;">{title}</div>
    <div style="font-size:12px; color:#9AA1AC;">{subtitle}</div>
  </div>
</div>
"""


def inject_base_css() -> None:
    st.markdown(_BASE_CSS, unsafe_allow_html=True)


def app_header(title: str, subtitle: str) -> None:
    st.markdown(
        _HEADER_TEMPLATE.format(accent=ACCENT, title=title, subtitle=subtitle),
        unsafe_allow_html=True,
    )


def doi_url(doi: str) -> str:
    # Decode first so an already encoded DOI is not encoded twice.
    doi = re.sub(r"^(?:https?://(?:dx\.|www\.)?doi\.org/|doi:\s*)", "", unquote(doi.strip()), flags=re.I)
    return (f"https://doi.org/{quote(doi, safe='/')}"
            if re.fullmatch(r"10\.\d{4,9}(?:\.\d+)*/\S+", doi) else "")


def escape_markdown(value: object) -> str:
    """Render untrusted text without Markdown images, links, or HTML."""
    # Every ASCII punctuation character may be escaped, which also keeps URLs,
    # dollar signs and colon shortcodes literal.
    return re.sub(r"([!-/:-@\[-`{-~])", r"\\\1", str(value))


def jump_to_paper(key: str, position: int, labels: list[str]) -> int:
    """Return the cursor, honoring external moves passed in ``f'{key}:moved'``."""
    if len(labels) < 2:
        return position
    # External moves take precedence over picker changes and stale selections.
    shown_key = f"{key}:shown"
    moved = st.session_state.pop(f"{key}:moved", None)
    chosen = st.session_state.get(key)
    last_shown = st.session_state.get(shown_key)
    if isinstance(moved, int) and 0 <= moved < len(labels):
        position = moved
    elif (isinstance(chosen, int) and 0 <= chosen < len(labels)
            and chosen != last_shown):
        position = chosen
    st.session_state[key] = position
    st.session_state[shown_key] = position
    st.selectbox(
        "Jump to paper", range(len(labels)), key=key,
        format_func=lambda index: labels[index], label_visibility="collapsed",
    )
    return position


def model_controls() -> tuple[str, str, str]:
    """Share controls across pages while keeping API keys separate by provider."""
    st.subheader("Configuration")
    provider = st.selectbox("Provider", list(PROVIDERS), key="llm:provider")
    model = st.selectbox("Model", PROVIDERS[provider], key=f"llm:model:{provider}")
    api_key = st.text_input("API key", type="password", help=PROVIDER_KEY_HELP[provider],
                            key=f"llm:api_key:{provider}")
    st.caption("Your key is kept in this app session, not in project settings. "
               "It authenticates requests to the selected model provider; "
               "recorded API errors are redacted.")
    # Provider keys contain no whitespace; a pasted tab or line break is noise.
    return provider, model, "".join(api_key.split())


def additional_models(provider: str, model: str, api_key: str) -> list[dict]:
    models = [{"provider": provider, "model": model, "api_key": api_key}]
    if not st.checkbox("Compare multiple models", key="llm:compare"):
        return models
    count = st.selectbox("Number of models", [2, 3], key="llm:count")
    keys = {provider: api_key}
    for slot in range(2, count + 1):
        providers = list(PROVIDERS)
        default = next((i for i, p in enumerate(providers) if p not in keys), 0)
        selected = st.selectbox(f"Provider {slot}", providers, index=default, key=f"llm:provider:{slot}")
        name = st.selectbox(f"Model {slot}", PROVIDERS[selected], key=f"llm:model:{slot}:{selected}")
        if selected not in keys:
            keys[selected] = "".join(st.text_input(
                f"API key · {selected}", type="password", help=PROVIDER_KEY_HELP[selected],
                key=f"llm:api_key:{selected}",
            ).split())
        models.append({"provider": selected, "model": name, "api_key": keys[selected]})
    st.caption("Each selected model receives the same input in a separate paid call. "
               "Keys stay in this session; human decisions are not changed by extra runs.")
    return models


def pending_file_cleanup() -> None:
    """Keep failed upload/project cleanup visible until a retry succeeds."""
    from core import auth
    from features.workflow import documents

    key = "_uncommitted_pdf_cleanup"
    if not st.session_state.get(key):
        return
    st.warning("Some PDF files still need cleanup. Keep this session open and retry; "
               "they have not been confirmed deleted from storage.")
    if st.button("Retry pending file cleanup", key="retry_pending_file_cleanup"):
        pending = st.session_state[key]
        failed = documents.cleanup_keys(auth.current_user(), pending)
        pending[:] = failed
        st.rerun()
