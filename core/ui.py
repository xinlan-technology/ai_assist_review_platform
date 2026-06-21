from __future__ import annotations

import streamlit as st

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
