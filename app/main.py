"""Entry point.  Run from the repo root:   streamlit run app/main.py

Builds the navigation, draws the shared sidebar, runs the selected page, then the
disclaimer. Pages live in app/views/ (not app/pages/, which Streamlit would try to
auto-load as a second navigation system).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from src import ui  # noqa: E402

st.set_page_config(
    page_title="DR grading prototype",
    page_icon=":material/visibility:",
    layout="wide",
)

pages = {
    "Start here": [
        st.Page("views/home.py", title="About diabetic retinopathy",
                icon=":material/menu_book:", default=True),
        st.Page("views/screening.py", title="Screening", icon=":material/upload:"),
    ],
    "How it works": [
        st.Page("views/how_built.py", title="How it was built", icon=":material/account_tree:"),
        st.Page("views/results.py", title="Results and decisions", icon=":material/insights:"),
        st.Page("views/comparison.py", title="Model comparison", icon=":material/compare:"),
    ],
}

nav = st.navigation(pages)
ui.sidebar_status()
nav.run()
ui.disclaimer()
