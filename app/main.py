"""Entry point: streamlit run app/main.py"""
import sys
from pathlib import Path
import shutil
import psutil
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import ui  

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

def get_folder_size(path):
    total = 0
    for file in Path(path).rglob("*"):
        if file.is_file():
            try:
                total += file.stat().st_size
            except OSError:
                pass
    return total


with st.sidebar.expander("System usage"):
    process = psutil.Process()

    ram_mb = process.memory_info().rss / 1024 / 1024
    project_gb = get_folder_size(ROOT) / 1024**3

    st.write(f"RAM: {ram_mb:.0f} MB / 2700 MB")
    st.progress(min(ram_mb / 2700, 1.0))

    st.write(f"Project size: {project_gb:.2f} GB / 50 GB")
    st.progress(min(project_gb / 50, 1.0))

nav.run()
ui.disclaimer()
