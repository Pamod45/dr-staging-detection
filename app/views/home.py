"""About diabetic retinopathy page."""
import streamlit as st

from src import config as C
from src import ui

ui.page_header(
    "About diabetic retinopathy",
    "What the disease is, how it is graded, and what this tool does with a photograph.",
)

st.subheader("The five ICDR grades")
cols = st.columns(len(C.LABELS))
for grade, (col, label) in enumerate(zip(cols, C.LABELS)):
    with col:
        with st.container(border=True):
            st.markdown(f"**Grade {grade}**  \n{label}")

st.subheader("Where to go next")
st.page_link("views/screening.py", label="Screen an image", icon=":material/upload:")
st.page_link("views/how_built.py", label="See how it was built", icon=":material/account_tree:")
st.page_link("views/results.py", label="See the evidence behind each decision",
             icon=":material/insights:")
