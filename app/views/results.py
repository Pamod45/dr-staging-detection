"""Results and decisions - the decision trail (v1 blocks 1-7), then the shipped v2 model.
Explanatory text, the learning-rate chart and the final-model blocks are added in stage 3.
Missing figures are skipped silently here; scripts/check_models.py is where gaps are reported.
"""
import streamlit as st

from src import config as C
from src import ui

ui.page_header(
    "Results and decisions",
    "Each design choice in the order it was made: the question, what was tested, "
    "and the evidence that decided it.",
)

st.subheader("How the design was chosen")
st.caption("Measured on the earlier notebook's " + C.EVAL_SET_TEXT["val"] + ".")
for i, (title, question, figure) in enumerate(C.RESULTS_BLOCKS, start=1):
    if figure is None or not figure.exists():
        continue
    with st.container(border=True):
        st.markdown(f"**{i}. {title}**  \n{question}")
        st.image(str(figure), width="stretch")
