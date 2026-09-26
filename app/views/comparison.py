"""Model comparison page."""
import pandas as pd
import streamlit as st

from src import config as C
from src import ui

ui.page_header(
    "Model comparison",
    "Models inside one comparison differ in a single setting, so any difference in "
    "results comes from that setting.",
)

comp = st.selectbox("Comparison", C.COMPARISONS, format_func=lambda c: c.title)
specs = [C.MODELS[m] for m in comp.members]

st.markdown(f"**What changed:** {comp.changed}")
st.caption(f"Measured on the {C.EVAL_SET_TEXT[specs[0].eval_set]}. Differences under "
           f"{C.NOISE_FLOOR_QWK} QWK are within run-to-run noise.")
if comp.note:
    st.info(comp.note)

st.dataframe(
    pd.DataFrame([{"Model": s.title, "Input size (px)": s.input_size} for s in specs]),
    hide_index=True,
)
