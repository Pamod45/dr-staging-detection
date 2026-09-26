"""How it was built page."""
import streamlit as st

from src import config as C
from src import ui

spec = C.MODELS[C.SCREENING_MODEL_ID]
ui.page_header(
    "How it was built",
    f"The final model ({spec.title}) from photograph to grade, shown on a real image.",
)

tabs = st.tabs(["Preprocessing", "Augmentation and imbalance", "Architecture", "Training"])
with tabs[0]:
    st.markdown(f"How a photograph becomes the {spec.input_size} x {spec.input_size} px "
                "input the model reads.")
with tabs[1]:
    st.markdown("How the training images were varied, and how rare grades were given "
                "enough weight.")
with tabs[2]:
    st.markdown("What the network computes, layer by layer, on the way to a grade.")
with tabs[3]:
    st.markdown(f"Two training phases: the new output layers alone for "
                f"{C.V2_PHASE1_EPOCHS} epochs, then fine-tuning the upper backbone.")
