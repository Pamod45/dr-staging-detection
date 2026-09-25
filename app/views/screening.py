"""Screening - owns the shared upload. Model output is added in stage 2."""
import streamlit as st

from src import config as C
from src import registry, state, ui

ui.page_header(
    "Screening",
    "Upload one retinal photograph. The model grades it on the five-step ICDR scale "
    "and says whether it should be referred to an eye specialist.",
)

upload = st.file_uploader(
    "Retinal photograph (JPG or PNG)",
    type=["jpg", "jpeg", "png"],
    key=state.uploader_key(),
)

if upload is not None:
    try:
        img = state.decode(upload.getvalue(), upload.name)
    except ValueError as err:
        st.error(str(err))
    else:
        if state.set_image(img):
            st.rerun()

if registry.check_model(C.SCREENING_MODEL_ID)["status"] != "ok":
    st.error("The grading model could not be found, so images cannot be graded right now. "
             "Run scripts/check_models.py for details.")

img = state.get_image()
if img is not None:
    left, _ = st.columns(2)
    with left:
        h, w = img.image_bgr.shape[:2]
        st.image(img.image_rgb, caption=f"Original, {w} x {h} px", width="stretch")
