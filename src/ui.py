"""Small UI pieces reused by every page, so all five pages look and behave the same."""
import streamlit as st

from src import config as C
from src import state

SCREENING_PAGE = "views/screening.py"


def page_header(title: str, blurb: str) -> None:
    st.title(title)
    st.markdown(blurb)


def sidebar_status() -> None:
    """Current shared image in the sidebar, visible from every page."""
    with st.sidebar:
        st.divider()
        img = state.get_image()
        if img is None:
            st.caption("No image loaded.")
            return
        st.image(img.image_rgb, caption=img.name, width="stretch")
        if st.button("Clear image", width="stretch"):
            state.clear_image()
            st.rerun()


def needs_image(purpose: str) -> bool:
    """Show a pointer to Screening when no image is loaded. Returns True if one is loaded."""
    if state.get_image() is not None:
        return True
    st.info(f"Upload a retinal photograph on the Screening page to {purpose}.")
    st.page_link(SCREENING_PAGE, label="Go to Screening", icon=":material/upload:")
    return False


def disclaimer() -> None:
    st.divider()
    st.caption(C.DISCLAIMER)
