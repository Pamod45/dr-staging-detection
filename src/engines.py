"""Models shared across pages. One cache per server process, so the screening model loaded on
the Screening page is the same object the How it was built page uses."""
import streamlit as st


@st.cache_resource(show_spinner="Loading the grading model...")
def screening_engine():
    from src.inference import ScreeningEngine
    return ScreeningEngine()


@st.cache_resource(max_entries=4, show_spinner=False)
def comparison_runner(model_id: str):
    # max_entries bounds memory: at most one comparison's models stay loaded at a time
    from src.runner import ModelRunner
    return ModelRunner(model_id)
