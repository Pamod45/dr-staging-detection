"""Models shared across pages. One cache per server process, so the screening model loaded on
the Screening page is the same object the How it was built page uses."""
import gc
import os

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


def low_memory() -> bool:
    """LOW_MEMORY = "1" in the host's secrets (or environment): keep one comparison model in
    memory at a time. Streamlit Community Cloud gives far less memory than a laptop."""
    value = os.environ.get("LOW_MEMORY")
    if value is None:
        try:
            value = st.secrets.get("LOW_MEMORY")
        except Exception:
            value = None
    return str(value or "").lower() in ("1", "true", "yes")


def release_comparison_models() -> None:
    comparison_runner.clear()
    gc.collect()
