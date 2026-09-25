"""Every page renders without error through the real navigation, with and without an image."""
from pathlib import Path

import cv2
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src import config as C
from src import state

MAIN = str(Path(__file__).resolve().parents[1] / "app" / "main.py")
VIEWS = ["home", "screening", "how_built", "results", "comparison"]


def _sample_image():
    img = np.zeros((64, 64, 3), np.uint8)
    cv2.circle(img, (32, 32), 28, (40, 80, 160), -1)
    ok, buf = cv2.imencode(".png", img)
    return state.decode(buf.tobytes(), "sample.png")


def _app(view, with_image=False):
    at = AppTest.from_file(MAIN, default_timeout=30)
    if with_image:
        at.session_state["shared_image"] = _sample_image()
    at.run()
    at.switch_page(f"views/{view}.py").run()
    return at


def test_decode_rejects_non_image():
    with pytest.raises(ValueError):
        state.decode(b"not an image", "notes.txt")


def test_decode_hash_is_stable():
    a, b = _sample_image(), _sample_image()
    assert a.sha256 == b.sha256 and a.image_bgr.shape == (64, 64, 3)


@pytest.mark.parametrize("view", VIEWS)
@pytest.mark.parametrize("with_image", [False, True])
def test_view_renders(view, with_image):
    at = _app(view, with_image)
    assert not at.exception, at.exception
    assert any(C.DISCLAIMER in c.value for c in at.caption)


def test_clear_image_button_resets_state():
    at = _app("home", with_image=True)
    at.sidebar.button[0].click().run()
    assert "shared_image" not in at.session_state


def test_every_comparison_renders():
    at = _app("comparison")
    box = at.selectbox[0]
    for i in range(len(box.options)):
        at.selectbox[0].set_value(C.COMPARISONS[i]).run()
        assert not at.exception, (C.COMPARISONS[i].id, at.exception)


DEV_WORDS = ("stage", "prototype", "coursework", "placeholder", "todo", "coming soon")


@pytest.mark.parametrize("view", VIEWS)
def test_no_build_notes_shown_to_users(view):
    """Pages read as a finished product: no build stages or placeholder wording on screen."""
    at = _app(view, with_image=True)
    shown = [e.value for kind in ("title", "subheader", "markdown", "caption", "info",
                                  "warning", "error") for e in getattr(at, kind)]
    for text in shown:
        low = str(text).lower()
        assert not any(w in low for w in DEV_WORDS), (view, text)
