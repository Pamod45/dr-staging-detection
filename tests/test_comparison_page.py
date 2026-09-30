"""Model comparison page renders every comparison from stored files, and the upload block
shows precomputed outputs without loading a model."""
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src import config as C
from src import state
from tests.helpers import fake_full_store

MAIN = str(Path(__file__).resolve().parents[1] / "app" / "main.py")


@pytest.fixture
def store(tmp_path, monkeypatch):
    return fake_full_store(tmp_path, monkeypatch)


def _page(image=None, cache=None):
    at = AppTest.from_file(MAIN, default_timeout=60)
    if image is not None:
        at.session_state["shared_image"] = image
    if cache is not None:
        at.session_state["per_image_cache"] = cache
    at.run()
    at.switch_page("views/comparison.py").run()
    return at


@pytest.mark.parametrize("comp", C.COMPARISONS, ids=lambda c: c.id)
def test_each_comparison_renders(store, comp):
    at = _page()
    at.selectbox[0].set_value(comp).run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    assert len(at.dataframe) >= 3
    if not comp.live_upload:
        assert any("never ported" in i.value for i in at.info)


def test_missing_files_show_error(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "MODELS_DIR", tmp_path)
    at = _page()
    assert any("could not be found" in e.value for e in at.error)


def test_upload_block_shows_precomputed_outputs(store):
    import cv2
    from src.runner import RunOutput
    from tests.helpers import fundus
    ok, buf = cv2.imencode(".png", fundus(300, 300, 140, notch=False))
    img = state.decode(buf.tobytes(), "eye.png")
    comp = C.COMPARISONS_BY_ID["resolution"]
    rgb = np.zeros((64, 64, 3), np.uint8)
    outputs = {m: RunOutput(m, rgb, rgb, np.array([0.1, 0.6, 0.1, 0.1, 0.1]))
               for m in comp.members}
    at = _page(image=img, cache={f"compare:{comp.id}": {img.sha256: outputs}})
    assert not at.exception, at.exception
    text = " ".join(m.value for m in at.markdown)
    assert text.count("### Mild") == len(comp.members)


def test_v2_curves_fall_back_when_a_run_has_no_qwk(store):
    d = store / "v2_512_2000"
    lines = (d / "history.csv").read_text().splitlines()
    (d / "history.csv").write_text("\n".join(
        ["epoch,val_accuracy"] + [",".join([l.split(",")[0], l.split(",")[2]]) for l in lines[1:]]))
    at = _page()
    at.selectbox[0].set_value(C.COMPARISONS_BY_ID["data_resolution"]).run()
    assert not at.exception, at.exception
    assert any("validation accuracy is shown" in c.value for c in at.caption)


def test_best_value_per_row_is_highlighted():
    import pandas as pd
    import importlib.util
    from pathlib import Path as P
    spec = importlib.util.spec_from_file_location(
        "cmp", P(__file__).resolve().parents[1] / "app" / "views" / "comparison.py")
    src = spec.loader.get_data(spec.origin).decode()
    ns = {}
    exec(src.split("def setup_section")[0], ns)
    df = pd.DataFrame({"a": [0.8, 0.5], "b": [0.9, 0.5]}, index=["QWK", "Recall Mild"])
    html = ns["highlight_best"](df).to_html()
    assert html.count("font-weight: 700") >= 1
