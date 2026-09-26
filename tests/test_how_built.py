"""How it was built: display helpers reproduce the notebook, and the page renders."""
from pathlib import Path

import cv2
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src import config as C
from src.augment import AugSettings, augment
from src.preprocess import circle_mask, preprocess, preprocess_steps
from tests.helpers import V2_TRAINING_CONFIG, fake_full_store, fundus

MAIN = str(Path(__file__).resolve().parents[1] / "app" / "main.py")


def test_steps_end_in_the_same_image_as_preprocess():
    img = fundus()
    assert np.array_equal(preprocess_steps(img, 768)["final"], preprocess(img, 768)[0])


def _notebook_augment(img, rng, c):
    """v2 notebook 3.3 augment(), copied verbatim with Config values passed in."""
    side = img.shape[0]
    zoom = rng.uniform(*c["ZOOM"])
    M = cv2.getRotationMatrix2D((side / 2, side / 2), rng.uniform(0, c["ROTATION"]), zoom)
    x = cv2.warpAffine(img, M, (side, side), flags=cv2.INTER_LINEAR, borderValue=0)
    if rng.random() < 0.5:
        x = x[:, ::-1]
    if rng.random() < 0.5:
        x = x[::-1]
    x = np.ascontiguousarray(x)
    inside = circle_mask(side, min(zoom, 1.0))
    mean = np.array(cv2.mean(x, mask=inside.astype(np.uint8))[:3], np.float32)
    x = x.astype(np.float32)
    x = (x - mean) * (1 + rng.uniform(-c["CONTRAST"], c["CONTRAST"])) + mean
    x = x * (1 + rng.uniform(-c["BRIGHTNESS"], c["BRIGHTNESS"])) * \
        (1 + rng.uniform(-c["CHANNEL"], c["CHANNEL"], 3))
    x = np.clip(x, 0, 255).astype(np.uint8)
    x[~inside] = 0
    return x


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_augment_matches_notebook_draw_for_draw(seed):
    img, _ = preprocess(fundus(), 256)
    ours, _ = augment(img, np.random.default_rng(seed), AugSettings.from_config(V2_TRAINING_CONFIG))
    theirs = _notebook_augment(img, np.random.default_rng(seed), V2_TRAINING_CONFIG)
    assert np.array_equal(ours, theirs)


@pytest.mark.parametrize("seed", [0, 3])
def test_augment_steps_end_in_the_same_image(seed):
    from src.augment import augment_steps
    img, _ = preprocess(fundus(), 256)
    s = AugSettings.from_config(V2_TRAINING_CONFIG)
    steps = augment_steps(img, np.random.default_rng(seed), s)
    final, _ = augment(img, np.random.default_rng(seed), s)
    assert [x["title"] for x in steps][0] == "Model input" and len(steps) == 6
    assert np.array_equal(steps[-1]["image"], final)


def test_static_diagram_needs_no_model():
    from src import cnn_diagram
    svg, width = cnn_diagram.architecture_svg(768, "dense", 0.5)
    assert "data:image" not in svg and "384 x 384 x 32" in svg and "24 x 24 x 1,280" in svg
    assert svg.count("Fine-tuned") == 2 + 1          # two tuned stacks plus the legend
    svg_simple, _ = cnn_diagram.architecture_svg(768, "simple", 0.5)
    assert "Dense layer" not in svg_simple


@pytest.fixture
def page_store(tmp_path, monkeypatch):
    fake_full_store(tmp_path / "models", monkeypatch)
    samples = tmp_path / "samples"
    samples.mkdir()
    cv2.imwrite(str(samples / "sample.jpg"), fundus(600, 800, 290))
    monkeypatch.setattr(C, "SAMPLES_DIR", samples)
    split = tmp_path / "split.csv"
    split.write_text("id_code,diagnosis,split\n" + "\n".join(
        f"x{i}.png,{g},train" for i, g in enumerate([0] * 50 + [1] * 5 + [2] * 30 + [3] * 2 + [4] * 6)))
    monkeypatch.setattr(C, "SPLIT_FILES", {**C.SPLIT_FILES, "v2": split})
    return tmp_path


def test_page_renders_with_sample_image(page_store):
    at = AppTest.from_file(MAIN, default_timeout=120)
    at.run()
    at.switch_page("views/how_built.py").run()
    assert not at.exception, at.exception
    assert any("built-in sample" in c.value for c in at.caption)
    assert len(at.tabs) == 4
    assert len(at.image) >= 6 + 7 + 8        # preprocessing, augmentation steps, draws
    text = " ".join(m.value for m in at.markdown)
    assert "50 / 2 = 25 to 1" in text        # imbalance arithmetic from the split file
    at.button[0].click().run()               # "Draw again"
    assert not at.exception, at.exception


def test_page_without_model_files(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "MODELS_DIR", tmp_path)
    at = AppTest.from_file(MAIN, default_timeout=60)
    at.run()
    at.switch_page("views/how_built.py").run()
    assert not at.exception, at.exception
    assert any("could not be found" in e.value for e in at.error)
