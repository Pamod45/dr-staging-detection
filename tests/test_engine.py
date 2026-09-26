"""Engine wiring on a random-weight model. Skipped without TensorFlow."""
import json

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from src import config as C                      # noqa: E402
from tests.helpers import fake_model_folder, fundus  # noqa: E402


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    root = tmp_path_factory.mktemp("models")
    mp = pytest.MonkeyPatch()
    mp.setattr(C, "MODELS_DIR", root)
    from src.model import Architecture, build, set_saved_state
    d = fake_model_folder(root)
    model, base = build(Architecture(768, "dense", 0.3, 0.5))
    set_saved_state(base, 0.5)
    (d / "best.weights.h5").unlink()
    model.save_weights(str(d / "best.weights.h5"))
    from src.inference import ScreeningEngine
    yield ScreeningEngine()
    mp.undo()


def test_matches_notebook_parameter_count(engine):
    assert engine.model.count_params() == 6_253_653
    assert engine.base.output.shape[1:] == (24, 24, 1280)


def test_run_produces_consistent_result(engine):
    r = engine.run(fundus())
    assert r.processed_rgb.shape == (768, 768, 3) and r.overlay_rgb.shape == (768, 768, 3)
    assert abs(r.probs.sum() - 1) < 1e-9 and r.decision.grade == int(r.probs.argmax())
    assert r.thresholds.referral == 0.35 and r.thresholds.abstention is None
    assert engine._cam_checked


def test_no_retina_raises(engine):
    from src.inference import NoRetinaFound
    with pytest.raises(NoRetinaFound):
        engine.run(np.zeros((400, 400, 3), np.uint8))


def test_normalisation_is_inside_the_model(engine):
    """0-255 and 0-1 inputs must give different outputs, or the rescaling layer is missing."""
    x = np.random.default_rng(0).uniform(0, 255, (1, 768, 768, 3)).astype("float32")
    a = engine.model.predict(x, verbose=0)
    b = engine.model.predict(x / 255, verbose=0)
    assert np.abs(a - b).max() > 1e-4


def test_trainable_counts_match_notebook(engine):
    from src.features import trainable_counts
    c = trainable_counts(engine)
    assert c["phase1"] == 331_781 and c["phase2"] == 4_941_037      # notebook 4.1 and 5.3
    assert (c["backbone_layers"], c["unfrozen_layers"]) == (270, 135)


def test_feature_maps_shapes(engine):
    from src.features import feature_maps
    r = engine.run(fundus())
    maps = feature_maps(engine, r.processed_rgb.astype("float32")[None])
    assert [m["shape"] for m in maps] == [(384, 384, 32), (192, 192, 32), (96, 96, 48),
                                          (48, 48, 112), (24, 24, 1280)]
    assert all(m["tiles"].dtype == np.uint8 and m["front"].shape == (256, 256, 3) for m in maps)
    assert [m["tuned"] for m in maps] == [False, False, False, True, True]   # cut at layer 135


def test_head_path_reproduces_the_probabilities(engine):
    from src.features import feature_maps, head_activations
    r = engine.run(fundus())
    maps = feature_maps(engine, r.processed_rgb.astype("float32")[None])
    head = head_activations(engine, maps[-1]["raw"])
    assert head["gap"].shape == (1280,) and head["dense_256"].shape == (256,)
    assert np.abs(head["grade"] - r.probs).max() < 1e-4


def test_tap_geometry_matches_the_network(engine):
    from src.features import BACKBONE_LAYERS, TAPS
    assert len(engine.base.layers) == BACKBONE_LAYERS
    for name, _, index, div, ch in TAPS:
        layer = engine.base.layers[index]
        assert layer.name == name and tuple(layer.output.shape[1:]) == (768 // div, 768 // div, ch)


def test_live_diagram_holds_every_image(engine):
    from src import cnn_diagram
    from src.features import feature_maps, head_activations
    r = engine.run(fundus())
    maps = feature_maps(engine, r.processed_rgb.astype("float32")[None])
    head = head_activations(engine, maps[-1]["raw"])
    svg, width = cnn_diagram.architecture_svg(768, "dense", 0.5, maps, head, r.processed_rgb)
    assert svg.count("data:image/png;base64,") == 1 + len(maps) and width > 1000
    assert "Severe" in svg and "%" in svg
    strip = cnn_diagram.gradcam_strip(r.processed_rgb, maps[-1]["front"], r.overlay_rgb,
                                      maps[-1]["shape"], "Moderate")
    assert strip.count("data:image/png;base64,") == 3
