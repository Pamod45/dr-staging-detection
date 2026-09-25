"""Preprocessing port: geometry and masking on synthetic frames."""
import numpy as np

from src.preprocess import circle_mask, detect_fov, preprocess, to_model_input
from tests.helpers import fundus


def test_detects_disc_centre_and_radius():
    fov = detect_fov(fundus(notch=False))
    assert abs(fov.cx - 675) < 3 and abs(fov.cy - 450) < 3 and abs(fov.r - 440) < 4


def test_notch_does_not_move_the_disc():
    a, b = detect_fov(fundus(notch=False)), detect_fov(fundus(notch=True))
    assert abs(a.cy - b.cy) < 2


def test_output_is_square_masked_and_uint8():
    out, _ = preprocess(fundus(), out=768)
    assert out.shape == (768, 768, 3) and out.dtype == np.uint8
    assert out[:8, :8].max() == 0 and out[-8:, -8:].max() == 0
    # the resize softens the mask edge by a pixel, so allow a 1% margin (notebook checks corners only)
    assert out[~circle_mask(768, 1.01)].max() == 0
    dark = (out.max(2) == 0).mean()
    assert 0.20 < dark < 0.30          # ~1 - pi/4 = 0.215 outside the circle, plus dark spots


def test_blank_image_has_no_retina():
    out, fov = preprocess(np.zeros((500, 500, 3), np.uint8), out=768)
    assert out is None and fov is None


def test_model_input_is_rgb_0_255_float():
    out, _ = preprocess(fundus(), out=768)
    x = to_model_input(out, 768)
    assert x.dtype == np.float32 and x.max() <= 255 and x.max() > 1
    assert np.array_equal(x[..., 0], out[..., 2].astype(np.float32))
