"""Reliability figures are plain counts on the stored test predictions."""
import numpy as np
import pytest

from src import trust

Y = np.array([2, 2, 1, 3, 0, 2])
P = np.array([
    [0.05, 0.05, 0.80, 0.05, 0.05],
    [0.10, 0.10, 0.65, 0.10, 0.05],
    [0.10, 0.25, 0.55, 0.05, 0.05],
    [0.02, 0.03, 0.72, 0.20, 0.03],
    [0.90, 0.05, 0.02, 0.02, 0.01],
    [0.60, 0.10, 0.20, 0.05, 0.05],
])


def test_same_grade_calls_and_what_they_really_were():
    r = trust.reliability(Y, P, grade=2, conf=0.7, refer=True, threshold=0.35)
    assert (r["grade_calls"], r["grade_right"]) == (4, 2)
    assert r["grade_truths"] == {1: 1, 2: 2, 3: 1}


def test_confidence_band():
    r = trust.reliability(Y, P, grade=2, conf=0.72, refer=True, threshold=0.35)
    assert r["band"] == (0.7, 0.8)
    assert (r["band_n"], r["band_right"], r["band_within_one"]) == (1, 0, 1)


def test_top_band_includes_certainty():
    assert trust.band(1.0) == (0.9, 1.0) and trust.band(0.9) == (0.9, 1.0)
    assert trust.band(0.31) == (0.0, 0.5)


def test_referral_calls():
    r = trust.reliability(Y, P, grade=2, conf=0.7, refer=True, threshold=0.35)
    assert (r["referral_calls"], r["referral_right"]) == (4, 3)
    r = trust.reliability(Y, P, grade=0, conf=0.9, refer=False, threshold=0.35)
    assert (r["referral_calls"], r["referral_right"]) == (2, 1)


def test_text_says_where_numbers_come_from(tmp_path, monkeypatch):
    from tests.helpers import fake_full_store, fake_result
    fake_full_store(tmp_path, monkeypatch)
    text = trust.reliability_text(fake_result([0.05, 0.05, 0.7, 0.1, 0.1]))
    assert "1,788 DDR test images" in text and "When the model's top grade was Moderate" in text
    withheld = trust.reliability_text(fake_result([0.3, 0.3, 0.2, 0.1, 0.1], abstention=0.5))
    assert "top grade was" not in withheld


def test_missing_files_give_no_block(tmp_path, monkeypatch):
    from src import config as C
    from tests.helpers import fake_result
    monkeypatch.setattr(C, "MODELS_DIR", tmp_path)
    assert trust.reliability_text(fake_result([0.9, 0.05, 0.02, 0.02, 0.01])) is None
