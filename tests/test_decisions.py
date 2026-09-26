"""Decision rules at their boundaries."""
import json

import pytest

from src import config as C
from src.decisions import Thresholds, decide, load_thresholds
from tests.helpers import fake_model_folder

T = Thresholds(referral=0.375, abstention=0.5)


def test_referral_at_threshold_refers():
    d = decide([0.625, 0, 0.125, 0.125, 0.125], T)
    assert d.refer and d.referable_prob == 0.375


def test_referral_just_below_does_not():
    assert not decide([0.625 + 2**-10, 0, 0.125, 0.125, 0.125 - 2**-10], T).refer


def test_abstains_below_threshold_but_still_refers():
    d = decide([0.25, 0, 0.25, 0.25, 0.25], T)
    assert d.abstained and d.refer


def test_no_abstention_when_threshold_missing():
    assert not decide([0.25, 0, 0.25, 0.25, 0.25], Thresholds(0.375, None)).abstained


def test_rejects_wrong_length():
    with pytest.raises(ValueError):
        decide([0.5, 0.5], T)


def test_thresholds_read_from_results_json(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "MODELS_DIR", tmp_path)
    fake_model_folder(tmp_path, results={"referral_threshold": 0.35})
    assert load_thresholds() == Thresholds(0.35, None)
    fake_model_folder(tmp_path, results={"referral_threshold": 0.35, "abstention_threshold": 0.61})
    assert load_thresholds() == Thresholds(0.35, 0.61)


def test_missing_referral_threshold_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "MODELS_DIR", tmp_path)
    fake_model_folder(tmp_path, results={"accuracy": 0.9})
    with pytest.raises(KeyError):
        load_thresholds()
