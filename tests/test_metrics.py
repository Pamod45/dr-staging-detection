"""NumPy metrics agree with scikit-learn, which the notebooks used."""
import numpy as np
import pytest

from src import metrics

sk = pytest.importorskip("sklearn.metrics")


@pytest.fixture
def sample():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 500)
    p = rng.dirichlet(np.ones(5), 500)
    p[np.arange(250), y[:250]] += 2
    return y, p / p.sum(1, keepdims=True)


def test_qwk_matches_sklearn(sample):
    y, p = sample
    yh = p.argmax(1)
    assert metrics.qwk(y, yh) == pytest.approx(sk.cohen_kappa_score(y, yh, weights="quadratic"))


def test_macro_f1_matches_sklearn(sample):
    y, p = sample
    yh = p.argmax(1)
    want = sk.f1_score(y, yh, average="macro", labels=range(5), zero_division=0)
    assert metrics.macro_f1(y, yh) == pytest.approx(want)


def test_macro_f1_with_a_missing_class():
    y, yh = np.array([0, 0, 2, 2]), np.array([0, 2, 2, 2])
    want = sk.f1_score(y, yh, average="macro", labels=range(5), zero_division=0)
    assert metrics.macro_f1(y, yh) == pytest.approx(want)


def test_recall_matches_sklearn(sample):
    y, p = sample
    yh = p.argmax(1)
    want = sk.recall_score(y, yh, labels=range(5), average=None, zero_division=0)
    assert np.allclose(metrics.recall_per_class(y, yh), want)


def test_referral_counts():
    y = np.array([0, 1, 2, 3, 4])
    p = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0.7, 0, 0.3, 0, 0],
                  [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]], float)
    r = metrics.referral(y, p, 0.35)
    assert r["missed"] == 1 and r["sensitivity"] == pytest.approx(2 / 3)
    assert r["specificity"] == 1.0 and r["referred_pct"] == pytest.approx(40.0)
