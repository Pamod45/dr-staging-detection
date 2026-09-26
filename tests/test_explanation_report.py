"""Explanation text and PDF report."""
from src.explanation import explain, threshold_pct
from src.report import build_pdf
from tests.helpers import fake_result

import numpy as np


def test_threshold_formatting_never_rounds_into_another_value():
    assert threshold_pct(0.35) == "35%" and threshold_pct(0.5496) == "54.96%"


def test_graded_text_names_grade_and_numbers():
    text = " ".join(explain(fake_result([0.05, 0.05, 0.7, 0.1, 0.1])))
    assert "Moderate (grade 2 of 4), with 70.0% confidence" in text
    assert "90.0%" in text and "at or above the referral threshold of 35%" in text
    assert "upper left" in text and "2 regions" in text


def test_withheld_grade_is_not_stated():
    paras = explain(fake_result([0.3, 0.3, 0.2, 0.1, 0.1], abstention=0.5))
    assert "No grade is given" in paras[0] and "most likely grade is" not in " ".join(paras)
    assert "50%" in paras[0]


def test_not_referred_wording():
    text = " ".join(explain(fake_result([0.9, 0.08, 0.01, 0.005, 0.005])))
    assert "below the referral threshold" in text and "not flagged for referral" in text


def test_pdf_builds_in_both_states():
    for r in (fake_result([0.05, 0.05, 0.7, 0.1, 0.1]),
              fake_result([0.3, 0.3, 0.2, 0.1, 0.1], abstention=0.5)):
        pdf = build_pdf(r, np.zeros((400, 600, 3), np.uint8), "eye.jpg", "abc")
        assert pdf[:5] == b"%PDF-" and len(pdf) > 5000
