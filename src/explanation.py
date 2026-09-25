"""Plain-language explanation of one screening result.

Deterministic: the same result always gives the same text, and every number in it comes from
the result object. It describes what the model output, never what the eye has; it names image
positions (upper left...), never anatomy. This text is what goes into the PDF report.
"""
from src import config as C


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def threshold_pct(t: float) -> str:
    """0.35 -> '35%', 0.5496 -> '54.96%'. Never rounds a threshold into a different value."""
    v = round(100 * t, 2)
    return f"{v:.0f}%" if v == int(v) else f"{v:.2f}%"


def explain(result) -> list[str]:
    """Paragraphs, in reading order."""
    d, t, a, p = result.decision, result.thresholds, result.attention, result.probs
    paras = []

    if d.abstained:
        paras.append(
            f"The model's highest confidence for any single grade is {pct(d.confidence)}, below "
            f"the {threshold_pct(t.abstention)} needed to report a grade. No grade is given; "
            f"this image should be graded by a person."
        )
    else:
        paras.append(
            f"The most likely grade is {C.LABELS[d.grade]} (grade {d.grade} of 4), "
            f"with {pct(d.confidence)} confidence."
        )

    paras.append("Probability for each grade: "
                 + ", ".join(f"{C.LABELS[i]} {pct(p[i])}" for i in range(C.N_CLASSES)) + ".")

    parts = " + ".join(pct(p[i]) for i in C.REFERABLE_GRADES)
    names = ", ".join(C.LABELS[i] for i in C.REFERABLE_GRADES)
    side = "at or above" if d.refer else "below"
    outcome = ("so the image is flagged for referral to an eye specialist" if d.refer
               else "so the image is not flagged for referral")
    paras.append(
        f"The referable probability ({names} combined) is {pct(d.referable_prob)} "
        f"({parts}; values are rounded, so the parts may not add up exactly). This is {side} "
        f"the referral threshold of {threshold_pct(t.referral)}, {outcome}."
    )

    if a["hot_regions"] > 0:
        regions = "1 region" if a["hot_regions"] == 1 else f"{a['hot_regions']} regions"
        paras.append(
            f"The heatmap shows which parts of the image most influenced the result. "
            f"{a['inside_retina_pct']:.1f}% of the model's attention falls inside the retina. "
            f"The strongest attention covers {a['hot_area_pct']:.1f}% of the retina in {regions}, "
            f"centred in the {a['hot_location']} of the image."
        )
    else:
        paras.append("The heatmap shows no region of concentrated attention for this image.")

    notes = ["the heatmap marks broad areas, not individual lesions",
             "the referral decision is computed separately from the grade"]
    if d.abstained:
        notes.append("the grade was withheld, but the heatmap still shows attention for the "
                     "most likely grade")
    paras.append("Note: " + "; ".join(notes) + ". This is a model output, "
                 "not a clinical assessment.")
    return paras
