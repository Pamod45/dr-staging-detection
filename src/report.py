import io
from datetime import datetime

import numpy as np
from PIL import Image

from src import config as C
from src.charts import probability_png
from src.explanation import explain, pct, threshold_pct


def _latin1(text: str) -> str:
    """fpdf2's built-in fonts are Latin-1 only; swap the few characters that are not."""
    for a, b in {"\u2013": "-", "\u2014": "-", "\u2019": "'", "\u00d7": "x"}.items():
        text = text.replace(a, b)
    return text.encode("latin-1", "replace").decode("latin-1")


def _square_thumb(rgb: np.ndarray, side: int = 600) -> Image.Image:
    img = Image.fromarray(rgb)
    img.thumbnail((side, side))
    canvas = Image.new("RGB", (side, side), (0, 0, 0))
    canvas.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    return canvas


def build_pdf(result, original_rgb: np.ndarray, image_name: str, image_id: str,
              when: datetime | None = None) -> bytes:
    from fpdf import FPDF

    when = when or datetime.now()
    d, t = result.decision, result.thresholds
    spec = C.MODELS[result.model_id]

    pdf = FPDF(format="A4", unit="mm")
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()
    pdf.set_margins(15, 15, 15)
    width = pdf.w - 30

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(width, 9, "Diabetic retinopathy screening report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(90, 90, 90)
    pdf.cell(width, 5, _latin1(f"{when:%Y-%m-%d %H:%M}   Image: {image_name} (id {image_id})"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(width, 5, _latin1(f"Model: {result.model_id} - {spec.title}, "
                               f"{result.input_size} x {result.input_size} px input"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)

    y, gap = pdf.get_y() + 4, 4
    side = (width - 2 * gap) / 3
    for k, (img, cap) in enumerate([(original_rgb, "Original"),
                                    (result.processed_rgb, "Model input"),
                                    (result.overlay_rgb, "Grad-CAM")]):
        x = 15 + k * (side + gap)
        pdf.image(_square_thumb(img), x=x, y=y, w=side, h=side)
        pdf.set_xy(x, y + side + 1)
        pdf.set_font("Helvetica", "", 8)
        pdf.cell(side, 4, cap, align="C")
    pdf.set_y(y + side + 8)

    grade = "Not graded (confidence below threshold)" if d.abstained else C.LABELS[d.grade]
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(width / 2, 6, _latin1(f"Grade: {grade}"))
    pdf.cell(width / 2, 6, "Referral: " + ("REFER to eye specialist" if d.refer else "Not referred"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(width / 2, 5, f"Confidence {pct(d.confidence)}")
    pdf.cell(width / 2, 5, f"Referable probability {pct(d.referable_prob)} "
                           f"(threshold {threshold_pct(t.referral)})", new_x="LMARGIN", new_y="NEXT")

    chart = Image.open(io.BytesIO(probability_png(result.probs, None if d.abstained else d.grade)))
    pdf.image(chart, x=15, y=pdf.get_y() + 2, w=width * 0.75)
    pdf.set_y(pdf.get_y() + 2 + width * 0.75 * chart.height / chart.width + 3)

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(width, 6, "Explanation", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    for para in explain(result):
        pdf.multi_cell(width, 4.6, _latin1(para), align="L", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1.5)

    if t.abstention is None:
        used = (f"Referral threshold {threshold_pct(t.referral)}, fitted on this model's "
                f"validation data. No abstention threshold applied.")
    else:
        used = (f"Referral threshold {threshold_pct(t.referral)} and abstention threshold "
                f"{threshold_pct(t.abstention)}, both fitted on this model's validation data.")
    pdf.set_y(pdf.h - 24)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(90, 90, 90)
    pdf.multi_cell(width, 4, _latin1(f"{used}\n{C.DISCLAIMER}"))
    return bytes(pdf.output())
