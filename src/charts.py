"""Probability charts: an Altair chart for the page (follows the app theme) and a matplotlib
PNG for the PDF report (white background, prints cleanly)."""
import io

import altair as alt
import numpy as np
import pandas as pd

from src import config as C

ACCENT = "#4FA3C7"   # matches primaryColor in .streamlit/config.toml
MUTED = "#5C6B75"


def probability_chart(probs: np.ndarray, highlight: int | None) -> alt.Chart:
    df = pd.DataFrame({"Grade": list(C.LABELS), "Probability": probs,
                       "Top": [i == highlight for i in range(C.N_CLASSES)]})
    df["Label"] = [f"{100 * v:.1f}%" for v in probs]
    y = alt.Y("Grade:N", sort=list(C.LABELS), title=None)
    bars = alt.Chart(df).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"),
                title=None),
        y=y,
        color=alt.condition("datum.Top", alt.value(ACCENT), alt.value(MUTED)),
        tooltip=["Grade", "Label"],
    )
    text = bars.mark_text(align="left", dx=4).encode(text="Label:N", color=alt.value("#E3EAEE"))
    return (bars + text).properties(height=210)


def probability_png(probs: np.ndarray, highlight: int | None) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 2.4), dpi=200)
    colors = [ACCENT if i == highlight else "#9AA7B0" for i in range(C.N_CLASSES)]
    ax.barh(list(C.LABELS)[::-1], probs[::-1], color=colors[::-1])
    for i, v in enumerate(probs[::-1]):
        ax.text(v + 0.01, i, f"{100 * v:.1f}%", va="center", fontsize=8)
    ax.set_xlim(0, 1.1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()
