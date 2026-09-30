import io

import altair as alt
import numpy as np
import pandas as pd

from src import config as C

ACCENT = "#4FA3C7"
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


def confusion_chart(cm: np.ndarray, title: str = "") -> alt.Chart:
    rows = cm.sum(1, keepdims=True).clip(min=1)
    df = pd.DataFrame([{"True": C.LABELS[i], "Predicted": C.LABELS[j], "Count": int(cm[i, j]),
                        "Row share": cm[i, j] / rows[i, 0]}
                       for i in range(C.N_CLASSES) for j in range(C.N_CLASSES)])
    base = alt.Chart(df, title=title).encode(
        x=alt.X("Predicted:N", sort=list(C.LABELS), axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("True:N", sort=list(C.LABELS)),
    )
    # colour by row share so rare grades are readable next to the 897-image No DR row
    cells = base.mark_rect().encode(
        color=alt.Color("Row share:Q", scale=alt.Scale(scheme="blues", domain=[0, 1]),
                        legend=None),
        tooltip=["True", "Predicted", "Count", alt.Tooltip("Row share:Q", format=".1%")],
    )
    text = base.mark_text(fontSize=12).encode(
        text="Count:Q",
        color=alt.condition("datum['Row share'] > 0.5", alt.value("white"), alt.value("#1B2430")),
    )
    return (cells + text).properties(height=300)


def training_curves(df: pd.DataFrame, series: dict, phase_switch: float | None = None,
                    best_epoch: int | None = None, y_title: str = "") -> alt.Chart:
    long = df.melt(id_vars="epoch", value_vars=list(series), var_name="key", value_name="value")
    long = long.dropna()
    long["Series"] = long["key"].map(series)
    layers = [alt.Chart(long).mark_line(point=True).encode(
        x=alt.X("epoch:Q", title="Epoch"),
        y=alt.Y("value:Q", title=y_title, scale=alt.Scale(zero=False)),
        color=alt.Color("Series:N", legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Series", "epoch", alt.Tooltip("value:Q", format=".4f")],
    )]
    marks = []
    if phase_switch is not None:
        marks.append({"epoch": phase_switch, "Marker": "Fine-tuning starts"})
    if best_epoch is not None:
        marks.append({"epoch": best_epoch, "Marker": f"Best epoch ({best_epoch})"})
    if marks:
        layers.append(alt.Chart(pd.DataFrame(marks)).mark_rule(strokeDash=[4, 4]).encode(
            x="epoch:Q", tooltip=["Marker"], color=alt.value("#9AA7B0")))
    return alt.layer(*layers).properties(height=300)


def referral_curve_chart(curve: pd.DataFrame, chosen: float, fitted: float) -> alt.Chart:
    long = curve.melt(id_vars="threshold", value_vars=["sensitivity", "specificity"],
                      var_name="Measure", value_name="value")
    lines = alt.Chart(long).mark_line().encode(
        x=alt.X("threshold:Q", title="Referral threshold", axis=alt.Axis(format="%")),
        y=alt.Y("value:Q", title=None, axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Measure:N", legend=alt.Legend(orient="bottom", title=None)),
        tooltip=[alt.Tooltip("threshold:Q", format=".2f"), "Measure",
                 alt.Tooltip("value:Q", format=".2%")],
    )
    rules = alt.Chart(pd.DataFrame([{"t": fitted, "Line": "Fitted on validation"},
                                    {"t": chosen, "Line": "Selected"}])).mark_rule().encode(
        x="t:Q", strokeDash=alt.StrokeDash("Line:N", legend=alt.Legend(orient="bottom",
                                                                       title=None)),
        color=alt.value("#9AA7B0"))
    return (lines + rules).properties(height=300)


def schedule_chart(histories: dict) -> alt.Chart:
    names = {"fixed_lr": "Fixed rate", "plateau": "Cut on plateau", "cosine": "Cosine decay"}
    frames, cuts = [], []
    for key, df in histories.items():
        frames.append(pd.DataFrame({"epoch": df["epoch"], "QWK": df["val_qwk"],
                                    "Schedule": names.get(key, key)}))
        if key == "plateau" and "learning_rate" in df:
            lr = df["learning_rate"].to_numpy()
            # learning_rate[i] is the rate used during epoch i, so a drop between two
            # entries means the cut fell between those epochs
            for e in np.nonzero(lr[1:] < lr[:-1] * 0.99)[0] + 1:
                cuts.append({"epoch": float(df["epoch"].iloc[e]) - 0.5,
                             "Cut": f"Rate {lr[e - 1]:.0e} to {lr[e]:.0e}"})
    lines = alt.Chart(pd.concat(frames)).mark_line(point=True).encode(
        x=alt.X("epoch:Q", title="Epoch"),
        y=alt.Y("QWK:Q", title="Validation QWK", scale=alt.Scale(zero=False)),
        color=alt.Color("Schedule:N", legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Schedule", "epoch", alt.Tooltip("QWK:Q", format=".4f")],
    )
    if not cuts:
        return lines.properties(height=300)
    rules = alt.Chart(pd.DataFrame(cuts)).mark_rule(strokeDash=[4, 4]).encode(
        x="epoch:Q", tooltip=["Cut"], color=alt.value("#9AA7B0"))
    return (lines + rules).properties(height=300)


def metric_bars(rows: list[dict], metrics_: tuple = ("QWK", "Macro F1")) -> alt.Chart:
    df = pd.DataFrame([{"Model": r["Model"], "Metric": m, "Value": r[m]}
                       for r in rows for m in metrics_])
    order = [r["Model"] for r in rows]
    bars = alt.Chart(df).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("Model:N", sort=order, title=None, axis=alt.Axis(labelAngle=0, labelLimit=160)),
        xOffset=alt.XOffset("Metric:N"),
        y=alt.Y("Value:Q", title=None, scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Metric:N", legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Model", "Metric", alt.Tooltip("Value:Q", format=".4f")],
    )
    text = bars.mark_text(dy=-6, fontSize=11).encode(text=alt.Text("Value:Q", format=".3f"),
                                                     color=alt.value("#E3EAEE"))
    return (bars + text).properties(height=280)
