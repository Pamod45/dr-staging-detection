"""Results and decisions: the v1 decision trail, then the shipped v2 model's results and its
referral threshold."""
import numpy as np
import pandas as pd
import streamlit as st

from src import charts, data, metrics, registry, ui
from src import config as C
from src.decision_trail import FINAL_NOTE, TRAIL
from src.pages_assets import FINAL_FIGURES

FINAL = C.SCREENING_MODEL_ID


@st.cache_data
def model_summary(model_id: str) -> dict:
    return metrics.summary(data.labels(model_id), data.probs(model_id))


@st.cache_data
def referral_table(model_id: str) -> pd.DataFrame:
    return pd.DataFrame(metrics.referral_curve(data.labels(model_id), data.probs(model_id)))


def available(model_id: str) -> bool:
    return registry.check_model(model_id)["status"] == "ok"


def metrics_table(columns: dict[str, dict]) -> pd.DataFrame:
    df = pd.DataFrame(columns)
    return df.map(lambda v: f"{v:.4f}" if isinstance(v, float) else v)


def decision_block(n: int, d) -> None:
    with st.container(border=True):
        st.markdown(f"#### {n}. {d.title}")
        st.markdown(f"*{d.question}*")
        st.markdown(f"**Tested.** {d.tested}")
        st.markdown(f"**Decided.** {d.decided}")
        st.markdown(f"**Cost.** {d.cost}")
        if d.models and all(available(m) for m in d.models):
            st.dataframe(metrics_table({C.MODELS[m].title: model_summary(m) for m in d.models}),
                         width="stretch")
        if d.lr_chart and data.STRATEGY_HISTORY.exists():
            st.altair_chart(charts.schedule_chart(data.strategy_histories()), width="stretch")
        if d.qwk_curves and all(available(m) for m in d.models):
            curves = None
            for m in d.models:
                h = data.history(m)[["epoch", "val_qwk"]].rename(columns={"val_qwk": m})
                curves = h if curves is None else curves.merge(h, on="epoch", how="outer")
            st.altair_chart(charts.training_curves(
                curves, {m: C.MODELS[m].title for m in d.models}, y_title="Validation QWK"),
                width="stretch")
        if d.figure is not None and d.figure.exists():
            st.image(str(d.figure), width="stretch")


def design_tab() -> None:
    st.caption(f"Earlier notebook, measured on its {C.EVAL_SET_TEXT['val']}. Differences under "
               f"{C.NOISE_FLOOR_QWK} QWK are within run-to-run noise.")
    for n, d in enumerate(TRAIL, start=1):
        decision_block(n, d)
    st.info(FINAL_NOTE)


def final_tab() -> None:
    if not available(FINAL):
        st.error("The final model's stored results could not be found.")
        return
    spec = C.MODELS[FINAL]
    res = data.run_results(FINAL)
    st.caption(f"{spec.title}. DDR numbers are on its {C.EVAL_SET_TEXT['test']}; IDRiD is 455 "
               "images from a different hospital and camera, never used in training.")

    cols = {"DDR test": model_summary(FINAL)}
    idrid = data.stored_summary(FINAL, "idrid")
    if idrid:
        cols["IDRiD (external)"] = idrid
    st.dataframe(metrics_table(cols), width="stretch")

    y, p = data.labels(FINAL), data.probs(FINAL)
    left, right = st.columns(2)
    with left:
        st.altair_chart(charts.confusion_chart(metrics.confusion(y, p.argmax(1)),
                                               "DDR test, rows are the true grade"),
                        width="stretch")
    with right:
        fig = FINAL_FIGURES["idrid_confusion"]
        if fig.exists():
            st.image(str(fig), caption="IDRiD external test", width="stretch")

    hist = data.history(FINAL)
    series = {c: n for c, n in [("val_qwk", "Validation QWK"),
                                ("val_accuracy", "Validation accuracy"),
                                ("clean_accuracy", "Training accuracy (no augmentation)")]
              if c in hist}
    st.markdown("**Training curves**")
    st.altair_chart(charts.training_curves(hist, series, phase_switch=C.V2_PHASE1_EPOCHS + 0.5,
                                           best_epoch=res.get("best_epoch")),
                    width="stretch")
    st.caption(f"The first {C.V2_PHASE1_EPOCHS} epochs train only the new output layers; "
               "fine-tuning of the upper backbone starts after the dashed line. The saved "
               "model is the epoch with the lowest validation loss.")


def referral_tab() -> None:
    if not available(FINAL):
        st.error("The final model's stored results could not be found.")
        return
    fitted = float(data.run_results(FINAL)["referral_threshold"])
    curve = referral_table(FINAL)
    y = data.labels(FINAL)
    n_ref = int((y >= 2).sum())

    st.markdown(
        f"A case is referred when P(Moderate) + P(Severe) + P(PDR) reaches the threshold. "
        f"The threshold was fitted on the validation set as the highest value that still "
        f"refers at least 95% of referable cases: **{fitted:.2f}**. Missing a sight-threatening "
        f"case costs more than an extra appointment, so the fit favours sensitivity."
    )
    t = st.slider("Referral threshold", 0.01, 0.99, fitted, 0.01)
    row = curve.loc[np.isclose(curve["threshold"], t)].iloc[0]
    a, b, c, d = st.columns(4)
    a.metric("Sensitivity", f"{row['sensitivity']:.1%}")
    b.metric("Specificity", f"{row['specificity']:.1%}")
    c.metric("Missed referable cases", f"{int(row['missed'])} of {n_ref}")
    d.metric("Share referred", f"{row['referred_pct']:.1f}%")
    st.altair_chart(charts.referral_curve_chart(curve, t, fitted), width="stretch")
    st.caption(f"Measured on the {C.EVAL_SET_TEXT['test']}, which played no part in choosing "
               "the threshold.")


ui.page_header(
    "Results and decisions",
    "Each design choice in the order it was made, with the evidence that decided it, then "
    "how the final model performs.",
)
tabs = st.tabs(["Design decisions", "Final model", "Referral threshold"])
with tabs[0]:
    design_tab()
with tabs[1]:
    final_tab()
with tabs[2]:
    referral_tab()
