"""Results and decisions: the v1 decision trail, then the shipped v2 model's results, its
referral threshold and its abstention threshold."""
import numpy as np
import pandas as pd
import streamlit as st

from src import charts, data, metrics, registry, ui
from src import config as C
from src.decision_trail import FINAL_NOTE, TRAIL
from src.pages_assets import FINAL_FIGURES

FINAL = C.SCREENING_MODEL_ID
ABSTENTION_TARGET = 0.90
ABSTENTION_GRID = np.round(np.arange(0.20, 1.0, 0.01), 2)
SPLIT_NAMES = {"val": "DDR validation", "test": "DDR test", "idrid": "IDRiD (external)"}


@st.cache_data
def model_summary(model_id: str) -> dict:
    return metrics.summary(data.labels(model_id), data.probs(model_id))


@st.cache_data
def referral_table(model_id: str) -> pd.DataFrame:
    return pd.DataFrame(metrics.referral_curve(data.labels(model_id), data.probs(model_id)))


@st.cache_data
def split_summary(model_id: str, split: str) -> dict | None:
    arrays = data.split_arrays(model_id, split)
    return None if arrays is None else metrics.summary(*arrays)


@st.cache_data
def abstention_table(model_id: str, split: str) -> pd.DataFrame | None:
    arrays = data.split_arrays(model_id, split)
    if arrays is None:
        return None
    return pd.DataFrame(metrics.abstention_curve(*arrays, ABSTENTION_GRID))


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
    idrid = split_summary(FINAL, "idrid") or data.stored_summary(FINAL, "idrid")
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
        external = data.split_arrays(FINAL, "idrid")
        if external is not None:
            yi, pi = external
            st.altair_chart(charts.confusion_chart(metrics.confusion(yi, pi.argmax(1)),
                                                   "IDRiD external test"), width="stretch")
        elif FINAL_FIGURES["idrid_confusion"].exists():
            st.image(str(FINAL_FIGURES["idrid_confusion"]), caption="IDRiD external test",
                     width="stretch")

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


def abstention_tab() -> None:
    if not available(FINAL):
        st.error("The final model's stored results could not be found.")
        return
    fitted = data.run_results(FINAL).get("abstention_threshold")
    tables = {s: abstention_table(FINAL, s) for s in SPLIT_NAMES}
    if fitted is None or tables["val"] is None:
        st.info("No abstention threshold has been fitted for this model, so every image is "
                "given a grade.")
        return
    fitted = float(fitted)

    st.markdown(
        f"When the model's highest grade probability is below a cut-off, the grade is withheld "
        f"and the image is left for a person to grade. The cut-off was fitted on the validation "
        f"set as the lowest confidence at which the graded images are still at least "
        f"{ABSTENTION_TARGET:.0%} correct: **{fitted:.4f}**. Referral is decided separately, so "
        f"a withheld image can still be referred.")
    options = sorted(set(ABSTENTION_GRID.tolist()) | {fitted})
    t = st.select_slider("Confidence cut-off", options, value=fitted,
                         format_func=lambda v: f"{v:.4f}" if v == fitted else f"{v:.2f}")

    y, p = data.split_arrays(FINAL, "test")
    now = metrics.abstention(y, p, t)
    a, b, c, d = st.columns(4)
    a.metric("Share graded", f"{now['coverage']:.1%}")
    b.metric("Accuracy on graded", f"{now['accuracy_kept']:.1%}",
             delta=f"{100 * (now['accuracy_kept'] - now['accuracy_all']):+.1f} points vs all")
    c.metric("Withheld", f"{now['withheld']} of {len(y):,}")
    d.metric("Referable among withheld", f"{now['referable_withheld_pct']:.1f}%"
             if now["withheld"] else "-")
    st.caption(f"DDR test set ({len(y):,} images), which played no part in choosing the "
               "cut-off.")

    rows = {}
    for split, name in SPLIT_NAMES.items():
        arrays = data.split_arrays(FINAL, split)
        if arrays is not None:
            r = metrics.abstention(*arrays, t)
            rows[name] = {"Share graded": f"{r['coverage']:.1%}",
                          "Withheld": f"{r['withheld']} of {len(arrays[0]):,}",
                          "Accuracy, all images": f"{r['accuracy_all']:.1%}",
                          "Accuracy, graded only": f"{r['accuracy_kept']:.1%}",
                          "QWK, graded only": f"{r['qwk_kept']:.4f}"}
    st.dataframe(pd.DataFrame(rows).T, width="stretch")

    curves = {SPLIT_NAMES[s]: df for s, df in tables.items() if df is not None}
    st.altair_chart(charts.abstention_chart(curves, t, fitted, ABSTENTION_TARGET),
                    width="stretch")
    st.caption(f"Each line runs from a strict cut-off (few images graded, high accuracy) to no "
               f"cut-off (all graded). The dashed line is the {ABSTENTION_TARGET:.0%} target.")
    external = data.split_arrays(FINAL, "idrid")
    if external is not None:
        dt, ex = metrics.abstention(y, p, fitted), metrics.abstention(*external, fitted)
        st.markdown(
            f"**The cut-off does not carry over to other cameras.** At {fitted:.4f}, DDR test "
            f"withholds {1 - dt['coverage']:.1%} of images and the graded ones are "
            f"{dt['accuracy_kept']:.1%} correct, close to validation. IDRiD, from a different "
            f"clinic and camera, withholds more ({1 - ex['coverage']:.1%}), yet its graded images "
            f"are only {ex['accuracy_kept']:.1%} correct. The model is confidently wrong more "
            f"often on unfamiliar images, so a confidence cut-off is only as reliable as the "
            f"images it was fitted on.")

ui.page_header(
    "Results and decisions",
    "Each design choice in the order it was made, with the evidence that decided it, then "
    "how the final model performs.",
)
tabs = st.tabs(["Design decisions", "Final model", "Referral threshold", "Abstention"])
with tabs[0]:
    design_tab()
with tabs[1]:
    final_tab()
with tabs[2]:
    referral_tab()
with tabs[3]:
    abstention_tab()
