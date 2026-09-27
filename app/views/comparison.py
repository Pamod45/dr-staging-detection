"""Model comparison: models that differ in one setting, side by side."""
import numpy as np
import pandas as pd
import streamlit as st

from src import charts, data, engines, metrics, model_info, registry, state, ui
from src import config as C
from src.explanation import pct

DIFF_BG = "background-color: rgba(79, 163, 199, 0.22)"
BEST_CELL = "background-color: rgba(79, 163, 199, 0.35); font-weight: 700"
SAME_FG = "color: #7A8791"


@st.cache_data
def model_summary(model_id: str) -> dict:
    return metrics.summary(data.labels(model_id), data.probs(model_id))


@st.cache_data
def model_confusion(model_id: str) -> np.ndarray:
    return metrics.confusion(data.labels(model_id), data.probs(model_id).argmax(1))


def highlight_differences(df: pd.DataFrame):
    def style(row):
        same = row.nunique() == 1
        return [SAME_FG if same else DIFF_BG] * len(row)
    return df.style.apply(style, axis=1)


def highlight_best(df: pd.DataFrame):
    def style(row):
        best = row.max()
        return [BEST_CELL if pd.notna(v) and np.isclose(v, best) else "" for v in row]
    return df.style.apply(style, axis=1).format("{:.4f}", na_rep="-")


def setup_section(comp, ids) -> None:
    st.subheader("Setup")
    df = pd.DataFrame({C.MODELS[m].title: model_info.settings(m) for m in ids}).loc[model_info.ROWS]
    st.dataframe(highlight_differences(df), width="stretch")
    st.caption("Highlighted rows differ between the models; grey rows are identical. Best epoch "
               f"is the {model_info.best_epoch_rule(C.MODELS[ids[0]].notebook)}.")


def results_section(ids, eval_set: str) -> list[dict]:
    st.subheader("Results")
    summaries = {C.MODELS[m].title: model_summary(m) for m in ids}
    df = pd.DataFrame(summaries)
    if C.MODELS[ids[0]].notebook == "v2":
        for m in ids:
            ext = data.stored_summary(m, "idrid")
            if ext:
                df.loc["IDRiD QWK (external)", C.MODELS[m].title] = ext["QWK"]
    st.dataframe(highlight_best(df), width="stretch")
    st.caption(f"Computed from each model's stored predictions on the {C.EVAL_SET_TEXT[eval_set]}. "
               f"The best value in each row is highlighted; a QWK lead under "
               f"{C.NOISE_FLOOR_QWK} is within run-to-run noise.")
    return [{"Model": t, **s} for t, s in summaries.items()]


def charts_section(ids, rows) -> None:
    st.subheader("Charts")
    left, right = st.columns([2, 3])
    with left:
        st.altair_chart(charts.metric_bars(rows), width="stretch")
    with right:
        shared = model_info.shared_curve_metric(ids)
        if shared is None:
            st.caption("These runs have no per-epoch metric in common, so no curves are drawn.")
        else:
            col, label = shared
            curves = None
            for m in ids:
                c = model_info.curve(m, col)
                curves = c if curves is None else curves.merge(c, on="epoch", how="outer")
            switch = C.V2_PHASE1_EPOCHS + 0.5 if C.MODELS[ids[0]].notebook == "v2" else None
            st.altair_chart(charts.training_curves(curves, {m: C.MODELS[m].title for m in ids},
                                                   phase_switch=switch, y_title=label),
                            width="stretch")
            if col != "val_qwk":
                st.caption("Not every run logged QWK per epoch, so validation accuracy is shown.")
    cols = st.columns(len(ids))
    for col, m in zip(cols, ids):
        with col:
            st.altair_chart(charts.confusion_chart(model_confusion(m), C.MODELS[m].title),
                            width="stretch")


def cost_section(ids) -> None:
    st.subheader("Training cost")
    first = ids[0]
    base_qwk, base_min = model_summary(first)["QWK"], model_info.minutes(first)
    rows = []
    for m in ids:
        q, mins = model_summary(m)["QWK"], model_info.minutes(m)
        rows.append({
            "Model": C.MODELS[m].title,
            "QWK": f"{q:.4f}",
            f"QWK change vs {C.MODELS[first].title}": "-" if m == first else f"{q - base_qwk:+.4f}",
            "Minutes": f"{mins:.1f}" if mins is not None else "Not recorded",
            "Minutes change": "-" if m == first or mins is None or base_min is None
            else f"{mins - base_min:+.1f}",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"A QWK change smaller than {C.NOISE_FLOOR_QWK} is within run-to-run noise, so it "
               "does not justify extra training time on its own.")


def upload_section(comp, ids) -> None:
    st.subheader("Your image through each model")
    if not comp.live_upload:
        st.info(comp.note)
        return
    if not ui.needs_image("compare these models on it"):
        return
    img = state.get_image()
    key = f"compare:{comp.id}"
    outputs = state.cache_get(key)
    if outputs is None:
        if not st.button(f"Run on {img.name}", icon=":material/play_arrow:"):
            st.caption(f"Runs {len(ids)} models on the uploaded image. The first run loads each "
                       "model, which takes a little longer.")
            return
        from src.inference import NoRetinaFound
        outputs = {}
        try:
            for m in ids:
                if engines.low_memory():
                    engines.release_comparison_models()
                with st.spinner(f"Running {C.MODELS[m].title}..."):
                    outputs[m] = engines.comparison_runner(m).run(img.image_bgr)
        except NoRetinaFound:
            st.error("No retina could be found in this image.")
            return
        except Exception as err:
            st.error(f"The comparison could not be run: {err}")
            return
        state.cache_set(key, outputs)

    cols = st.columns(len(ids))
    for col, m in zip(cols, ids):
        out = outputs[m]
        grade = int(out.probs.argmax())
        with col, st.container(border=True):
            st.markdown(f"**{C.MODELS[m].title}**")
            st.markdown(f"### {C.LABELS[grade]}")
            st.caption(f"Confidence {pct(out.probs.max())}")
            st.image(out.overlay_rgb, caption=f"Grad-CAM on the {out.input_rgb.shape[0]} px input",
                     width="stretch")
            st.altair_chart(charts.probability_chart(out.probs, grade), width="stretch")
    st.caption("Only the grade is shown here. Referral and abstention thresholds belong to the "
               "screening model, so they are not applied to these models.")


ui.page_header(
    "Model comparison",
    "Models inside one comparison differ in a single setting, so any difference in "
    "results comes from that setting.",
)

comp = st.selectbox("Comparison", C.COMPARISONS, format_func=lambda c: c.title)
ids = list(comp.members)
eval_set = C.MODELS[ids[0]].eval_set
st.markdown(f"**What changed:** {comp.changed}")
st.caption(f"Measured on the {C.EVAL_SET_TEXT[eval_set]}. Differences under "
           f"{C.NOISE_FLOOR_QWK} QWK are within run-to-run noise.")

missing = [m for m in ids if registry.check_model(m)["status"] != "ok"]
if missing:
    st.error("Files for " + ", ".join(C.MODELS[m].title for m in missing) + " could not be "
             "found, so this comparison cannot be shown.")
else:
    setup_section(comp, ids)
    rows = results_section(ids, eval_set)
    charts_section(ids, rows)
    cost_section(ids)
    upload_section(comp, ids)
