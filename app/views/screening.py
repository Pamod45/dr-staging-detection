"""Screening page."""
from datetime import datetime

import streamlit as st

from src import config as C
from src import registry, state, ui
from src.charts import probability_chart
from src.explanation import explain, pct, threshold_pct

ui.page_header(
    "Screening",
    "Upload one retinal photograph. The model grades it on the five-step ICDR scale "
    "and says whether it should be referred to an eye specialist.",
)


@st.cache_resource(show_spinner="Loading the grading model...")
def get_engine():
    from src.inference import ScreeningEngine
    return ScreeningEngine()


def render():
    """Whole page as a function, so an early exit is a plain return. st.stop() would also
    stop app/main.py before it draws the footer."""
    upload = st.file_uploader("Retinal photograph (JPG or PNG)", type=["jpg", "jpeg", "png"],
                              key=state.uploader_key())
    if upload is not None:
        try:
            new = state.decode(upload.getvalue(), upload.name)
        except ValueError as err:
            st.error(str(err))
        else:
            if state.set_image(new):
                st.rerun()

    img = state.get_image()
    if img is None:
        return

    if registry.check_model(C.SCREENING_MODEL_ID)["status"] != "ok":
        st.error("The grading model could not be found, so images cannot be graded right now. "
                 "Run scripts/check_models.py for details.")
        return

    result = state.cache_get("screening")
    if result is None:
        from src.inference import NoRetinaFound
        try:
            with st.spinner("Grading the image..."):
                result = get_engine().run(img.image_bgr)
        except NoRetinaFound:
            st.error("No retina could be found in this image. Upload a colour fundus photograph "
                     "with the round retinal area visible on a dark background.")
            return
        except Exception as err:
            st.error(f"The image could not be graded: {err}")
            return
        state.cache_set("screening", result)

    d, t = result.decision, result.thresholds

    left, right = st.columns(2)
    with left, st.container(border=True):
        st.caption("Grade")
        if d.abstained:
            st.markdown("### :orange[Not graded]")
            st.markdown(f"Confidence {pct(d.confidence)} is below the "
                        f"{threshold_pct(t.abstention)} needed to report a grade. "
                        "This image should be graded by a person.")
        else:
            st.markdown(f"### {C.LABELS[d.grade]}")
            st.markdown(f"Grade {d.grade} of 4, confidence {pct(d.confidence)}")
    with right, st.container(border=True):
        st.caption("Referral")
        if d.refer:
            st.markdown("### :orange[Refer to eye specialist]")
        else:
            st.markdown("### :green[No referral needed]")
        st.markdown(f"Referable probability {pct(d.referable_prob)}, "
                    f"threshold {threshold_pct(t.referral)}")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(img.image_rgb, caption="Original", width="stretch")
    with c2:
        st.image(result.processed_rgb, caption=f"Model input, {result.input_size} px",
                 width="stretch")
    with c3:
        st.image(result.overlay_rgb, caption="Where the model looked (Grad-CAM)", width="stretch")

    chart_col, text_col = st.columns([2, 3])
    with chart_col:
        st.subheader("Probability per grade")
        st.altair_chart(probability_chart(result.probs, None if d.abstained else d.grade),
                        width="stretch")
    with text_col:
        st.subheader("Explanation")
        for para in explain(result):
            st.markdown(para)

    report = state.cache_get("report")
    if report is None:
        from src.report import build_pdf
        when = datetime.now()
        report = (build_pdf(result, img.image_rgb, img.name, img.short_hash, when),
                  f"dr_report_{img.short_hash}_{when:%Y%m%d_%H%M%S}.pdf")
        state.cache_set("report", report)
    st.download_button("Download report (PDF)", data=report[0], file_name=report[1],
                       mime="application/pdf", icon=":material/download:")
    st.caption(f"Graded in {result.seconds:.1f} s")


render()
