from datetime import datetime

import streamlit as st

from src import config as C
from src import engines, registry, state, ui
from src.charts import probability_chart
from src.explanation import explain, pct, threshold_pct

ui.page_header(
    "Screening",
    "Upload one retinal photograph. The model grades it on the five-step ICDR scale "
    "and says whether it should be referred to an eye specialist.",
)


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
        chat_section(None)
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
                result = engines.screening_engine().run(img.image_bgr)
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
    chat_section(result)


def _history(result) -> list:
    """Per-image history once a photograph is graded; one general thread before that."""
    if result is None:
        return st.session_state.get("chat_general", [])
    return state.cache_get("chat") or []


def _save(result, history: list) -> None:
    if result is None:
        st.session_state["chat_general"] = history
    else:
        state.cache_set("chat", history)


def chat_section(result) -> None:
    from src import chat

    if result is None:
        st.subheader("Ask about diabetic retinopathy")
        st.caption("Upload a photograph above to ask about its result as well.")
    else:
        st.subheader("Ask about this result")
    key = chat.api_key()
    if not key:
        st.info("The question assistant is not available on this server.")
        return

    history = _history(result)
    for m in history:
        with st.chat_message("user" if m["role"] == "user" else "assistant"):
            st.markdown(m["text"])

    used = chat.questions_used(history)
    scope = "this image" if result is not None else "this session"
    left, right = st.columns([4, 1])
    with left:
        st.caption(f"{used} of {chat.MAX_QUESTIONS} questions used for {scope}. Answers are "
                   "generated from the About page text" + (" and this result" if result else "")
                   + ", and are not checked line by line. They are not medical advice.")
    with right:
        if history and st.button("Clear chat", icon=":material/delete:", width="stretch"):
            _save(result, [])
            st.rerun()

    if used >= chat.MAX_QUESTIONS:
        st.info(f"The question limit for {scope} has been reached. Clear the chat to start again.")
        return

    options = chat.SUGGESTED if result is not None else chat.SUGGESTED_GENERAL
    suggested = None
    st.caption("Try one of these, or type your own:")
    cols = st.columns(len(options))
    for i, (col, q) in enumerate(zip(cols, options)):
        if col.button(q, key=f"suggest_{'r' if result else 'g'}_{i}", width="stretch"):
            suggested = q
    placeholder = ("Ask about this result, or about diabetic retinopathy" if result is not None
                   else "Ask about diabetic retinopathy or this tool")
    question = st.chat_input(placeholder) or suggested
    if not question:
        return
    with st.chat_message("user"):
        st.markdown(question)
    try:
        with st.spinner("Thinking..."):
            answer = chat.reply(chat.make_client(key), chat.system_prompt(result), history,
                                question)
    except Exception as err:
        print(f"[chat] {type(err).__name__}: {err}")
        st.error(f"The assistant could not answer: {chat.error_message(err)}")
        return
    _save(result, history + [{"role": "user", "text": question},
                             {"role": "model", "text": answer}])
    st.rerun()


render()
