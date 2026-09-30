import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src import charts, cnn_diagram, data, diagrams, engines, features, registry, state, ui
from src import config as C
from src.augment import AugSettings, augment, augment_steps
from src.explanation import pct
from src.preprocess import draw_fov, preprocess_steps

MODEL = C.SCREENING_MODEL_ID
SAMPLE_TYPES = (".jpg", ".jpeg", ".png")


def model_config() -> dict:
    return json.loads((registry.model_dir(MODEL) / "config.json").read_text())


def source_image():
    """The shared upload if there is one, otherwise the built-in sample."""
    img = state.get_image()
    if img is not None:
        return img, False
    samples = sorted(p for p in C.SAMPLES_DIR.glob("*") if p.suffix.lower() in SAMPLE_TYPES)
    if not samples:
        return None, True
    return state.decode(samples[0].read_bytes(), samples[0].name), True


@st.cache_data(max_entries=4, show_spinner=False)
def steps_for(raw: bytes, size: int):
    return preprocess_steps(state.decode(raw, "image").image_bgr, size)


@st.cache_data(max_entries=4, show_spinner="Running the model...")
def model_view(raw: bytes):
    engine = engines.screening_engine()
    result = engine.run(state.decode(raw, "image").image_bgr)
    maps = features.feature_maps(engine, result.processed_rgb.astype(np.float32)[None])
    head = features.head_activations(engine, maps[-1]["raw"])
    for m in maps:
        del m["raw"]
    return result, maps, head


@st.cache_data(show_spinner=False)
def trainable_counts():
    return features.trainable_counts(engines.screening_engine())


@st.cache_data
def train_counts() -> pd.Series:
    split = pd.read_csv(C.SPLIT_FILES["v2"])
    return split.loc[split["split"] == "train", "diagnosis"].value_counts().sort_index()


def rgb(bgr: np.ndarray) -> np.ndarray:
    return bgr[..., ::-1]


def preprocessing_tab(img, size: int) -> None:
    steps = steps_for(img.raw, size)
    if steps is None:
        st.error("No retina could be found in this image, so there are no steps to show.")
        return
    fov, h, w = steps["fov"], *img.image_bgr.shape[:2]
    panels = [
        ("1. Original", rgb(img.image_bgr), f"{w} x {h} px, as uploaded."),
        ("2. Retina found", rgb(draw_fov(img.image_bgr, fov)),
         f"The brightest colour channel separates the retina from the dark background. Centre "
         f"({fov.cx:.0f}, {fov.cy:.0f}), radius {fov.r:.0f} px. Green is the detected edge, blue "
         f"the circle kept: 95% of the radius, so the camera's marker notch at the rim is cut."),
        ("3. Square crop", rgb(steps["crop"]),
         f"A {steps['side']} x {steps['side']} px square around the kept circle. Any part "
         f"outside the photo is filled with black."),
        ("4. Circular mask", rgb(steps["masked"]),
         "Everything outside the circle is set to black, so every image has the same clean edge "
         "whatever camera took it."),
        ("5. Model input", rgb(steps["final"]),
         f"One resize to {size} x {size} px ({steps['interp']}). Resizing once keeps more fine "
         f"detail than resizing in several steps. This is exactly what the model reads."),
    ]
    cols = st.columns(len(panels))
    for col, (title, image, _) in zip(cols, panels):
        with col:
            st.image(image, caption=title, width="stretch")
    choice = st.segmented_control("Step", [p[0] for p in panels], default=panels[-1][0],
                                  label_visibility="collapsed")
    title, image, text = next(p for p in panels if p[0] == (choice or panels[-1][0]))
    left, right = st.columns([3, 2])
    with left:
        st.image(image, width="stretch")
    with right:
        st.markdown(f"#### {title}")
        st.markdown(text)


def augmentation_tab(img, size: int, cfg: dict) -> None:
    steps = steps_for(img.raw, size)
    if steps is None:
        st.error("No retina could be found in this image.")
        return
    s = AugSettings.from_config(cfg)
    st.markdown(
        f"During training every image was changed at random each time it was used, so the model "
        f"never saw exactly the same picture twice: rotation up to {s.rotation:g} degrees, zoom "
        f"{s.zoom[0]:g} to {s.zoom[1]:g}, horizontal and vertical flips, brightness within "
        f"{s.brightness:.0%}, contrast within {s.contrast:.0%} and each colour within "
        f"{s.channel:.0%}. A retina has no fixed 'up', and cameras differ in exposure and "
        f"colour, so these changes teach the model what should not matter.")
    seed = st.session_state.setdefault("aug_seed", 0)
    if st.button("Draw again", icon=":material/refresh:"):
        st.session_state["aug_seed"] = seed = seed + 1

    st.markdown("**One training image, step by step**")
    aug = augment_steps(steps["final"], np.random.default_rng(seed), s)
    for col, step in zip(st.columns(len(aug)), aug):
        with col:
            st.image(rgb(step["image"]), caption=step["title"], width="stretch")
    pick = st.segmented_control("Augmentation step", [a["title"] for a in aug],
                                default=aug[-1]["title"], label_visibility="collapsed",
                                key="aug_step")
    chosen = next(a for a in aug if a["title"] == (pick or aug[-1]["title"]))
    left, right = st.columns([3, 2])
    with left:
        st.image(rgb(chosen["image"]), width="stretch")
    with right:
        st.markdown(f"#### {chosen['title']}")
        st.markdown(chosen["text"])

    st.markdown("**Eight random draws**")
    st.caption("The first one is the image built step by step above.")
    rng = np.random.default_rng(seed)
    draws = [augment(steps["final"], rng, s) for _ in range(8)]
    for row in (draws[:4], draws[4:]):
        for col, (image, d) in zip(st.columns(4), row):
            flips = " + ".join(f for f, on in (("H flip", d["hflip"]), ("V flip", d["vflip"])) if on)
            with col:
                st.image(rgb(image), width="stretch")
                st.caption(f"{d['angle']:.0f} deg, zoom {d['zoom']:.2f}"
                           + (f", {flips}" if flips else "")
                           + f", brightness {d['brightness']:+.0%}")
    st.caption("Training only. Screening always uses the unchanged model input.")

    st.divider()
    st.markdown("#### Class imbalance")
    if not C.SPLIT_FILES["v2"].exists():
        st.error("The training split file could not be found, so class counts cannot be shown.")
        return
    counts = train_counts()
    total = int(counts.sum())
    df = pd.DataFrame({"Grade": [C.LABELS[i] for i in counts.index], "Images": counts.values})
    df["Share"] = df["Images"] / total
    bars = alt.Chart(df).mark_bar(cornerRadiusEnd=3, color=charts.ACCENT).encode(
        x=alt.X("Images:Q"), y=alt.Y("Grade:N", sort=list(C.LABELS), title=None),
        tooltip=["Grade", "Images", alt.Tooltip("Share:Q", format=".1%")])
    labels = bars.mark_text(align="left", dx=4, color="#E3EAEE").encode(text="Images:Q")
    most, least = df.loc[df["Images"].idxmax()], df.loc[df["Images"].idxmin()]
    left, right = st.columns([3, 2])
    with left:
        st.altair_chart((bars + labels).properties(height=220), width="stretch")
    with right:
        st.markdown(
            f"The {total:,} training images are far from balanced: {most['Grade']} has "
            f"{most['Images']:,} ({most['Share']:.1%}) and {least['Grade']} only "
            f"{least['Images']:,} ({least['Share']:.1%}), a ratio of {most['Images']:,} / "
            f"{least['Images']:,} = {most['Images'] / least['Images']:.0f} to 1. A model can "
            f"score well by mostly ignoring the rare grades.")

    st.markdown("#### How focal loss handles it")
    gamma = float(cfg.get("FOCAL_GAMMA", 2.0))
    p = st.slider("Model's confidence in the correct grade (p)", 0.05, 0.99, 0.90, 0.01)
    ce = -np.log(p)
    weight = (1 - p) ** gamma
    a, b, c = st.columns(3)
    a.metric("Ordinary loss, -ln p", f"{ce:.3f}")
    b.metric(f"Focal weight, (1 - p)^{gamma:g}", f"{weight:.4f}")
    c.metric("Focal loss", f"{weight * ce:.4f}")
    st.markdown(
        f"At p = {p:.2f} the ordinary loss is -ln {p:.2f} = {ce:.3f}. Focal loss multiplies it by "
        f"(1 - {p:.2f})^{gamma:g} = {weight:.4f}, giving {weight * ce:.4f}. Easy, confident cases "
        f"are turned down almost to nothing, so training effort goes to the hard ones, and rare "
        f"grades are mostly hard. It weights hard examples, not rare ones directly.")


def architecture_tab(img, cfg: dict) -> None:
    size, head_type, unfreeze = int(cfg["IMG_SIZE"]), cfg.get("HEAD", "dense"), float(cfg["UNFREEZE"])
    st.markdown(
        "EfficientNetV2-B0, pretrained on ImageNet, with new output layers trained on DDR. Each "
        "step deeper halves the size of the feature maps and adds more of them, trading position "
        f"detail for richer patterns: {size} px in, {size // 32} x {size // 32} maps at the end. "
        "Pooling then turns those maps into one list of numbers for the new layers to grade.")
    svg, width = cnn_diagram.architecture_svg(size, head_type, unfreeze)
    st.html(cnn_diagram.svg_img(svg, width))

    st.markdown("#### The same network on this image")
    try:
        result, maps, head = model_view(img.raw)
    except Exception as err:
        st.error(f"The model could not be run on this image: {err}")
        return
    svg, width = cnn_diagram.architecture_svg(size, head_type, unfreeze, maps, head,
                                              result.processed_rgb)
    st.html(cnn_diagram.svg_img(svg, width))
    st.caption("Early layers respond to edges and colour changes; later ones combine these into "
               "larger patterns. What a single map means is not labelled, because it cannot be "
               "read off reliably.")

    grade = C.LABELS[result.decision.grade]
    st.markdown("#### From the last layer to the heatmap")
    st.html(cnn_diagram.gradcam_strip(result.processed_rgb, maps[-1]["front"],
                                      result.overlay_rgb, maps[-1]["shape"], grade))

    st.markdown("#### Look inside a layer")
    choice = st.selectbox("Layer", [m["label"] for m in maps], index=len(maps) - 1,
                          label_visibility="collapsed")
    m = next(m for m in maps if m["label"] == choice)
    h, w, ch = m["shape"]
    st.caption(f"The 8 strongest of {ch:,} feature maps at {h} x {w}, from layer {m['layer']}. "
               "Each is scaled on its own, so brightness compares positions within one map, "
               "not maps with each other.")
    st.html(f'<img src="{cnn_diagram.data_uri(m["tiles"])}" style="width:100%;'
            f'border-radius:4px;image-rendering:pixelated">')


def training_tab(cfg: dict) -> None:
    res = data.run_results(MODEL)
    try:
        counts = trainable_counts()
    except Exception as err:
        st.error(f"The model could not be loaded: {err}")
        return
    st.html(diagrams.phases(
        int(cfg["PHASE1_EPOCHS"]), float(cfg["PHASE1_LR"]), int(res["epochs_run"]),
        int(res["best_epoch"]), float(cfg["PHASE2_LR"]), counts, float(cfg["RLR_FACTOR"]),
        int(cfg["RLR_PATIENCE"]), int(cfg["ES_PATIENCE"])))

    hist = data.history(MODEL)
    switch, best = int(cfg["PHASE1_EPOCHS"]) + 0.5, int(res["best_epoch"])
    acc = {c: n for c, n in [("accuracy", "Training, augmented"),
                             ("clean_accuracy", "Training, unchanged images"),
                             ("val_accuracy", "Validation")] if c in hist}
    left, right = st.columns(2)
    with left:
        st.markdown("**Accuracy**")
        st.altair_chart(charts.training_curves(hist, acc, switch, best, "Accuracy"),
                        width="stretch")
    with right:
        st.markdown("**Validation QWK**")
        if "val_qwk" in hist:
            st.altair_chart(charts.training_curves(hist, {"val_qwk": "Validation QWK"},
                                                   switch, best, "QWK"), width="stretch")
    if "learning_rate" in hist:
        lr = alt.Chart(hist).mark_line(point=True, interpolate="step-after").encode(
            x=alt.X("epoch:Q", title="Epoch"),
            y=alt.Y("learning_rate:Q", title="Learning rate", scale=alt.Scale(type="log")),
            tooltip=["epoch", alt.Tooltip("learning_rate:Q", format=".1e")])
        st.markdown("**Learning rate**")
        st.altair_chart(lr.properties(height=180), width="stretch")
    st.caption("Training accuracy on augmented images sits below the same images unchanged, "
               "because augmentation makes them harder. The gap between unchanged training "
               "images and validation shows how much the model fits the training set.")


ui.page_header(
    "How it was built",
    f"The final model ({C.MODELS[MODEL].title}) from photograph to grade, shown on a real image.",
)

if registry.check_model(MODEL)["status"] != "ok":
    st.error("The final model's files could not be found, so this page cannot be shown.")
else:
    img, is_sample = source_image()
    cfg = model_config()
    size = int(cfg["IMG_SIZE"])
    if img is None:
        st.error("No image to show. Upload one on the Screening page.")
    else:
        st.caption(f"Showing the built-in sample image ({img.name}). {C.PHOTO_CREDIT} Upload "
                   f"your own on the Screening page to see it here." if is_sample
                   else f"Showing {img.name}.")
        tabs = st.tabs(["Preprocessing", "Augmentation and imbalance", "Architecture",
                        "Training"])
        with tabs[0]:
            preprocessing_tab(img, size)
        with tabs[1]:
            augmentation_tab(img, size, cfg)
        with tabs[2]:
            architecture_tab(img, cfg)
        with tabs[3]:
            training_tab(cfg)
