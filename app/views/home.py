"""About diabetic retinopathy: background for a non-specialist, from content/dr_facts.md.
No model output and no performance numbers on this page."""
import altair as alt
import pandas as pd
import streamlit as st

from src import charts, content, data, lesions, ui
from src import config as C
from src.cnn_diagram import BG, MUTED, data_uri, svg_img

GRADE_DIR = C.SAMPLES_DIR / "grades"
REFER_COLOUR, RESCREEN_COLOUR = "#C8872B", "#3E7C59"
SPLIT_NAMES = {"train": "Training", "val": "Validation", "test": "Test"}


def grade_photo(grade: int):
    for ext in (".jpg", ".jpeg", ".png"):
        p = GRADE_DIR / f"grade_{grade}{ext}"
        if p.exists():
            return str(p)
    return None


def lesion_files() -> tuple:
    """Folder contents as a cache key, so adding or replacing files is picked up."""
    if not lesions.LESION_DIR.exists():
        return ()
    return tuple((str(p.relative_to(lesions.LESION_DIR)), p.stat().st_mtime)
                 for p in sorted(lesions.LESION_DIR.rglob("*")) if p.is_file())


@st.cache_resource(show_spinner=False)
def _sets(files: tuple, folder: str) -> dict:
    return lesions.by_grade()


def annotated_sets() -> dict:
    return _sets(lesion_files(), str(lesions.LESION_DIR))


@st.cache_data(show_spinner=False)
def callout(grade: int, show: tuple, files: tuple):
    return lesions.callout_svg(annotated_sets()[grade], show)


@st.cache_data(show_spinner=False)
def close_up(grade: int, code: str, files: tuple) -> str:
    return data_uri(lesions.crop(annotated_sets()[grade], code))


def sign_cards(grade: int, sets: dict) -> tuple[str, str | None]:
    """HTML cards for the signs that define a grade, and a note on where the close-ups came
    from when they are borrowed from another grade's image."""
    source = grade if grade in sets else lesions.BORROW_FROM.get(grade)
    note = None
    if source is not None and source != grade and source in sets:
        note = (f"Close-up taken from the annotated {C.LABELS[source]} photograph "
                f"({sets[source].name}); these signs look the same at any grade.")
    cards = []
    for code in lesions.GRADE_SIGNS[grade]:
        sign = lesions.SIGNS[code]
        picture = ""
        if source in sets:
            picture = (f'<img src="{close_up(source, code, lesion_files())}" style="width:100%;'
                       f'border-radius:6px;border:2px solid {sign.colour}">')
        cards.append(f'<div style="flex:1 1 170px;max-width:220px">{picture}'
                     f'<div style="font-weight:700;margin-top:6px;color:{sign.colour}">'
                     f'{sign.name}</div><div style="font-size:13px;color:{MUTED}">'
                     f'{sign.looks_like}</div></div>')
    for name, text in lesions.UNMARKED.get(grade, []):
        cards.append(f'<div style="flex:1 1 170px;max-width:220px;background:{BG};'
                     f'border-radius:6px;padding:10px"><div style="font-weight:700">{name}'
                     f'</div><div style="font-size:13px;color:{MUTED}">{text}</div></div>')
    return f'<div style="display:flex;gap:12px;flex-wrap:wrap">{"".join(cards)}</div>', note


def annotated_figure(grade: int, a) -> None:
    names = {lesions.SIGNS[c].name: c for c in ("MA", "HE", "EX", "SE", "OD") if c in a.masks}
    picked = st.pills("Signs to show", list(names), selection_mode="multi",
                      default=list(names), label_visibility="collapsed", key=f"signs_{grade}")
    svg, width = callout(grade, tuple(names[n] for n in (picked or [])), lesion_files())
    st.html(svg_img(svg, width))
    st.caption(f"{a.name}, graded {C.LABELS[grade]} in the official IDRiD labels. Every "
               f"outline was drawn by eye specialists. {lesions.CREDIT} Microaneurysms are "
               "only a few pixels wide, so they are ringed rather than outlined.")


def grades_section(section: str) -> None:
    intro, _ = content.split_paragraphs(section)
    defs = content.grade_definitions(section)
    sets = annotated_sets()
    st.header("The five ICDR grades")
    st.markdown(intro)

    cols = st.columns(C.N_CLASSES)
    for g, col in enumerate(cols):
        with col:
            photo = grade_photo(g)
            if photo:
                st.image(photo, width="stretch")
            st.markdown(f"**Grade {g}**  \n{defs.get(g, (C.LABELS[g], ''))[0]}")
    st.caption(C.PHOTO_CREDIT)

    st.subheader("Grade by grade")
    tabs = st.tabs([f"{g}. {defs.get(g, (C.LABELS[g], ''))[0]}" for g in range(C.N_CLASSES)])
    for g, tab in enumerate(tabs):
        with tab:
            name, text = defs.get(g, (C.LABELS[g], ""))
            st.markdown(text[:1].upper() + text[1:])
            if g in sets:
                annotated_figure(g, sets[g])
            else:
                photo = grade_photo(g)
                if photo:
                    st.image(photo, caption=f"A {name} photograph", width=360)
            if g == 0:
                st.caption("Vessels, optic disc and macula look normal.")
                continue
            st.markdown("**What to look for**")
            cards, note = sign_cards(g, sets)
            st.html(cards)
            if note:
                st.caption(note)
            if g in lesions.UNMARKED:
                st.caption("No openly licensed photograph with expert outlines of these signs "
                           "was available, so they are described in words.")
    extra = content.after_bullets(section)
    if extra:
        st.caption(extra)


def referral_section(section: str) -> None:
    st.header("Referable or not")
    paragraphs = [p for p in section.split("\n\n") if p.strip()]
    st.markdown(paragraphs[0])
    cells = "".join(
        f'<div style="flex:1;padding:10px 6px;border-radius:6px;text-align:center;'
        f'background:{REFER_COLOUR if g >= 2 else RESCREEN_COLOUR};color:#fff;font-size:14px">'
        f'<b>Grade {g}</b><br>{C.LABELS[g]}<br><span style="font-size:12px">'
        f'{"Refer to specialist" if g >= 2 else "Re-screen routinely"}</span></div>'
        for g in range(C.N_CLASSES))
    st.html(f'<div style="display:flex;gap:6px;margin:6px 0 10px">{cells}</div>')
    for p in paragraphs[1:]:
        st.markdown(p)


def datasets_section(section: str) -> None:
    st.header("The datasets")
    st.markdown(section)
    left, right = st.columns([3, 2])
    with left:
        if C.SPLIT_FILES["v2"].exists():
            df = data.ddr_grade_counts()
            df["Grade"] = df["diagnosis"].map(lambda g: C.LABELS[g])
            df["Split"] = df["split"].map(SPLIT_NAMES)
            total = int(df["images"].sum())
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X("images:Q", title="Images"),
                y=alt.Y("Grade:N", sort=list(C.LABELS), title=None),
                color=alt.Color("Split:N", sort=list(SPLIT_NAMES.values()),
                                legend=alt.Legend(orient="bottom", title=None)),
                order=alt.Order("split:N"),
                tooltip=["Grade", "Split", "images"],
            ).properties(height=230, title=f"DDR, {total:,} images")
            st.altair_chart(chart, width="stretch")
    with right:
        counts = data.idrid_counts()
        idr = pd.DataFrame({"Grade": [C.LABELS[g] for g in counts],
                            "images": list(counts.values())})
        bars = alt.Chart(idr).mark_bar(color=charts.ACCENT).encode(
            x=alt.X("images:Q", title="Images"),
            y=alt.Y("Grade:N", sort=list(C.LABELS), title=None),
            tooltip=["Grade", "images"])
        labels = bars.mark_text(align="left", dx=4, color="#E3EAEE").encode(text="images:Q")
        st.altair_chart((bars + labels).properties(
            height=230, title=f"IDRiD, {sum(counts.values())} images, external test"),
            width="stretch")
    st.caption("The grades are far from balanced in both datasets, which is why the model was "
               "trained with focal loss. How it was built shows the details.")


ui.page_header(
    "About diabetic retinopathy",
    "What the disease is, how it is graded, and what this tool does with a photograph.",
)

if not content.FACTS.exists():
    st.error("The page text could not be found (content/dr_facts.md).")
else:
    facts = content.sections()
    st.markdown(facts.get("What diabetic retinopathy is", ""))
    if "The five ICDR grades" in facts:
        grades_section(facts["The five ICDR grades"])
    if "Referable or not" in facts:
        referral_section(facts["Referable or not"])
    if "The datasets" in facts:
        datasets_section(facts["The datasets"])
    if "What this tool does not do" in facts:
        st.header("What this tool does not do")
        st.markdown(facts["What this tool does not do"])

st.subheader("Where to go next")
st.page_link("views/screening.py", label="Screen an image", icon=":material/upload:")
st.page_link("views/how_built.py", label="See how it was built", icon=":material/account_tree:")
st.page_link("views/results.py", label="See the evidence behind each decision",
             icon=":material/insights:")
