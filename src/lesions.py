"""Signs of DR on real photographs, outlined by eye specialists: images from the IDRiD
segmentation set with their expert masks, one per grade, in content/lesions/grade_N/ as
IMAGE.jpg plus IMAGE_MA.tif, _HE.tif, _EX.tif, _SE.tif and optionally _OD.tif. Each image's
grade comes from the official IDRiD grading labels, matched by scripts/pick_lesion_examples.py.

Nothing here is model output. The masks are the dataset's own annotations; this module only
draws them. IDRiD, Porwal et al. 2018, licensed CC BY 4.0."""
from dataclasses import dataclass
from html import escape

import cv2
import numpy as np
from PIL import Image

from src import config as C
from src.cnn_diagram import BG, FONT, MUTED, TEXT, data_uri

LESION_DIR = C.CONTENT_DIR / "lesions"
CREDIT = ("Expert annotations from the IDRiD segmentation set (Porwal et al., 2018), "
          "CC BY 4.0.")
FIG_W = 760


@dataclass(frozen=True)
class Sign:
    code: str
    name: str
    colour: str        # hex, for SVG
    first_grade: int | None
    looks_like: str


SIGNS = {
    "MA": Sign("MA", "Microaneurysms", "#00B4FF", 1,
               "tiny red dots: bulges in the walls of the smallest vessels"),
    "HE": Sign("HE", "Haemorrhages", "#FF3B3B", 2, "dark red spots and blots: small bleeds"),
    "EX": Sign("EX", "Hard exudates", "#27D17F", 2,
               "yellow fatty deposits leaked from vessels"),
    "SE": Sign("SE", "Cotton-wool spots", "#FFC400", 2,
               "pale fluffy patches where small vessels have closed"),
    "OD": Sign("OD", "Optic disc", "#C9D2D8", None,
               "a normal structure, where vessels and nerve fibres leave the eye"),
}
LESIONS = ("MA", "HE", "EX", "SE")

# Signs the ICDR definitions name for each grade, with no open expert masks: text only.
UNMARKED = {
    3: [("Venous beading", "veins that look like a string of beads, in two or more quadrants"),
        ("IRMA", "abnormal branching vessels inside the retina")],
    4: [("New vessels", "fragile new vessels growing on the retina or optic disc"),
        ("Pre-retinal or vitreous bleed", "blood in front of the retina or in the eye's gel")],
}
GRADE_SIGNS = {0: [], 1: ["MA"], 2: ["HE", "EX", "SE"], 3: ["HE"], 4: []}
# Where a grade has no annotated image of its own (Mild: the segmentation set holds none),
# close-ups are borrowed from this grade, and the caption says so.
BORROW_FROM = {1: 2}


def _hex_bgr(h: str) -> tuple:
    return int(h[5:7], 16), int(h[3:5], 16), int(h[1:3], 16)


@dataclass
class Annotated:
    name: str
    image: np.ndarray            # full resolution RGB
    masks: dict                  # code -> full resolution bool mask

    @property
    def scale(self) -> float:
        return FIG_W / self.image.shape[1]


def find(folder=None) -> Annotated | None:
    """The first image in folder that has all four lesion masks."""
    folder = folder or LESION_DIR
    for img_path in sorted(folder.glob("*.jpg")):
        stem = img_path.stem
        paths = {c: folder / f"{stem}_{c}.tif" for c in SIGNS}
        if all(paths[c].exists() for c in LESIONS):
            image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            masks = {c: np.array(Image.open(p).convert("L")) > 0
                     for c, p in paths.items() if p.exists()}
            return Annotated(stem, image, masks)
    return None


def by_grade() -> dict:
    """{grade: Annotated} for every content/lesions/grade_N/ folder holding a full set."""
    out = {}
    for g in range(C.N_CLASSES):
        a = find(LESION_DIR / f"grade_{g}") if (LESION_DIR / f"grade_{g}").is_dir() else None
        if a is not None:
            out[g] = a
    return out


def components(mask: np.ndarray) -> list[dict]:
    """Separate marked regions, largest first."""
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = [{"area": int(stats[i, cv2.CC_STAT_AREA]), "centre": tuple(cents[i]),
            "box": tuple(int(v) for v in stats[i, :4])} for i in range(1, n)]
    return sorted(out, key=lambda c: -c["area"])


def counts(a: Annotated) -> dict:
    return {c: len(components(a.masks[c])) for c in LESIONS}


def _outline(img_bgr: np.ndarray, mask: np.ndarray, colour_hex: str, thickness: int,
             ring: int = 0) -> None:
    colour = _hex_bgr(colour_hex)
    if ring:
        for comp in components(mask):
            cx, cy = comp["centre"]
            cv2.circle(img_bgr, (int(cx), int(cy)), ring, colour, thickness, cv2.LINE_AA)
        return
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_bgr, cnts, -1, colour, thickness, cv2.LINE_AA)


def overlay(a: Annotated, show: tuple, width: int = FIG_W) -> np.ndarray:
    """Downscaled photo with the chosen signs outlined. Microaneurysms are a few pixels wide,
    so they get rings instead of outlines, or they would vanish at this size."""
    s = width / a.image.shape[1]
    h = int(round(a.image.shape[0] * s))
    img = cv2.cvtColor(cv2.resize(a.image, (width, h), interpolation=cv2.INTER_AREA),
                       cv2.COLOR_RGB2BGR)
    for code in show:
        if code not in a.masks:
            continue
        small = cv2.resize(a.masks[code].astype(np.uint8), (width, h),
                           interpolation=cv2.INTER_NEAREST) > 0
        if code == "MA":
            _outline(img, small, SIGNS[code].colour, 1, ring=6)
        else:
            _outline(img, small, SIGNS[code].colour, 2 if code != "OD" else 1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def example(a: Annotated, code: str) -> tuple[float, float]:
    """A representative instance to point at: the largest region, except for microaneurysms,
    where the largest can be a merged cluster, so a mid-sized one is taken."""
    comps = components(a.masks[code])
    pick = comps[len(comps) // 2] if code == "MA" and len(comps) > 2 else comps[0]
    return pick["centre"]


def crop(a: Annotated, code: str, side: int = 420, out: int = 300) -> np.ndarray:
    """Full-resolution close-up around an example of one sign, with that sign outlined."""
    cx, cy = example(a, code)
    h, w = a.image.shape[:2]
    x0 = int(np.clip(cx - side / 2, 0, w - side))
    y0 = int(np.clip(cy - side / 2, 0, h - side))
    patch = cv2.cvtColor(a.image[y0:y0 + side, x0:x0 + side], cv2.COLOR_RGB2BGR).copy()
    m = a.masks[code][y0:y0 + side, x0:x0 + side]
    if code == "MA":
        _outline(patch, m, SIGNS[code].colour, 2, ring=10)
    else:
        _outline(patch, m, SIGNS[code].colour, 2)
    return cv2.resize(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB), (out, out),
                      interpolation=cv2.INTER_AREA)


def callout_svg(a: Annotated, show: tuple) -> tuple[str, int]:
    """The photo with outlines on the left and one labelled box per sign on the right, each
    joined to an example on the photo by a leader line, like a textbook figure."""
    photo = overlay(a, show)
    ph, pw = photo.shape[:2]
    box_x, box_w, box_h = pw + 90, 340, 58
    # boxes ordered by the height of what they point at, so leader lines do not cross
    codes = sorted((c for c in ("MA", "HE", "EX", "SE", "OD") if c in show and c in a.masks),
                   key=lambda c: example(a, c)[1])
    total = len(codes) * (box_h + 16) - 16
    y = max(10, (ph - total) / 2)
    n = counts(a)
    markers = "".join(
        f'<marker id="arrow{c}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" '
        f'fill="{SIGNS[c].colour}"/></marker>' for c in codes)
    parts = [f'<defs>{markers}</defs>',
             f'<image href="{data_uri(photo)}" x="0" y="0" width="{pw}" height="{ph}"/>']
    for code in codes:
        sign = SIGNS[code]
        ex, ey = (v * a.scale for v in example(a, code))
        by = y + box_h / 2
        line = f'M{box_x} {by:.1f} H{box_x - 30} L{ex:.1f} {ey:.1f}'
        parts.append(f'<path d="{line}" fill="none" stroke="#000" stroke-opacity="0.55" '
                     f'stroke-width="5"/>')
        parts.append(f'<path d="{line}" fill="none" stroke="{sign.colour}" stroke-width="2" '
                     f'marker-end="url(#arrow{code})"/>')
        parts.append(f'<circle cx="{box_x - 30}" cy="{by:.1f}" r="3.5" fill="{sign.colour}"/>')
        parts.append(f'<rect x="{box_x}" y="{y:.1f}" width="{box_w}" height="{box_h}" rx="6" '
                     f'fill="{BG}" stroke="{sign.colour}" stroke-width="2"/>')
        parts.append(f'<text x="{box_x + 12}" y="{y + 24:.1f}" font-size="18" fill="{TEXT}" '
                     f'font-weight="700" style="{FONT}">{escape(sign.name)}</text>')
        if sign.first_grade is not None:
            sub = (f"{n[code]} marked, first appears at grade {sign.first_grade} "
                   f"({C.LABELS[sign.first_grade]})")
        else:
            sub = "Normal structure, for orientation"
        parts.append(f'<text x="{box_x + 12}" y="{y + 45:.1f}" font-size="14" fill="{MUTED}" '
                     f'style="{FONT}">{escape(sub)}</text>')
        y += box_h + 16
    width = box_x + box_w + 6
    height = max(ph, int(y))
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
           f'width="{width}" height="{height}">{"".join(parts)}</svg>')
    return svg, width
