"""CNN diagrams in the usual textbook style - input, stacks of feature maps shrinking with
depth, a pooled vector, fully connected nodes, output probabilities.

Each diagram is one SVG drawn in viewBox units and shown as an <img> at 100% width, so it
scales to any screen instead of being cut off. The static version is plain shapes and needs no
model; the live version puts this model's real maps, values and probabilities in the same
places. Images inside the SVG are data URIs, which browsers allow in an SVG shown as <img>."""
import base64
from html import escape

import cv2
import numpy as np

from src import config as C

BG = "#1A242B"
TEXT = "#E3EAEE"
MUTED = "#8A99A3"
FROZEN = "#51606B"
TUNED = "#2E6F8E"
HEAD = "#4FA3C7"
LINE = "#3B4A54"

STACK_SIZES = (112, 100, 88, 76, 64)
STACK_OFFSET = 6
FONT = "font-family:Segoe UI,Helvetica,Arial,sans-serif"

_CSS = f"""
<style>
.cam {{ display: flex; gap: 14px; align-items: center; flex-wrap: wrap; color: {TEXT}; }}
.cam figure {{ margin: 0; text-align: center; }}
.cam img {{ border-radius: 4px; display: block; }}
.cam figcaption {{ font-size: 12px; color: {MUTED}; margin-top: 4px; max-width: 220px; }}
.cam .op {{ font-size: 12px; color: {MUTED}; max-width: 150px; text-align: center; }}
</style>
"""


def data_uri(rgb: np.ndarray, size: int | None = None) -> str:
    img = rgb if size is None else cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def svg_img(svg: str, max_width: int) -> str:
    """HTML that shows an SVG string as a scalable picture, capped at its natural width."""
    uri = "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()
    return f'<img src="{uri}" style="width:100%;max-width:{max_width}px;height:auto">'


def _viridis_hex(v: float) -> str:
    b, g, r = cv2.applyColorMap(np.uint8([[int(np.clip(v, 0, 1) * 255)]]),
                                cv2.COLORMAP_VIRIDIS)[0, 0]
    return f"#{r:02x}{g:02x}{b:02x}"


def _depth(channels: int) -> int:
    """Cards drawn behind the front map: more channels, deeper stack (log scale)."""
    return int(np.clip(round(np.log2(channels)) - 3, 2, 7))


def _text(x, y, s, size=12, colour=TEXT, weight=400, anchor="middle") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{colour}" '
            f'font-weight="{weight}" text-anchor="{anchor}" style="{FONT}">{escape(s)}</text>')


def _grid_card(x, y, size, cells, colour) -> str:
    """Static stand-in for a feature map: a tinted card with a grid, finer for bigger maps."""
    lines = [f'<rect x="{x}" y="{y}" width="{size}" height="{size}" rx="3" fill="{colour}" '
             f'stroke="#0008"/>']
    step = size / cells
    for k in range(1, cells):
        p = k * step
        lines.append(f'<line x1="{x + p:.1f}" y1="{y}" x2="{x + p:.1f}" y2="{y + size}" '
                     f'stroke="#FFFFFF22" stroke-width="0.7"/>')
        lines.append(f'<line x1="{x}" y1="{y + p:.1f}" x2="{x + size}" y2="{y + p:.1f}" '
                     f'stroke="#FFFFFF22" stroke-width="0.7"/>')
    return "".join(lines)


def _stack(x, top, size, channels, colour, front: str | None) -> tuple[str, float]:
    """Back cards step down-left from the front card, like a deck. Returns (svg, width)."""
    n = _depth(channels)
    parts = []
    for k in range(n):
        o = (n - k) * STACK_OFFSET
        parts.append(f'<rect x="{x + k * STACK_OFFSET}" y="{top + o}" width="{size}" '
                     f'height="{size}" rx="3" fill="{colour}" stroke="#0006" '
                     f'opacity="{0.35 + 0.5 * k / n:.2f}"/>')
    fx = x + n * STACK_OFFSET
    if front is None:
        parts.append(_grid_card(fx, top, size, max(3, int(size / 12)), colour))
    else:
        parts.append(f'<image href="{front}" x="{fx}" y="{top}" width="{size}" '
                     f'height="{size}" preserveAspectRatio="none" '
                     f'style="image-rendering:pixelated"/>')
        parts.append(f'<rect x="{fx}" y="{top}" width="{size}" height="{size}" rx="3" '
                     f'fill="none" stroke="#0008"/>')
    return "".join(parts), size + n * STACK_OFFSET


def _column(values, k, y0, y1):
    """Evenly spaced positions for the k strongest values, scaled 0-1. Static: all 0.5."""
    if values is None:
        return [(y, 0.5) for y in np.linspace(y0, y1, k)]
    idx = np.sort(np.argsort(values)[::-1][:k])
    v = values[idx]
    v = (v - v.min()) / (v.max() - v.min() + 1e-8)
    return list(zip(np.linspace(y0, y1, k), v))


def architecture_svg(size: int, head_type: str, unfreeze: float, maps: list | None = None,
                     head: dict | None = None, input_rgb: np.ndarray | None = None
                     ) -> tuple[str, int]:
    """(svg, natural width). Static when maps is None, live otherwise."""
    from src.features import BACKBONE_LAYERS, TAPS

    live = maps is not None
    cut = int(BACKBONE_LAYERS * (1 - unfreeze))
    mid, top_y, label_y = 125, 55, 258
    parts, labels = [], []
    x = 12

    in_size = 120
    if live:
        parts.append(f'<image href="{data_uri(input_rgb, 240)}" x="{x}" y="{mid - in_size / 2}" '
                     f'width="{in_size}" height="{in_size}"/>')
    else:
        parts.append(f'<rect x="{x}" y="{mid - in_size / 2}" width="{in_size}" '
                     f'height="{in_size}" rx="3" fill="#05080A"/>'
                     f'<circle cx="{x + in_size / 2}" cy="{mid}" r="{in_size / 2 - 2}" '
                     f'fill="#8C3B1E"/><circle cx="{x + in_size / 2 + 18}" cy="{mid - 6}" '
                     f'r="9" fill="#D98A4E"/>')
    labels += [_text(x + in_size / 2, label_y, "Input", 13, weight=600),
               _text(x + in_size / 2, label_y + 17, f"{size} x {size} x 3", 11, MUTED)]
    x += in_size
    fe_start = None

    for i, ((layer, label, index, div, ch), px) in enumerate(zip(TAPS, STACK_SIZES)):
        parts.append(_text(x + 17, mid + 5, "→", 18, MUTED))
        x += 34
        fe_start = fe_start if fe_start is not None else x
        tuned = index >= cut
        colour = TUNED if tuned else FROZEN
        front = data_uri(maps[i]["front"], px * 2) if live else None
        svg, w = _stack(x, mid - (px + _depth(ch) * STACK_OFFSET) / 2, px, ch, colour, front)
        parts.append(svg)
        cx = x + w / 2
        side = size // div
        labels += [_text(cx, label_y, label, 13, weight=600),
                   _text(cx, label_y + 17, f"{side} x {side} x {ch:,}", 11, MUTED),
                   f'<rect x="{cx - 34}" y="{label_y + 25}" width="68" height="16" rx="8" '
                   f'fill="{colour}"/>',
                   _text(cx, label_y + 37, "Fine-tuned" if tuned else "Frozen", 10)]
        x += w
    fe_end = x

    parts.append(_text(x + 27, mid + 5, "→", 18, MUTED))
    x += 54
    head_start = x
    dense = head.get("dense_256") if live else None
    has_dense = head_type == "dense"
    pool = _column(head["gap"] if live else None, 12, mid - 90, mid + 90)
    nodes = _column(dense, 8, mid - 78, mid + 78) if has_dense else None
    px_pool, px_dense, px_out = x, x + 130, x + 240
    outs = np.linspace(mid - 92, mid + 92, C.N_CLASSES)
    probs = head["grade"] if live else None
    top = int(np.argmax(probs)) if live else None

    a = [(px_pool + 12, y) for y, _ in pool]
    b = [(px_dense, y) for y, _ in nodes] if nodes else None
    c = [(px_out, y) for y in outs]
    for src, dst in ([(a, b), (b, c)] if b else [(a, c)]):
        for x1, y1 in src:
            for x2, y2 in dst:
                parts.append(f'<line x1="{x1}" y1="{y1:.1f}" x2="{x2}" y2="{y2:.1f}" '
                             f'stroke="{LINE}" stroke-width="0.7"/>')
    for y, v in pool:
        fill = _viridis_hex(v) if live else HEAD
        parts.append(f'<rect x="{px_pool}" y="{y - 6:.1f}" width="12" height="12" rx="2" '
                     f'fill="{fill}"/>')
    for y, v in nodes or []:
        fill = _viridis_hex(v) if live else BG
        parts.append(f'<circle cx="{px_dense}" cy="{y:.1f}" r="8" fill="{fill}" '
                     f'stroke="{HEAD}" stroke-width="1.2"/>')
    for k, y in enumerate(outs):
        chosen = k == top
        parts.append(f'<rect x="{px_out}" y="{y - 13:.1f}" width="150" height="26" rx="5" '
                     f'fill="{BG}" stroke="{HEAD if chosen else LINE}" '
                     f'stroke-width="{2 if chosen else 1}"/>')
        if live:
            parts.append(f'<rect x="{px_out + 1}" y="{y - 12:.1f}" '
                         f'width="{148 * float(probs[k]):.1f}" height="24" rx="4" '
                         f'fill="{HEAD}" opacity="0.35"/>')
            parts.append(_text(px_out + 144, y + 4, f"{100 * float(probs[k]):.1f}%", 11,
                               anchor="end"))
        parts.append(_text(px_out + 8, y + 4, C.LABELS[k], 11, weight=700 if chosen else 400,
                           anchor="start"))
    head_end = px_out + 150

    pool_note = "12 of 1,280 shown" if live else "1,280 values"
    dense_note = "8 of 256 shown" if live else "256 units, ReLU"
    out_note = "grade probabilities" if live else "5 grade probabilities"
    labels += [_text(px_pool + 6, label_y, "Pooling", 13, weight=600),
               _text(px_pool + 6, label_y + 17, pool_note, 11, MUTED)]
    if has_dense:
        labels += [_text(px_dense, label_y, "Dense layer", 13, weight=600),
                   _text(px_dense, label_y + 17, dense_note, 11, MUTED)]
    labels += [_text(px_out + 75, label_y, "Output", 13, weight=600),
               _text(px_out + 75, label_y + 17, out_note, 11, MUTED)]

    by = label_y + 58

    def bracket(x0, x1, caption):
        return (f'<path d="M{x0} {by + 8} V{by} H{x1} V{by + 8}" fill="none" stroke="{LINE}" '
                f'stroke-width="2"/>' + _text((x0 + x1) / 2, by + 24, caption, 12, MUTED))

    labels.append(bracket(fe_start, fe_end, "Feature extraction: EfficientNetV2-B0, "
                                            "pretrained on ImageNet"))
    labels.append(bracket(head_start, head_end, "Classification: new layers, trained on DDR"))

    ly = by + 50
    legend = []
    lx = 12
    for name, colour in (("Frozen", FROZEN), ("Fine-tuned", TUNED), ("New", HEAD)):
        legend.append(f'<rect x="{lx}" y="{ly - 10}" width="11" height="11" rx="3" '
                      f'fill="{colour}"/>' + _text(lx + 17, ly, name, 12, MUTED, anchor="start"))
        lx += 110
    note = ("Front picture: the strongest of that layer's feature maps; the cards behind it "
            "stand for the rest." if live else
            "Grid density shows map resolution; stacked cards show the number of maps.")
    legend.append(_text(lx + 10, ly, note, 12, MUTED, anchor="start"))

    width, height = int(head_end + 14), int(ly + 14)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
           f'width="{width}" height="{height}">{"".join(parts)}{"".join(labels)}'
           f'{"".join(legend)}</svg>')
    return svg, width


def gradcam_strip(model_input: np.ndarray, last_front: np.ndarray, overlay: np.ndarray,
                  last_shape: tuple, grade: str) -> str:
    h, w, ch = last_shape
    fig = ('<figure><img src="{src}" width="200" height="200">'
           '<figcaption>{cap}</figcaption></figure>')
    op = '<div class="op">{}</div>'
    return _CSS + (
        '<div class="cam">'
        + fig.format(src=data_uri(model_input, 400), cap="Model input")
        + op.format("&rarr;<br>backbone")
        + fig.format(src=data_uri(last_front, 400),
                     cap=f"Last layer: {h} x {w} maps, {ch:,} of them (strongest shown)")
        + op.format(f"&rarr;<br>weight each map by how much it raised the {escape(grade)} "
                    f"score, add them up")
        + fig.format(src=data_uri(overlay, 400), cap=f"Grad-CAM for {escape(grade)}")
        + '</div>')
