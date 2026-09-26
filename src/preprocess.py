"""Retina crop and mask, ported from v2 notebook section 2.1. Must stay identical to the
notebook: the model only ever saw images produced by exactly these steps."""
from dataclasses import dataclass

import cv2
import numpy as np

MASK_FRAC = 0.95
FOV_THRESH = 12
FOV_WORK = 512


@dataclass(frozen=True)
class Fov:
    """Detected retinal disc, in the original image's pixel coordinates."""
    cx: float
    cy: float
    r: float


def detect_fov(img: np.ndarray, thresh: int = FOV_THRESH, work: int = FOV_WORK) -> Fov | None:
    s = work / max(img.shape[:2])
    sm = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    # brightest channel, not grey: IDRiD's blue channel is so dark that a grey threshold
    # cuts into the retina
    m = (sm.max(2) > thresh).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    # largest contour, so the camera's marker notch cannot win over the disc
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    return Fov((x + w / 2) / s, (y + h / 2) / s, max(w, h) / 2 / s)


def circle_mask(side: int, frac: float = 1.0) -> np.ndarray:
    yy, xx = np.ogrid[:side, :side]
    c = side / 2
    return (xx - c) ** 2 + (yy - c) ** 2 <= (c * frac) ** 2


def preprocess(img: np.ndarray, out: int, mask_frac: float = MASK_FRAC):
    """BGR uint8 image -> (out x out BGR uint8, Fov), or (None, None) if no retina is found."""
    fov = detect_fov(img)
    if fov is None:
        return None, None
    cx, cy, r = fov.cx, fov.cy, fov.r
    r *= mask_frac
    side = int(round(2 * r))
    x0, y0 = int(round(cx - r)), int(round(cy - r))
    canvas = np.zeros((side, side, 3), np.uint8)
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + side, img.shape[1]), min(y0 + side, img.shape[0])
    canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    canvas[~circle_mask(side)] = 0
    interp = cv2.INTER_AREA if side > out else cv2.INTER_CUBIC
    return cv2.resize(canvas, (out, out), interpolation=interp), fov


def to_model_input(img_bgr: np.ndarray, size: int) -> np.ndarray:
    """Resize if needed, BGR to RGB, float32 in 0-255."""
    if img_bgr.shape[0] != size:
        img_bgr = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
