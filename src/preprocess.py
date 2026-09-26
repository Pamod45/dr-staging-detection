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


def geom_eval(img: np.ndarray, out: int) -> np.ndarray:
    """v1 notebook geom() on its evaluation path (no rotation, no flip, INTER_AREA). v1 models
    were fed the 512 px cache through this warp, then re-masked, at every input size."""
    src = img.shape[0]
    k = out / src
    m = cv2.getRotationMatrix2D((src / 2, src / 2), 0.0, k)
    m[0, 2] += out / 2 - src / 2
    m[1, 2] += out / 2 - src / 2
    o = cv2.warpAffine(np.ascontiguousarray(img), m, (out, out), flags=cv2.INTER_AREA,
                       borderValue=(0, 0, 0))
    o[~circle_mask(out)] = 0
    return o


def preprocess_steps(img: np.ndarray, out: int) -> dict | None:
    """The same steps as preprocess(), keeping each intermediate image for display.
    The 'final' image is identical to preprocess(img, out)[0]; a test checks this."""
    fov = detect_fov(img)
    if fov is None:
        return None
    r = fov.r * MASK_FRAC
    side = int(round(2 * r))
    x0, y0 = int(round(fov.cx - r)), int(round(fov.cy - r))
    crop = np.zeros((side, side, 3), np.uint8)
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + side, img.shape[1]), min(y0 + side, img.shape[0])
    crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    masked = crop.copy()
    masked[~circle_mask(side)] = 0
    interp = cv2.INTER_AREA if side > out else cv2.INTER_CUBIC
    return {
        "fov": fov,
        "side": side,
        "interp": "area averaging" if interp == cv2.INTER_AREA else "cubic",
        "crop": crop,
        "masked": masked,
        "final": cv2.resize(masked, (out, out), interpolation=interp),
    }


def draw_fov(img: np.ndarray, fov: Fov, mask_frac: float = MASK_FRAC) -> np.ndarray:
    """Detected disc (green) and the mask circle actually kept (blue) on a copy of the image."""
    out = img.copy()
    t = max(2, int(max(img.shape[:2]) / 250))
    centre = (int(round(fov.cx)), int(round(fov.cy)))
    cv2.circle(out, centre, int(round(fov.r)), (80, 200, 80), t)
    cv2.circle(out, centre, int(round(fov.r * mask_frac)), (199, 163, 79), t)
    cv2.drawMarker(out, centre, (80, 200, 80), cv2.MARKER_CROSS, 6 * t, t)
    return out
