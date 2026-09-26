"""Training-time augmentation, ported from v2 notebook section 3.3, for display only. The
screening path never calls this.

Random draws are made in the notebook's order, so a given generator state produces the same
image the notebook would, and each draw also returns the values it used for the captions."""
from dataclasses import dataclass

import cv2
import numpy as np

from src.preprocess import circle_mask


@dataclass(frozen=True)
class AugSettings:
    rotation: float
    zoom: tuple
    brightness: float
    contrast: float
    channel: float

    @classmethod
    def from_config(cls, cfg: dict) -> "AugSettings":
        return cls(float(cfg["ROTATION"]), tuple(cfg["ZOOM"]), float(cfg["BRIGHTNESS"]),
                   float(cfg["CONTRAST"]), float(cfg["CHANNEL"]))


def augment(img: np.ndarray, rng: np.random.Generator, s: AugSettings):
    side = img.shape[0]
    zoom = rng.uniform(*s.zoom)
    angle = rng.uniform(0, s.rotation)
    m = cv2.getRotationMatrix2D((side / 2, side / 2), angle, zoom)
    x = cv2.warpAffine(img, m, (side, side), flags=cv2.INTER_LINEAR, borderValue=0)
    hflip = rng.random() < 0.5
    if hflip:
        x = x[:, ::-1]
    vflip = rng.random() < 0.5
    if vflip:
        x = x[::-1]
    x = np.ascontiguousarray(x)
    inside = circle_mask(side, min(zoom, 1.0))
    mean = np.array(cv2.mean(x, mask=inside.astype(np.uint8))[:3], np.float32)
    x = x.astype(np.float32)
    contrast = rng.uniform(-s.contrast, s.contrast)
    x = (x - mean) * (1 + contrast) + mean
    brightness = rng.uniform(-s.brightness, s.brightness)
    channel = rng.uniform(-s.channel, s.channel, 3)
    x = x * (1 + brightness) * (1 + channel)
    x = np.clip(x, 0, 255).astype(np.uint8)
    x[~inside] = 0
    return x, {"angle": angle, "zoom": zoom, "hflip": hflip, "vflip": vflip,
               "brightness": brightness, "contrast": contrast, "channel": channel}


def augment_steps(img: np.ndarray, rng: np.random.Generator, s: AugSettings) -> list[dict]:
    """augment() one operation at a time, keeping each intermediate for display. Same random
    draws in the same order, so the last image equals augment() for the same generator state;
    a test checks this."""
    side = img.shape[0]
    steps = [{"title": "Model input", "image": img, "text": "The preprocessed image, unchanged."}]
    zoom = rng.uniform(*s.zoom)
    angle = rng.uniform(0, s.rotation)
    m = cv2.getRotationMatrix2D((side / 2, side / 2), angle, zoom)
    x = cv2.warpAffine(img, m, (side, side), flags=cv2.INTER_LINEAR, borderValue=0)
    steps.append({"title": "Rotate and zoom", "image": x,
                  "text": f"Rotated {angle:.0f} degrees and zoomed {zoom:.2f}x in one warp."})
    hflip = rng.random() < 0.5
    if hflip:
        x = x[:, ::-1]
    vflip = rng.random() < 0.5
    if vflip:
        x = x[::-1]
    x = np.ascontiguousarray(x)
    flips = [n for n, on in (("left-right", hflip), ("top-bottom", vflip)) if on]
    steps.append({"title": "Flip", "image": x,
                  "text": ("Flipped " + " and ".join(flips) + ".") if flips
                  else "No flip this time (each flip has a 50% chance)."})
    inside = circle_mask(side, min(zoom, 1.0))
    mean = np.array(cv2.mean(x, mask=inside.astype(np.uint8))[:3], np.float32)
    x = x.astype(np.float32)
    contrast = rng.uniform(-s.contrast, s.contrast)
    x = (x - mean) * (1 + contrast) + mean
    steps.append({"title": "Contrast", "image": np.clip(x, 0, 255).astype(np.uint8),
                  "text": f"Contrast {contrast:+.0%} around the retina's mean colour. The black "
                          f"edge shifts too; the last step fixes that."})
    brightness = rng.uniform(-s.brightness, s.brightness)
    channel = rng.uniform(-s.channel, s.channel, 3)
    x = x * (1 + brightness) * (1 + channel)
    x = np.clip(x, 0, 255).astype(np.uint8)
    b, g, r = channel
    steps.append({"title": "Brightness and colour", "image": x.copy(),
                  "text": f"Brightness {brightness:+.0%}, then red {r:+.1%}, green {g:+.1%}, "
                          f"blue {b:+.1%}, like a different camera's exposure and white balance."})
    x[~inside] = 0
    steps.append({"title": "Re-mask", "image": x,
                  "text": "Everything outside the retina set back to black, so the edge stays "
                          "clean. This is one training image."})
    return steps
