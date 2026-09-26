"""Run any comparison model on one photograph, fed exactly the way its notebook fed it:

    v1: preprocess to the 512 px cache size, then geom_eval to the model's input size
    v2: preprocess to the run's CACHE_SIZE, then an INTER_AREA resize to IMG_SIZE if different
"""
import json
from dataclasses import dataclass

import cv2
import numpy as np

from src import config as C
from src import gradcam, model, registry
from src.inference import CAM_TOLERANCE, NoRetinaFound
from src.preprocess import geom_eval, preprocess, to_model_input

V1_CACHE_SIZE = 512


@dataclass
class RunOutput:
    model_id: str
    input_rgb: np.ndarray
    overlay_rgb: np.ndarray
    probs: np.ndarray


def model_input(image_bgr: np.ndarray, model_id: str) -> np.ndarray:
    """(size, size, 3) float32 RGB 0-255, or NoRetinaFound."""
    spec = C.MODELS[model_id]
    if spec.notebook == "v1":
        cached, _ = preprocess(image_bgr, out=V1_CACHE_SIZE)
        if cached is None:
            raise NoRetinaFound("No retina could be found in this image.")
        x = geom_eval(cached, spec.input_size)
        return cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32)
    cfg = json.loads((registry.model_dir(model_id) / "config.json").read_text())
    size = int(cfg["IMG_SIZE"])
    cached, _ = preprocess(image_bgr, out=int(cfg.get("CACHE_SIZE", size)))
    if cached is None:
        raise NoRetinaFound("No retina could be found in this image.")
    return to_model_input(cached, size)


class ModelRunner:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model, self.base, self.input_size = model.load_any(model_id)
        self.cam = gradcam.GradCAM(self.model, self.base)
        self._checked = False

    def run(self, image_bgr: np.ndarray) -> RunOutput:
        x = model_input(image_bgr, self.model_id)
        if x.shape[0] != self.input_size:
            raise ValueError(f"{self.model_id}: built {x.shape[0]} px input for a "
                             f"{self.input_size} px model")
        cam, probs = self.cam(x[None])
        if not self._checked:
            diff = float(np.abs(self.model.predict(x[None], verbose=0)[0] - probs).max())
            if diff > CAM_TOLERANCE:
                raise RuntimeError(f"Grad-CAM path disagrees with the model by {diff:.4f}")
            self._checked = True
        probs = probs.astype(np.float64)
        rgb = x.astype(np.uint8)
        return RunOutput(self.model_id, rgb, gradcam.overlay(rgb, cam), probs / probs.sum())
