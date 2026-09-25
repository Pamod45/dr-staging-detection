"""ScreeningEngine: photograph in, everything the Screening page shows out.

    preprocess -> model -> decisions -> Grad-CAM -> overlay + attention summary

Loaded once per server process (the page caches it with st.cache_resource).
"""
import time
from dataclasses import dataclass

import numpy as np

from src import config as C
from src import decisions, gradcam, model
from src.preprocess import preprocess, to_model_input

CAM_TOLERANCE = 2e-3   # notebook 6.3 assert: Grad-CAM path must reproduce the model's output


class NoRetinaFound(ValueError):
    """Preprocessing could not find a retinal disc in the image."""


@dataclass
class ScreeningResult:
    model_id: str
    input_size: int
    processed_rgb: np.ndarray
    overlay_rgb: np.ndarray
    probs: np.ndarray
    decision: decisions.Decision
    thresholds: decisions.Thresholds
    attention: dict
    seconds: float


class ScreeningEngine:
    def __init__(self, model_id: str = C.SCREENING_MODEL_ID):
        self.model_id = model_id
        self.model, self.base, self.arch = model.load(model_id)
        self.cam = gradcam.GradCAM(self.model, self.base)
        self.thresholds = decisions.load_thresholds(model_id)
        self._cam_checked = False

    def run(self, image_bgr: np.ndarray) -> ScreeningResult:
        t0 = time.perf_counter()
        size = self.arch.input_size
        processed, _ = preprocess(image_bgr, out=size)
        if processed is None:
            raise NoRetinaFound("No retina could be found in this image.")
        x = to_model_input(processed, size)[None]

        cam, cam_probs = self.cam(x)
        if not self._cam_checked:
            p = self.model.predict(x, verbose=0)[0]
            diff = float(np.abs(p - cam_probs).max())
            if diff > CAM_TOLERANCE:
                raise RuntimeError(f"Grad-CAM path disagrees with the model by {diff:.4f}")
            self._cam_checked = True
        probs = cam_probs.astype(np.float64)
        probs = probs / probs.sum()

        rgb = x[0].astype(np.uint8)
        return ScreeningResult(
            model_id=self.model_id,
            input_size=size,
            processed_rgb=rgb,
            overlay_rgb=gradcam.overlay(rgb, cam),
            probs=probs,
            decision=decisions.decide(probs, self.thresholds),
            thresholds=self.thresholds,
            attention=gradcam.summarise(cam, size),
            seconds=time.perf_counter() - t0,
        )
