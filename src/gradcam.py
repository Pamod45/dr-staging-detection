"""Grad-CAM - port of v2 notebook 6.3.

The backbone is run as its own model (base.input -> base.output) and the head layers are
applied one by one after it, so the gradient can be taken at the backbone's last feature map
(24 x 24 x 1280 at 768 px). The notebook asserts this path gives the same probabilities as the
full model; the engine repeats that check once per session.
"""
import cv2
import numpy as np

from src.preprocess import circle_mask

HOT = 0.5            # a CAM cell counts as "hot" at half the maximum or above
OVERLAY_ALPHA = 0.45  # notebook display setting


class GradCAM:
    def __init__(self, model, base):
        import keras
        self._tf = __import__("tensorflow")
        self.feat_model = keras.Model(base.input, base.output)
        names = [layer.name for layer in model.layers]
        self.head = model.layers[names.index("gap"):]

    def __call__(self, x: np.ndarray, cls: int | None = None):
        """x: (1, H, W, 3) float32 0-255. Returns (cam in 0-1 at feature-map size, probs)."""
        tf = self._tf
        x = tf.constant(x, tf.float32)
        with tf.GradientTape() as tape:
            conv = self.feat_model(x, training=False)
            tape.watch(conv)
            z = conv
            for layer in self.head:
                z = layer(z, training=False)
            target = tf.argmax(z, axis=1) if cls is None else tf.constant([cls], tf.int64)
            score = tf.gather(z, target, axis=1, batch_dims=1)
        g = tape.gradient(score, conv)
        cam = tf.nn.relu(tf.reduce_sum(tf.reduce_mean(g, axis=(1, 2), keepdims=True) * conv, -1))
        cam = cam / (tf.reduce_max(cam, axis=(1, 2), keepdims=True) + 1e-8)
        return cam.numpy()[0], z.numpy()[0]


def upsample(cam: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(cam.astype(np.float32), (size, size))


def overlay(image_rgb: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """Jet heatmap over the image, like the notebook figures. Outside the retina stays black."""
    size = image_rgb.shape[0]
    heat = cv2.applyColorMap((upsample(cam, size) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    out = cv2.addWeighted(image_rgb, 1 - OVERLAY_ALPHA, heat, OVERLAY_ALPHA, 0)
    out[~circle_mask(size)] = 0
    return out


def _location(y: float, x: float, size: int) -> str:
    rows, cols = ("upper", "middle", "lower"), ("left", "centre", "right")
    r, c = min(int(3 * y / size), 2), min(int(3 * x / size), 2)
    if (r, c) == (1, 1):
        return "centre"
    if r == 1:
        return f"middle {cols[c]}"
    return f"{rows[r]} {cols[c]}" if c != 1 else f"{rows[r]} centre"


def summarise(cam: np.ndarray, size: int) -> dict:
    """Plain numbers describing where the attention sits, for the explanation text.
    Locations are image positions (upper left, centre...), not anatomy."""
    up = upsample(cam, size)
    retina = circle_mask(size)
    total = float(up.sum()) + 1e-8
    hot = (up >= HOT) & retina
    n_regions = cv2.connectedComponents(hot.astype(np.uint8))[0] - 1
    if hot.any():
        ys, xs = np.nonzero(hot)
        w = up[hot]
        where = _location(float((ys * w).sum() / w.sum()), float((xs * w).sum() / w.sum()), size)
    else:
        where = None
    return {
        "inside_retina_pct": 100 * float(up[retina].sum()) / total,
        "hot_area_pct": 100 * float(hot.sum()) / float(retina.sum()),
        "hot_regions": int(n_regions),
        "hot_location": where,
    }
