"""Real intermediate activations of the screening model, for the architecture panel."""
import cv2
import numpy as np

# (layer, page label, layer index in the backbone, size divisor, channels). One tap wherever
# the backbone halves resolution. Index, divisor and channels let the static diagram be drawn
# without loading the model; a test checks them against the real network.
TAPS = [
    ("stem_activation", "First layer", 5, 2, 32),
    ("block2b_add", "Early blocks", 20, 4, 32),
    ("block3b_add", "Early-middle blocks", 32, 8, 48),
    ("block5e_add", "Middle blocks", 148, 16, 112),
    ("top_activation", "Last layer", 269, 32, 1280),
]
BACKBONE_LAYERS = 270


def extractor(engine):
    """keras.Model from the backbone input to each tap. Built once and kept on the engine."""
    if getattr(engine, "_feature_model", None) is None:
        import keras
        engine._feature_model = keras.Model(
            engine.base.input, [engine.base.get_layer(t[0]).output for t in TAPS])
    return engine._feature_model


def feature_maps(engine, x: np.ndarray) -> list[dict]:
    """x: (1, H, W, 3) float32 0-255. One dict per tap: shape, whether that layer was
    fine-tuned, the single most active channel, an 8-channel grid, and the raw map."""
    outs = extractor(engine)(x, training=False)
    maps = []
    for (name, label, *_), o in zip(TAPS, outs):
        fmap = np.asarray(o)[0]
        maps.append({
            "layer": name, "label": label, "shape": tuple(int(d) for d in fmap.shape),
            "tuned": bool(engine.base.get_layer(name).trainable),
            "front": colour(fmap[..., int(fmap.mean(axis=(0, 1)).argmax())]),
            "tiles": tile(fmap),
            "raw": fmap,
        })
    return maps


def colour(channel: np.ndarray, size: int = 256) -> np.ndarray:
    """One activation channel, stretched to its 1st-99th percentile range, viridis, RGB."""
    a = channel.astype(np.float32)
    lo, hi = np.percentile(a, [1, 99])
    a = np.clip((a - lo) / (hi - lo + 1e-8), 0, 1)
    img = cv2.resize((a * 255).astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)


def head_activations(engine, last_map: np.ndarray) -> dict:
    """Run the new layers on the backbone's last map, keeping what each one outputs:
    pooled vector (1280), the 256-unit layer if the head has one, and the probabilities."""
    names = [layer.name for layer in engine.model.layers]
    z = last_map[None].astype(np.float32)
    out = {}
    for layer in engine.model.layers[names.index("gap"):]:
        z = layer(z, training=False)
        if layer.name in ("gap", "dense_256", "grade"):
            out[layer.name] = np.asarray(z)[0]
    return out


def tile(fmap: np.ndarray, n: int = 8, cell: int = 128, gap: int = 3) -> np.ndarray:
    """The n most active channels in two rows, each scaled on its own, because raw channel
    ranges differ by orders of magnitude and the weak ones would otherwise look blank."""
    order = np.argsort(fmap.mean(axis=(0, 1)))[::-1][:n]
    cols = n // 2
    grid = np.zeros((2 * cell + gap, cols * cell + (cols - 1) * gap, 3), np.uint8)
    for k, ch in enumerate(order):
        r, c = divmod(k, cols)
        grid[r * (cell + gap):r * (cell + gap) + cell,
             c * (cell + gap):c * (cell + gap) + cell] = colour(fmap[..., ch], cell)
    return grid


def trainable_counts(engine) -> dict:
    """Trainable parameters in each training phase, counted on the loaded model, whose backbone
    is already in its phase 2 state (upper half trainable)."""
    names = [layer.name for layer in engine.model.layers]
    head = engine.model.layers[names.index("gap"):]
    head_n = sum(int(np.prod(w.shape)) for layer in head for w in layer.trainable_weights)
    total_trainable = sum(int(np.prod(w.shape)) for w in engine.model.trainable_weights)
    frozen = sum(1 for layer in engine.base.layers if not layer.trainable)
    return {"phase1": head_n, "phase2": total_trainable, "backbone_layers": len(engine.base.layers),
            "unfrozen_layers": len(engine.base.layers) - frozen, "total": engine.model.count_params()}
