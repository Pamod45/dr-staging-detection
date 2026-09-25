"""Rebuild a trained model exactly and load its weights - port of v2 notebook 4.1 and 5.3.

Two traps that load without an error and then give wrong answers:

1. Layer names must match the notebook (gap, head_bn, drop_1, dense_256, drop_2, grade).
   Keras 3 stores weights by layer path, so a renamed layer is a layer with no weights.
2. The backbone must be in the state it was saved in: trainable, with the lower half frozen
   (notebook 5.3). We reproduce that before loading so the file and the model line up.

Normalisation lives inside the backbone (include_preprocessing=True), so the input is raw RGB
float32 in 0-255. No mixed precision here: it was a GPU speed setting for training.
"""
import json
from dataclasses import dataclass

from src import config as C
from src import registry


@dataclass(frozen=True)
class Architecture:
    input_size: int
    head: str
    dropout: float
    unfreeze: float


def architecture(model_id: str) -> Architecture:
    """Read the settings a v2 model was trained with from its own config.json."""
    spec = C.MODELS[model_id]
    if spec.notebook != "v2":
        raise NotImplementedError("v1 models are loaded by the comparison page (stage 4)")
    cfg = json.loads((registry.model_dir(model_id) / "config.json").read_text())
    arch = Architecture(int(cfg["IMG_SIZE"]), cfg["HEAD"], float(cfg["DROPOUT"]),
                        float(cfg["UNFREEZE"]))
    if arch.input_size != spec.input_size:
        raise ValueError(f"{model_id}: config.json says {arch.input_size} px, "
                         f"app config says {spec.input_size} px")
    if cfg.get("NORM", "none") != "none":
        raise ValueError(f"{model_id}: trained with NORM={cfg['NORM']}, which the app does not port")
    return arch


def build(arch: Architecture, backbone_weights=None):
    """Notebook build_model, unchanged apart from taking settings as an argument.
    backbone_weights=None skips the ImageNet download; load_weights overwrites them anyway."""
    import keras
    from keras import layers

    size = arch.input_size
    inp = keras.Input((size, size, 3), name="rgb_0_255")
    base = keras.applications.EfficientNetV2B0(include_top=False, weights=backbone_weights,
                                               input_shape=(size, size, 3),
                                               include_preprocessing=True)
    base.trainable = False
    x = base(inp, training=False)          # training=False keeps BatchNorm statistics frozen
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    if arch.head == "dense":
        x = layers.BatchNormalization(name="head_bn")(x)
        x = layers.Dropout(arch.dropout, name="drop_1")(x)
        x = layers.Dense(256, activation="relu", name="dense_256")(x)
        x = layers.Dropout(arch.dropout, name="drop_2")(x)
    else:
        x = layers.Dropout(arch.dropout, name="drop_1")(x)
    out = layers.Dense(C.N_CLASSES, activation="softmax", dtype="float32", name="grade")(x)
    return keras.Model(inp, out, name=f"effnetv2b0_{arch.head}"), base


def set_saved_state(base, unfreeze: float) -> None:
    """Backbone trainable, lower (1 - unfreeze) of its layers frozen - notebook 5.3."""
    base.trainable = True
    cut = int(len(base.layers) * (1 - unfreeze))
    for layer in base.layers[:cut]:
        layer.trainable = False


def load(model_id: str):
    """Build the model for model_id and load its trained weights. Returns (model, base, arch)."""
    weights = registry.find_weights(model_id)
    if weights is None:
        raise FileNotFoundError(f"no weights found in models/{model_id}")
    arch = architecture(model_id)
    model, base = build(arch)
    set_saved_state(base, arch.unfreeze)
    model.load_weights(str(weights))
    return model, base, arch
