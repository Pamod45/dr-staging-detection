"""Rebuild a trained model and load its weights: v2 notebook 4.1 and 5.3, v1 notebook 4.1."""
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
    """Notebook build_model with settings passed in. backbone_weights=None skips the
    ImageNet download, since load_weights overwrites every weight anyway."""
    import keras
    from keras import layers

    size = arch.input_size
    # Layer names must match the notebook: Keras 3 stores weights by layer path, and a
    # renamed layer silently keeps random weights. Input is raw RGB 0-255 because
    # include_preprocessing=True puts normalisation inside the backbone.
    inp = keras.Input((size, size, 3), name="rgb_0_255")
    base = keras.applications.EfficientNetV2B0(include_top=False, weights=backbone_weights,
                                               input_shape=(size, size, 3),
                                               include_preprocessing=True)
    base.trainable = False
    x = base(inp, training=False)
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


def build_v1(size: int, dropout: float = 0.3):
    """v1 notebook build_model('effnetv2b0'): pooling, dropout and the output layer only."""
    import keras
    from keras import layers

    inp = keras.Input((size, size, 3), name="raw_rgb_0_255")
    base = keras.applications.EfficientNetV2B0(include_top=False, weights=None,
                                               input_shape=(size, size, 3),
                                               include_preprocessing=True)
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="drop")(x)
    out = layers.Dense(C.N_CLASSES, activation="softmax", dtype="float32", name="grade")(x)
    return keras.Model(inp, out, name="effnetv2b0_dr"), base


def load_any(model_id: str):
    """(model, base, input_size) for a model from either notebook."""
    spec = C.MODELS[model_id]
    if spec.needs_clahe:
        raise ValueError(f"{model_id} needs the CLAHE step, which the app does not port")
    if spec.notebook == "v2":
        model, base, arch = load(model_id)
        return model, base, arch.input_size
    weights = registry.find_weights(model_id)
    if weights is None:
        raise FileNotFoundError(f"no weights found in models/{model_id}")
    model, base = build_v1(spec.input_size)
    model.load_weights(str(weights))
    return model, base, spec.input_size
