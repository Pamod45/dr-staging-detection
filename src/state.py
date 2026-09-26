"""Shared uploaded image and per-image cache in session state."""
import hashlib
from dataclasses import dataclass

import cv2
import numpy as np
import streamlit as st

_IMAGE_KEY = "shared_image"
_NONCE_KEY = "uploader_nonce"
_CACHE_KEY = "per_image_cache"


@dataclass(frozen=True)
class SharedImage:
    name: str
    sha256: str
    raw: bytes
    image_bgr: np.ndarray

    @property
    def short_hash(self) -> str:
        return self.sha256[:10]

    @property
    def image_rgb(self) -> np.ndarray:
        return cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB)


def decode(raw: bytes, name: str) -> SharedImage:
    """Bytes -> SharedImage. Raises ValueError if the bytes are not a readable image."""
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"{name} could not be read as an image. Use a JPG or PNG file.")
    return SharedImage(name, hashlib.sha256(raw).hexdigest(), raw, img)


def get_image() -> SharedImage | None:
    return st.session_state.get(_IMAGE_KEY)


def set_image(img: SharedImage) -> bool:
    """Store a new image; True if it replaced a different one."""
    current = get_image()
    if current is not None and current.sha256 == img.sha256:
        return False
    st.session_state[_IMAGE_KEY] = img
    return True


def clear_image() -> None:
    """Forget the image and everything derived from it, and reset the uploader widget.
    Clearing is an explicit button because Streamlit empties the uploader whenever the
    user changes page, so an empty uploader does not mean the image is unwanted."""
    st.session_state.pop(_IMAGE_KEY, None)
    st.session_state[_CACHE_KEY] = {}
    st.session_state[_NONCE_KEY] = st.session_state.get(_NONCE_KEY, 0) + 1


def uploader_key() -> str:
    return f"uploader_{st.session_state.get(_NONCE_KEY, 0)}"


def cache_get(namespace: str):
    """Value stored for the current image under namespace, or None."""
    img = get_image()
    if img is None:
        return None
    return st.session_state.get(_CACHE_KEY, {}).get(namespace, {}).get(img.sha256)


def cache_set(namespace: str, value) -> None:
    """Store a value for the current image. Keyed by hash, so a new upload never sees it."""
    img = get_image()
    if img is None:
        raise RuntimeError("cache_set called with no image loaded")
    cache = st.session_state.setdefault(_CACHE_KEY, {})
    cache.setdefault(namespace, {})[img.sha256] = value
