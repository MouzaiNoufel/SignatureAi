"""
Image preprocessing pipeline.

Pipeline (per the specification):
    1. Read image with OpenCV (BGR).
    2. Convert BGR -> RGB.
    3. Resize to 155 x 220 (H x W).
    4. Normalise pixel values to [0, 1].
    5. Return float32 NumPy array of shape (H, W, 3).

Also provides:
    - `augment_image`: light, signature-friendly augmentation for training.
    - `batch_preprocess`: convenience for a list of paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Union

import cv2
import numpy as np

from .utils import CONFIG, get_logger

logger = get_logger(__name__)

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Core preprocessing
# ---------------------------------------------------------------------------


def load_image(path: PathLike) -> np.ndarray:
    """Read an image from disk as an unmodified BGR uint8 NumPy array.

    Args:
        path: file system path to a readable image.

    Raises:
        FileNotFoundError: if the file is missing.
        ValueError:        if OpenCV cannot decode the file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    # `cv2.imread` returns None on failure rather than raising.
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode image: {path}")
    return image


def preprocess_image(
    path_or_array: Union[PathLike, np.ndarray],
    target_height: int | None = None,
    target_width: int | None = None,
) -> np.ndarray:
    """Run the full preprocessing pipeline on a single image.

    Args:
        path_or_array: filesystem path OR an already-decoded BGR/RGB array.
        target_height: defaults to CONFIG.img_height (155).
        target_width:  defaults to CONFIG.img_width  (220).

    Returns:
        float32 NumPy array, shape (H, W, 3), values in [0, 1], RGB order.
    """
    target_height = target_height or CONFIG.img_height
    target_width = target_width or CONFIG.img_width

    # 1. Read / accept array
    if isinstance(path_or_array, (str, Path)):
        image_bgr = load_image(path_or_array)
    else:
        image_bgr = np.asarray(path_or_array)
        if image_bgr.ndim == 2:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        elif image_bgr.shape[-1] == 4:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)

    # 2. BGR -> RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 3. Resize. cv2.resize takes (width, height).
    image_resized = cv2.resize(
        image_rgb, (target_width, target_height), interpolation=cv2.INTER_AREA
    )

    # 4. Normalise to [0, 1] as float32 — keeps memory low for batching.
    image_norm = image_resized.astype(np.float32) / 255.0

    return image_norm


def batch_preprocess(paths: Iterable[PathLike]) -> np.ndarray:
    """Vectorised preprocessing for a list of paths."""
    arrays: List[np.ndarray] = [preprocess_image(p) for p in paths]
    return np.stack(arrays, axis=0)


# ---------------------------------------------------------------------------
# Augmentation (signature-friendly: small geometric jitter only)
# ---------------------------------------------------------------------------


def augment_image(image: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Apply a light, label-preserving augmentation.

    We deliberately avoid horizontal flips (signatures are not symmetric) and
    aggressive colour jitter (signatures are dark strokes on light paper).

    The image is expected to be float32 in [0, 1], shape (H, W, 3).
    """
    rng = rng or np.random.default_rng()
    out = image
    h, w = out.shape[:2]

    # Rotation ±5°
    angle = float(rng.uniform(-5.0, 5.0))
    M_rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    out = cv2.warpAffine(
        out, M_rot, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(1.0, 1.0, 1.0)
    )

    # Translation ±5% of image dimensions
    tx = float(rng.uniform(-0.05, 0.05) * w)
    ty = float(rng.uniform(-0.05, 0.05) * h)
    T = np.float32([[1, 0, tx], [0, 1, ty]])
    out = cv2.warpAffine(
        out, T, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(1.0, 1.0, 1.0)
    )

    # Zoom (scale 0.95–1.05)
    scale = float(rng.uniform(0.95, 1.05))
    new_h, new_w = int(h * scale), int(w * scale)
    scaled = cv2.resize(out, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.ones((h, w, out.shape[2]), dtype=np.float32)
    y0 = max(0, (new_h - h) // 2)
    x0 = max(0, (new_w - w) // 2)
    sh = min(new_h, h)
    sw = min(new_w, w)
    py = max(0, (h - new_h) // 2)
    px = max(0, (w - new_w) // 2)
    canvas[py : py + sh, px : px + sw] = scaled[y0 : y0 + sh, x0 : x0 + sw]
    out = canvas

    # Shear ±5°
    shear_tan = float(np.tan(np.radians(rng.uniform(-5.0, 5.0))))
    M_shear = np.array(
        [[1.0, shear_tan, -shear_tan * h / 2], [0.0, 1.0, 0.0]], dtype=np.float32
    )
    out = cv2.warpAffine(
        out, M_shear, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(1.0, 1.0, 1.0)
    )

    # Brightness jitter
    if rng.random() < 0.5:
        factor = float(rng.uniform(0.9, 1.1))
        out = np.clip(out * factor, 0.0, 1.0)

    # Light Gaussian noise
    if rng.random() < 0.3:
        noise = rng.normal(0.0, 0.01, out.shape).astype(np.float32)
        out = np.clip(out + noise, 0.0, 1.0)

    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Convenience for the predict / web modules
# ---------------------------------------------------------------------------


def preprocess_pair(path_a: PathLike, path_b: PathLike) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess two paths and return arrays with a leading batch dimension."""
    a = preprocess_image(path_a)
    b = preprocess_image(path_b)
    return np.expand_dims(a, 0), np.expand_dims(b, 0)


__all__ = [
    "load_image",
    "preprocess_image",
    "batch_preprocess",
    "augment_image",
    "preprocess_pair",
]
