"""
Inference module.

Public surface:

* :class:`SignatureVerifier` — load a saved Siamese model once and reuse it.
* :func:`predict_pair`       — convenience one-shot prediction.
* CLI :  ``python -m src.predict signature1.png signature2.png``

The model's raw output is a Euclidean distance in [0, +inf) on an
L2-normalised embedding (so practical values fall roughly in [0, sqrt(2)]).

Similarity score (in %) is computed as

    similarity = max(0, 1 - distance / margin) * 100

and the binary verdict is

    GENUINE if distance < decision_threshold else FORGED
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np

# Make sure ``KERAS_BACKEND`` is set before keras is imported.
from . import __init__  # noqa: F401  (side-effect import)

import keras

from .model import get_custom_objects
from .preprocess import preprocess_image
from .utils import CONFIG, MODELS_DIR, get_logger

logger = get_logger(__name__)

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class PredictionResult:
    """Structured prediction result.

    Attributes:
        distance:          raw Euclidean distance in embedding space.
        similarity:        percentage similarity in [0, 100].
        label:             "GENUINE" or "FORGED".
        threshold:         decision threshold used.
        is_genuine:        bool convenience flag.
    """

    distance: float
    similarity: float
    label: str
    threshold: float
    is_genuine: bool

    def to_dict(self) -> dict:
        return {
            "distance": round(self.distance, 4),
            "similarity": round(self.similarity, 2),
            "label": self.label,
            "threshold": round(self.threshold, 4),
            "is_genuine": self.is_genuine,
        }


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class SignatureVerifier:
    """Cached wrapper around the trained Siamese model."""

    def __init__(
        self,
        model_path: PathLike | None = None,
        threshold: float | None = None,
        margin: float | None = None,
    ) -> None:
        self.model_path = Path(model_path or (MODELS_DIR / CONFIG.model_name))
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. Train it first via "
                "`python -m src.train` or pass an explicit --model-path."
            )

        self.threshold = float(
            threshold if threshold is not None else CONFIG.decision_threshold
        )
        self.margin = float(margin if margin is not None else CONFIG.margin)

        logger.info("Loading Siamese model from %s", self.model_path)
        self.model = keras.models.load_model(
            self.model_path,
            custom_objects=get_custom_objects(
                margin=self.margin, decision_threshold=self.threshold
            ),
            compile=False,
        )

    # -- core inference -----------------------------------------------------

    def distance(self, image_a: PathLike, image_b: PathLike) -> float:
        a = np.expand_dims(preprocess_image(image_a), 0)
        b = np.expand_dims(preprocess_image(image_b), 0)
        dist = self.model.predict([a, b], verbose=0).ravel()[0]
        return float(dist)

    def predict(self, image_a: PathLike, image_b: PathLike) -> PredictionResult:
        dist = self.distance(image_a, image_b)
        similarity = self._distance_to_similarity(dist)
        is_genuine = dist < self.threshold
        return PredictionResult(
            distance=dist,
            similarity=similarity,
            label="GENUINE" if is_genuine else "FORGED",
            threshold=self.threshold,
            is_genuine=is_genuine,
        )

    # -- helpers ------------------------------------------------------------

    def _distance_to_similarity(self, distance: float) -> float:
        """Map a distance to a [0, 100] similarity score.

        Uses the contrastive-loss margin as a soft upper bound; distances at
        or above the margin map to 0% similarity.
        """
        margin = max(self.margin, 1e-6)
        return float(max(0.0, 1.0 - distance / margin) * 100.0)


# ---------------------------------------------------------------------------
# Functional convenience
# ---------------------------------------------------------------------------


_VERIFIER_CACHE: dict[str, SignatureVerifier] = {}


def _cached_verifier(
    model_path: PathLike | None,
    threshold: float | None,
    margin: float | None,
) -> SignatureVerifier:
    key = f"{model_path}|{threshold}|{margin}"
    if key not in _VERIFIER_CACHE:
        _VERIFIER_CACHE[key] = SignatureVerifier(model_path, threshold, margin)
    return _VERIFIER_CACHE[key]


def predict_pair(
    image_a: PathLike,
    image_b: PathLike,
    model_path: PathLike | None = None,
    threshold: float | None = None,
    margin: float | None = None,
) -> PredictionResult:
    """Predict whether two signatures belong to the same person."""
    verifier = _cached_verifier(model_path, threshold, margin)
    return verifier.predict(image_a, image_b)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two signature images with a trained Siamese model."
    )
    parser.add_argument("image_a", type=str, help="Path to the first signature image.")
    parser.add_argument("image_b", type=str, help="Path to the second signature image.")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--margin", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    result = predict_pair(
        args.image_a,
        args.image_b,
        model_path=args.model_path,
        threshold=args.threshold,
        margin=args.margin,
    )

    print("\n" + "=" * 50)
    print(f"  Similarity Score : {result.similarity:.2f}%")
    print(f"  Distance         : {result.distance:.4f}")
    print(f"  Threshold        : {result.threshold:.4f}")
    print(f"  Prediction       : {result.label}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
