"""
Project-wide configuration, path resolution, logging and small helpers.

The module is import-safe (no heavy ML imports at top level) so it can be
re-used from every other module, the Flask app and the CLI orchestrator.
"""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Project root = signature_fraud_detection/
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# Sub-folders expected by the spec
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
APP_DIR: Path = PROJECT_ROOT / "app"
UPLOADS_DIR: Path = APP_DIR / "static" / "uploads"


def _candidate_dataset_roots() -> List[Path]:
    """All known places where the CEDAR-style dataset might live."""
    return [
        # Default location — user copied the dataset under data/raw/
        RAW_DATA_DIR,
        # User-supplied via environment variable
        Path(os.environ.get("SIGNATURE_DATA_DIR", "")) if os.environ.get(
            "SIGNATURE_DATA_DIR"
        ) else RAW_DATA_DIR,
        # The original location used in this workspace
        PROJECT_ROOT.parent / "signature dataset" / "signatures",
        PROJECT_ROOT / "signature dataset" / "signatures",
    ]


def resolve_dataset_dir() -> Path:
    """Locate a directory that contains `full_org` and `full_forg` sub-folders.

    Raises:
        FileNotFoundError: if no candidate path contains the expected layout.
    """
    for candidate in _candidate_dataset_roots():
        if not candidate:
            continue
        if (candidate / "full_org").is_dir() and (candidate / "full_forg").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not locate the signature dataset. Expected a directory "
        "containing `full_org/` and `full_forg/`. Searched: "
        + ", ".join(str(p) for p in _candidate_dataset_roots() if p)
    )


def ensure_directories() -> None:
    """Create every standard sub-folder the project writes into."""
    for directory in (
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        RESULTS_DIR,
        UPLOADS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Hyper-parameter / runtime configuration
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """Central training / inference configuration.

    All values match the project specification. Adjust here, not in callers.
    """

    # Image geometry (height, width, channels). Spec: 155 x 220 RGB.
    img_height: int = 155
    img_width: int = 220
    img_channels: int = 3

    # Training
    batch_size: int = 32
    epochs: int = 77
    learning_rate: float = 1e-4
    margin: float = 2.0
    embedding_dim: int = 128

    # Pair generation
    pairs_per_writer: int = 12  # +/- pairs of each kind, per writer per epoch
    val_split: float = 0.15
    test_split: float = 0.15

    # Misc
    seed: int = 42
    model_name: str = "siamese_model.h5"
    mobilenet_model_name: str = "siamese_mobilenetv2.h5"
    backbone: str = "custom"  # "custom" or "mobilenetv2"
    use_augmentation: bool = True

    # Decision threshold on Euclidean distance — distance < threshold => genuine.
    # Automatically recalculated as the EER-optimal threshold on the validation
    # set after each training run (via sklearn roc_curve).  Falls back to 0.5
    # until the first training run completes.
    decision_threshold: float = 0.5

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return (self.img_height, self.img_width, self.img_channels)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


# A module-level singleton callers can mutate before training if desired.
CONFIG = Config()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_global_seed(seed: int | None = None) -> int:
    """Seed Python, NumPy and TensorFlow's RNGs.

    TF is imported lazily so importing this module stays cheap.
    """
    seed = CONFIG.seed if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import keras

        keras.utils.set_random_seed(seed)
    except Exception:  # pragma: no cover — Keras optional at import time
        pass
    return seed


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def get_logger(name: str = "signet") -> logging.Logger:
    """Return a singleton-style logger with a sensible default format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(levelname)-7s %(name)s — %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Dataset indexing
# ---------------------------------------------------------------------------


def index_signature_dataset(
    dataset_dir: Path | None = None,
) -> Tuple[Dict[int, List[Path]], Dict[int, List[Path]]]:
    """Group every image by writer id.

    Filenames follow the CEDAR convention:
        full_org/original_{writer}_{sample}.png
        full_forg/forgeries_{writer}_{sample}.png

    Returns:
        (genuine, forged): each maps writer-id -> sorted list of image paths.
    """
    dataset_dir = dataset_dir or resolve_dataset_dir()
    genuine: Dict[int, List[Path]] = {}
    forged: Dict[int, List[Path]] = {}

    def _collect(folder: Path, prefix: str, bucket: Dict[int, List[Path]]) -> None:
        for path in sorted(folder.glob("*.png")):
            name = path.stem  # e.g. "original_3_12"
            if not name.startswith(prefix):
                continue
            parts = name.split("_")
            if len(parts) < 3:
                continue
            try:
                writer_id = int(parts[1])
            except ValueError:
                continue
            bucket.setdefault(writer_id, []).append(path)

    _collect(dataset_dir / "full_org", "original_", genuine)
    _collect(dataset_dir / "full_forg", "forgeries_", forged)

    if not genuine or not forged:
        raise RuntimeError(
            f"No images indexed from {dataset_dir}. Check the dataset layout."
        )
    return genuine, forged


def split_writers(
    writer_ids: List[int],
    val_split: float,
    test_split: float,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """Split writers (not pairs) into train/val/test for a clean evaluation.

    Splitting on writers prevents identity leakage between the splits.
    """
    rng = random.Random(seed)
    shuffled = writer_ids.copy()
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_test = max(1, int(round(n_total * test_split)))
    n_val = max(1, int(round(n_total * val_split)))
    test_ids = sorted(shuffled[:n_test])
    val_ids = sorted(shuffled[n_test : n_test + n_val])
    train_ids = sorted(shuffled[n_test + n_val :])
    return train_ids, val_ids, test_ids


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)
