"""
Pair generator for the Siamese network.

Two public surfaces:

* ``build_pair_index``      — deterministic enumeration of pairs used for
                              evaluation / a static training set.
* ``SignaturePairSequence`` — a ``tf.keras.utils.Sequence`` subclass that
                              streams *balanced* batches of pairs from disk
                              without materialising the full dataset in RAM.

Labelling convention used everywhere in the project:

    label = 1  -> pair belongs to the same writer and BOTH are genuine
                  (i.e. the model should output a SMALL distance).
    label = 0  -> negative pair, one of:
                    a) genuine vs forgery of the SAME writer (skilled forgery),
                    b) genuine vs genuine from DIFFERENT writers.

The pair generator mixes both types of negatives so the model learns
both forgery detection and identity discrimination.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

# Make sure ``KERAS_BACKEND`` is set before keras is imported.
from . import __init__  # noqa: F401  (side-effect import)

try:
    import keras  # noqa: F401  — only the PyDataset base class is used
    _HAS_KERAS = True
except Exception:  # pragma: no cover
    _HAS_KERAS = False

from .preprocess import augment_image, preprocess_image
from .utils import CONFIG, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Static pair index (used for validation / test)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pair:
    path_a: Path
    path_b: Path
    label: int  # 1 = genuine pair, 0 = forged / different writer


def build_pair_index(
    genuine: Dict[int, List[Path]],
    forged: Dict[int, List[Path]],
    writer_ids: Sequence[int],
    pairs_per_writer: int,
    seed: int = 42,
) -> List[Pair]:
    """Deterministically build a balanced list of pairs.

    For each writer we emit roughly:
        * ``pairs_per_writer`` positives: genuine-vs-genuine, same writer.
        * ``pairs_per_writer // 2`` negatives: genuine-vs-forgery, same writer.
        * ``pairs_per_writer // 2`` negatives: genuine-vs-genuine, different
          writer (sampled uniformly from the supplied writer pool).

    Args:
        genuine: writer_id -> list of genuine image paths.
        forged:  writer_id -> list of forged image paths.
        writer_ids: subset of writer ids this split is allowed to touch.
        pairs_per_writer: density knob (see above).
        seed: RNG seed for reproducibility.

    Returns:
        A shuffled list of :class:`Pair` records.
    """
    rng = random.Random(seed)
    pairs: List[Pair] = []
    writer_ids = list(writer_ids)

    if len(writer_ids) < 2:
        raise ValueError("Need at least 2 writers to build negative pairs.")

    n_neg_each = max(1, pairs_per_writer // 2)

    for wid in writer_ids:
        gens = genuine.get(wid, [])
        if len(gens) < 2:
            continue

        # ---- positives: same writer, both genuine -------------------------
        for _ in range(pairs_per_writer):
            a, b = rng.sample(gens, 2)
            pairs.append(Pair(a, b, 1))

        # ---- negatives type A: genuine vs forgery, same writer ------------
        forgs = forged.get(wid, [])
        if forgs:
            for _ in range(n_neg_each):
                a = rng.choice(gens)
                b = rng.choice(forgs)
                pairs.append(Pair(a, b, 0))

        # ---- negatives type B: genuine vs genuine, different writer -------
        other_ids = [w for w in writer_ids if w != wid and genuine.get(w)]
        if other_ids:
            for _ in range(n_neg_each):
                other = rng.choice(other_ids)
                a = rng.choice(gens)
                b = rng.choice(genuine[other])
                pairs.append(Pair(a, b, 0))

    rng.shuffle(pairs)
    logger.info(
        "Built %d pairs from %d writers (pairs_per_writer=%d)",
        len(pairs),
        len(writer_ids),
        pairs_per_writer,
    )
    return pairs


def materialise_pairs(pairs: Sequence[Pair]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load every pair into memory. Suitable for small evaluation sets only."""
    xa = np.stack([preprocess_image(p.path_a) for p in pairs], axis=0)
    xb = np.stack([preprocess_image(p.path_b) for p in pairs], axis=0)
    y = np.asarray([p.label for p in pairs], dtype=np.float32)
    return xa, xb, y


# ---------------------------------------------------------------------------
# Streaming generator (training)
# ---------------------------------------------------------------------------


if _HAS_KERAS:
    _SequenceBase = keras.utils.PyDataset  # type: ignore[attr-defined]
else:  # pragma: no cover — allows the file to import without Keras for tooling

    class _SequenceBase:  # type: ignore[no-redef]
        pass


class SignaturePairSequence(_SequenceBase):
    """A Keras ``Sequence`` that streams balanced signature pairs.

    The generator stores only the *paths* of the images and produces pairs on
    the fly, which keeps memory consumption O(#images) instead of O(#pairs²).

    Each emitted batch is class-balanced (half positives, half negatives) so
    the contrastive loss receives a stable gradient signal.
    """

    def __init__(
        self,
        genuine: Dict[int, List[Path]],
        forged: Dict[int, List[Path]],
        writer_ids: Sequence[int],
        batch_size: int = 32,
        steps_per_epoch: int | None = None,
        pairs_per_writer: int = 12,
        augment: bool = True,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        if not _HAS_KERAS:
            raise RuntimeError(
                "Keras is required for SignaturePairSequence. "
                "Install it via `pip install keras`."
            )
        # PyDataset accepts worker/queue kwargs; we default to single-process.
        super().__init__(workers=1, use_multiprocessing=False, max_queue_size=10)

        if batch_size % 2 != 0:
            raise ValueError("batch_size must be even for balanced batches.")

        self.genuine = {w: list(p) for w, p in genuine.items() if p}
        self.forged = {w: list(p) for w, p in forged.items() if p}
        self.writer_ids = [w for w in writer_ids if w in self.genuine]

        if len(self.writer_ids) < 2:
            raise ValueError("Need >= 2 writers with genuine samples in this split.")

        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.seed = seed
        self.pairs_per_writer = pairs_per_writer
        self._rng = np.random.default_rng(seed)
        self._py_rng = random.Random(seed)

        # Approx number of unique positive pairs in this split -> steps/epoch
        if steps_per_epoch is None:
            n_total_pairs = len(self.writer_ids) * pairs_per_writer * 2
            steps_per_epoch = max(1, n_total_pairs // batch_size)
        self._steps = steps_per_epoch

    # -- Sequence protocol --------------------------------------------------

    def __len__(self) -> int:
        return self._steps

    def __getitem__(self, index: int) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        half = self.batch_size // 2
        xa = np.empty(
            (self.batch_size, CONFIG.img_height, CONFIG.img_width, CONFIG.img_channels),
            dtype=np.float32,
        )
        xb = np.empty_like(xa)
        y = np.empty(self.batch_size, dtype=np.float32)

        # positives — same writer, both genuine
        for i in range(half):
            wid = self._py_rng.choice(self.writer_ids)
            gens = self.genuine[wid]
            if len(gens) < 2:
                a = b = gens[0]
            else:
                a, b = self._py_rng.sample(gens, 2)
            xa[i] = self._load(a)
            xb[i] = self._load(b)
            y[i] = 1.0

        # negatives — alternate between forgery-vs-genuine and cross-writer
        for j in range(half):
            i = half + j
            wid = self._py_rng.choice(self.writer_ids)
            gens = self.genuine[wid]
            if j % 2 == 0 and self.forged.get(wid):
                # skilled forgery negative
                a = self._py_rng.choice(gens)
                b = self._py_rng.choice(self.forged[wid])
            else:
                # cross-writer negative
                other_ids = [w for w in self.writer_ids if w != wid]
                other = self._py_rng.choice(other_ids)
                a = self._py_rng.choice(gens)
                b = self._py_rng.choice(self.genuine[other])
            xa[i] = self._load(a)
            xb[i] = self._load(b)
            y[i] = 0.0

        # in-batch shuffle so positives and negatives are interleaved
        order = self._rng.permutation(self.batch_size)
        return (xa[order], xb[order]), y[order]

    def on_epoch_end(self) -> None:
        if self.shuffle:
            # Re-seed the PY RNG so each epoch yields fresh pairs.
            self._py_rng.seed(self._py_rng.randint(0, 2**31 - 1))

    # -- internals ----------------------------------------------------------

    def _load(self, path: Path) -> np.ndarray:
        image = preprocess_image(path)
        if self.augment:
            image = augment_image(image, self._rng)
        return image


__all__ = [
    "Pair",
    "build_pair_index",
    "materialise_pairs",
    "SignaturePairSequence",
]
