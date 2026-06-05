"""
Training entry point for the Siamese signature-verification network.

Usage
-----

From within ``signature_fraud_detection/`` ::

    python -m src.train --epochs 77 --batch-size 32
    python -m src.train --backbone mobilenetv2 --epochs 30

The script:
    * resolves the dataset,
    * splits writers into train/val/test (writer-disjoint),
    * builds either the custom CNN or MobileNetV2 backbone,
    * trains with Adam + Contrastive Loss,
    * applies EarlyStopping, ModelCheckpoint and ReduceLROnPlateau,
    * saves the model to ``models/`` and a training-history plot to
      ``results/training_history.png``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np

# Make sure ``KERAS_BACKEND`` is set before keras is imported.
from . import __init__  # noqa: F401  (side-effect import)

import keras

from .evaluate import compute_eer_threshold
from .model import build_mobilenetv2_siamese_model, build_siamese_model
from .pair_generator import SignaturePairSequence, build_pair_index, materialise_pairs
from .utils import (
    CONFIG,
    MODELS_DIR,
    RESULTS_DIR,
    Config,
    ensure_directories,
    get_logger,
    index_signature_dataset,
    resolve_dataset_dir,
    save_json,
    set_global_seed,
    split_writers,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Siamese signature model.")
    parser.add_argument("--epochs", type=int, default=CONFIG.epochs)
    parser.add_argument("--batch-size", type=int, default=CONFIG.batch_size)
    parser.add_argument("--learning-rate", type=float, default=CONFIG.learning_rate)
    parser.add_argument("--margin", type=float, default=CONFIG.margin)
    parser.add_argument("--embedding-dim", type=int, default=CONFIG.embedding_dim)
    parser.add_argument(
        "--pairs-per-writer", type=int, default=CONFIG.pairs_per_writer
    )
    parser.add_argument(
        "--backbone",
        choices=("custom", "mobilenetv2"),
        default=CONFIG.backbone,
        help="Which feature-extractor to use.",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable training-time data augmentation.",
    )
    parser.add_argument(
        "--fine-tune-mobilenet",
        action="store_true",
        help="Unfreeze MobileNetV2 base for fine-tuning.",
    )
    parser.add_argument("--seed", type=int, default=CONFIG.seed)
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="EarlyStopping patience (epochs without val-loss improvement).",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Override model filename (defaults to siamese_model.h5).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def _build_callbacks(
    checkpoint_path: Path, patience: int
) -> list[keras.callbacks.Callback]:
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(3, patience // 2),
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.TerminateOnNaN(),
    ]


class HardNegativeMiningCallback(keras.callbacks.Callback):
    """Refreshes the training sequence's hard-negative pool every N epochs.

    For the first ``warmup_epochs`` epochs the model is not yet stable enough
    for meaningful hard-negative selection, so the sequence falls back to
    random negatives.  After warm-up the pool is rebuilt every
    ``refresh_every`` epochs so the mined pairs stay aligned with the
    evolving embeddings.
    """

    def __init__(
        self,
        sequence: SignaturePairSequence,
        warmup_epochs: int = 5,
        refresh_every: int = 3,
    ) -> None:
        super().__init__()
        self.sequence = sequence
        self.warmup_epochs = warmup_epochs
        self.refresh_every = refresh_every

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        if epoch < self.warmup_epochs:
            return
        if (epoch - self.warmup_epochs) % self.refresh_every == 0:
            self.sequence.model = self.model
            self.sequence._refresh_hard_negative_pool()


def _plot_history(history: dict, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["loss"], label="train")
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Contrastive Loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    acc_key = "siamese_accuracy"
    if acc_key in history:
        axes[1].plot(history[acc_key], label="train")
    if f"val_{acc_key}" in history:
        axes[1].plot(history[f"val_{acc_key}"], label="val")
    axes[1].set_title("Pair Accuracy @ threshold")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    logger.info("Saved training-history plot to %s", output_path)


# ---------------------------------------------------------------------------
# Public training routine
# ---------------------------------------------------------------------------


def train(config: Config | None = None, **overrides) -> Dict[str, object]:
    """Run a full training session.

    Args:
        config:    Optional explicit Config; defaults to the module singleton.
        overrides: Any Config field name -> value.

    Returns:
        Dict with ``model_path``, ``history_path``, ``writer_splits`` and the
        final Keras ``History.history`` dict.
    """
    cfg = config or CONFIG
    for k, v in overrides.items():
        if hasattr(cfg, k) and v is not None:
            setattr(cfg, k, v)

    ensure_directories()
    set_global_seed(cfg.seed)

    dataset_dir = resolve_dataset_dir()
    logger.info("Dataset directory: %s", dataset_dir)

    genuine, forged = index_signature_dataset(dataset_dir)
    writer_ids = sorted(set(genuine) & set(forged))
    train_ids, val_ids, test_ids = split_writers(
        writer_ids, cfg.val_split, cfg.test_split, cfg.seed
    )
    logger.info(
        "Writer split — train=%d  val=%d  test=%d",
        len(train_ids),
        len(val_ids),
        len(test_ids),
    )

    train_seq = SignaturePairSequence(
        genuine,
        forged,
        writer_ids=train_ids,
        batch_size=cfg.batch_size,
        pairs_per_writer=cfg.pairs_per_writer,
        augment=cfg.use_augmentation,
        shuffle=True,
        seed=cfg.seed,
        mine_hard_negatives=True,
        mining_ratio=4,
    )
    val_seq = SignaturePairSequence(
        genuine,
        forged,
        writer_ids=val_ids,
        batch_size=cfg.batch_size,
        pairs_per_writer=max(4, cfg.pairs_per_writer // 2),
        augment=False,
        shuffle=False,
        seed=cfg.seed + 1,
    )

    if cfg.backbone == "mobilenetv2":
        model = build_mobilenetv2_siamese_model(
            input_shape=cfg.input_shape,
            embedding_dim=cfg.embedding_dim,
            learning_rate=cfg.learning_rate,
            margin=cfg.margin,
            decision_threshold=cfg.decision_threshold,
            trainable_base=overrides.get("trainable_base", False),
        )
        model_filename = cfg.mobilenet_model_name
    else:
        model = build_siamese_model(
            input_shape=cfg.input_shape,
            embedding_dim=cfg.embedding_dim,
            learning_rate=cfg.learning_rate,
            margin=cfg.margin,
            decision_threshold=cfg.decision_threshold,
        )
        model_filename = cfg.model_name

    model.summary(print_fn=logger.info)

    model_path = MODELS_DIR / model_filename
    callbacks = _build_callbacks(model_path, patience=overrides.get("patience", 10))
    callbacks.append(HardNegativeMiningCallback(train_seq, warmup_epochs=5, refresh_every=3))

    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=cfg.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Always persist the final weights too (best is already on disk via checkpoint)
    final_path = MODELS_DIR / f"final_{model_filename}"
    model.save(final_path)
    logger.info("Saved final model to %s", final_path)
    logger.info("Best model (val_loss) kept at %s", model_path)

    # ------------------------------------------------------------------
    # Auto-compute EER threshold on the validation set using sklearn ROC
    # ------------------------------------------------------------------
    logger.info("Computing EER threshold on validation set …")
    val_pairs = build_pair_index(
        genuine,
        forged,
        writer_ids=val_ids,
        pairs_per_writer=max(8, cfg.pairs_per_writer),
        seed=cfg.seed + 77,
    )
    val_xa, val_xb, val_y = materialise_pairs(val_pairs)
    val_distances = model.predict(
        [val_xa, val_xb], batch_size=cfg.batch_size, verbose=0
    ).ravel()
    eer_threshold, eer_value = compute_eer_threshold(val_distances, val_y)
    cfg.decision_threshold = eer_threshold
    CONFIG.decision_threshold = eer_threshold
    logger.info(
        "EER threshold = %.4f  (EER = %.1f%%)",
        eer_threshold,
        eer_value * 100.0,
    )

    history_path = RESULTS_DIR / "training_history.png"
    _plot_history(history.history, history_path)

    save_json(
        {
            "config": json.loads(cfg.to_json()),
            "writer_splits": {
                "train": train_ids,
                "val": val_ids,
                "test": test_ids,
            },
            "model_path": str(model_path),
            "final_model_path": str(final_path),
            "eer_threshold": eer_threshold,
            "eer_value": eer_value,
            "history": {k: [float(x) for x in v] for k, v in history.history.items()},
        },
        RESULTS_DIR / "training_summary.json",
    )

    return {
        "model_path": str(model_path),
        "final_model_path": str(final_path),
        "history_path": str(history_path),
        "writer_splits": {"train": train_ids, "val": val_ids, "test": test_ids},
        "eer_threshold": eer_threshold,
        "eer_value": eer_value,
        "history": history.history,
    }


# ---------------------------------------------------------------------------
# Module entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    overrides = dict(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        margin=args.margin,
        embedding_dim=args.embedding_dim,
        pairs_per_writer=args.pairs_per_writer,
        backbone=args.backbone,
        seed=args.seed,
        use_augmentation=not args.no_augment,
    )
    if args.output_name:
        if args.backbone == "mobilenetv2":
            overrides["mobilenet_model_name"] = args.output_name
        else:
            overrides["model_name"] = args.output_name

    train(
        patience=args.patience,
        trainable_base=args.fine_tune_mobilenet,
        **overrides,
    )


if __name__ == "__main__":
    main()
