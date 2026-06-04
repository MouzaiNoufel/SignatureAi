"""
Evaluation pipeline for the trained Siamese model.

Reports
-------
* Accuracy, Precision, Recall, F1-score
* ROC curve + AUC
* Confusion matrix
* FAR (False Acceptance Rate) and FRR (False Rejection Rate) — both at the
  configured operating threshold AND a sweep saved alongside the plots.
* Optimal threshold (EER) saved in ``results/eval_summary.json``.

Plots are written to ``results/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Make sure ``KERAS_BACKEND`` is set before keras is imported.
from . import __init__  # noqa: F401  (side-effect import)

import keras
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

from .model import get_custom_objects
from .pair_generator import build_pair_index, materialise_pairs
from .utils import (
    CONFIG,
    MODELS_DIR,
    RESULTS_DIR,
    ensure_directories,
    get_logger,
    index_signature_dataset,
    load_json,
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
    parser = argparse.ArgumentParser(description="Evaluate a trained Siamese model.")
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(MODELS_DIR / CONFIG.model_name),
        help="Path to the .h5/.keras model to evaluate.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Distance threshold (default: CONFIG.decision_threshold or EER).",
    )
    parser.add_argument(
        "--pairs-per-writer",
        type=int,
        default=20,
        help="Density of evaluation pairs per writer.",
    )
    parser.add_argument(
        "--use-saved-split",
        action="store_true",
        help="Re-use the writer split saved by train.py (results/training_summary.json).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_confusion_matrix(cm: np.ndarray, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax)

    classes = ("Forged (0)", "Genuine (1)")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def _plot_roc(fpr: np.ndarray, tpr: np.ndarray, auc_value: float, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc_value:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", alpha=0.7)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def _plot_far_frr(
    thresholds: np.ndarray,
    far: np.ndarray,
    frr: np.ndarray,
    eer_threshold: float,
    eer_value: float,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thresholds, far, label="FAR (False Acceptance)")
    ax.plot(thresholds, frr, label="FRR (False Rejection)")
    ax.axvline(eer_threshold, color="red", linestyle="--", alpha=0.7, label=f"EER={eer_value:.3f}")
    ax.set_xlabel("Distance Threshold")
    ax.set_ylabel("Rate")
    ax.set_title("FAR / FRR vs Threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def compute_far_frr(
    distances: np.ndarray, labels: np.ndarray, num_thresholds: int = 200
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Compute FAR/FRR over a threshold sweep and locate the EER point.

    A pair is *accepted* (predicted genuine) when ``distance < threshold``.

    Returns:
        thresholds, far_array, frr_array, eer_threshold, eer_value
    """
    if num_thresholds < 5:
        raise ValueError("num_thresholds too small.")

    lo = float(np.min(distances))
    hi = float(np.max(distances))
    if hi <= lo:
        hi = lo + 1e-3
    thresholds = np.linspace(lo, hi, num_thresholds)

    genuine_mask = labels == 1
    forged_mask = labels == 0

    far = np.zeros_like(thresholds)
    frr = np.zeros_like(thresholds)
    for i, t in enumerate(thresholds):
        accepted = distances < t
        # FAR = forged accepted / forged total
        if forged_mask.any():
            far[i] = float(np.mean(accepted[forged_mask]))
        # FRR = genuine rejected / genuine total
        if genuine_mask.any():
            frr[i] = float(np.mean(~accepted[genuine_mask]))

    diff = np.abs(far - frr)
    idx = int(np.argmin(diff))
    eer_threshold = float(thresholds[idx])
    eer_value = float((far[idx] + frr[idx]) / 2.0)
    return thresholds, far, frr, eer_threshold, eer_value


# ---------------------------------------------------------------------------
# Public evaluation routine
# ---------------------------------------------------------------------------


def evaluate(
    model_path: str | Path | None = None,
    threshold: float | None = None,
    pairs_per_writer: int = 20,
    use_saved_split: bool = False,
) -> Dict[str, object]:
    """Run evaluation and persist all plots + JSON summary."""
    ensure_directories()
    set_global_seed(CONFIG.seed)

    model_path = Path(model_path or (MODELS_DIR / CONFIG.model_name))
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    logger.info("Loading model from %s", model_path)

    model = keras.models.load_model(
        model_path,
        custom_objects=get_custom_objects(),
        compile=False,
    )

    dataset_dir = resolve_dataset_dir()
    genuine, forged = index_signature_dataset(dataset_dir)
    writer_ids = sorted(set(genuine) & set(forged))

    if use_saved_split:
        summary_path = RESULTS_DIR / "training_summary.json"
        if summary_path.exists():
            saved = load_json(summary_path)
            test_ids = saved.get("writer_splits", {}).get("test", [])
            if not test_ids:
                logger.warning("No test split in saved summary, falling back.")
        else:
            test_ids = []
    else:
        test_ids = []

    if not test_ids:
        _, _, test_ids = split_writers(
            writer_ids, CONFIG.val_split, CONFIG.test_split, CONFIG.seed
        )

    logger.info("Evaluating on %d held-out writers.", len(test_ids))

    pairs = build_pair_index(
        genuine,
        forged,
        writer_ids=test_ids,
        pairs_per_writer=pairs_per_writer,
        seed=CONFIG.seed + 99,
    )
    xa, xb, y_true = materialise_pairs(pairs)
    logger.info("Materialised %d evaluation pairs.", len(y_true))

    distances = model.predict([xa, xb], batch_size=CONFIG.batch_size, verbose=1).ravel()

    # ---- FAR / FRR & EER --------------------------------------------------
    thresholds, far_arr, frr_arr, eer_threshold, eer_value = compute_far_frr(
        distances, y_true
    )

    if threshold is None:
        threshold = float(CONFIG.decision_threshold)
    logger.info("Operating threshold: %.4f (EER threshold: %.4f)", threshold, eer_threshold)

    y_pred = (distances < threshold).astype(int)

    # ---- classical metrics -----------------------------------------------
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # ROC works on "similarity" scores -> use negated distance
    scores = -distances
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc_value = auc(fpr, tpr)

    # FAR / FRR at the operating threshold
    accepted = distances < threshold
    far_at_thr = float(np.mean(accepted[y_true == 0])) if np.any(y_true == 0) else 0.0
    frr_at_thr = float(np.mean(~accepted[y_true == 1])) if np.any(y_true == 1) else 0.0

    # ---- write plots ------------------------------------------------------
    _plot_confusion_matrix(cm, RESULTS_DIR / "confusion_matrix.png")
    _plot_roc(fpr, tpr, auc_value, RESULTS_DIR / "roc_curve.png")
    _plot_far_frr(
        thresholds, far_arr, frr_arr, eer_threshold, eer_value,
        RESULTS_DIR / "far_frr_curve.png",
    )

    # ---- summary ----------------------------------------------------------
    summary: Dict[str, object] = {
        "model_path": str(model_path),
        "n_pairs": int(len(y_true)),
        "threshold": float(threshold),
        "eer_threshold": eer_threshold,
        "eer": eer_value,
        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "auc": float(auc_value),
            "far": far_at_thr,
            "frr": frr_at_thr,
        },
        "confusion_matrix": cm.tolist(),
    }
    save_json(summary, RESULTS_DIR / "eval_summary.json")

    logger.info("=" * 60)
    logger.info("Accuracy : %.4f", accuracy)
    logger.info("Precision: %.4f", precision)
    logger.info("Recall   : %.4f", recall)
    logger.info("F1-score : %.4f", f1)
    logger.info("AUC      : %.4f", auc_value)
    logger.info("FAR @ th : %.4f", far_at_thr)
    logger.info("FRR @ th : %.4f", frr_at_thr)
    logger.info("EER      : %.4f  (threshold %.4f)", eer_value, eer_threshold)
    logger.info("=" * 60)
    logger.info("Plots & summary written to %s", RESULTS_DIR)

    return summary


# ---------------------------------------------------------------------------
# Module entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    evaluate(
        model_path=args.model_path,
        threshold=args.threshold,
        pairs_per_writer=args.pairs_per_writer,
        use_saved_split=args.use_saved_split,
    )


if __name__ == "__main__":
    main()
