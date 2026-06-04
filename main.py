"""
Top-level orchestrator for the Signature Fraud Detection project.

Examples
--------

Full pipeline (custom CNN backbone, 77 epochs as per the spec) ::

    python main.py all

Individual phases ::

    python main.py train --backbone mobilenetv2 --epochs 30
    python main.py evaluate --use-saved-split
    python main.py predict path/to/sig1.png path/to/sig2.png
    python main.py serve --host 0.0.0.0 --port 8000

This script wires the ``src.*`` modules together so a fresh checkout can be
trained, evaluated and served end-to-end with a single command.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Make sure `src` and `app` resolve when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _cmd_train(args: argparse.Namespace) -> int:
    from src import train as train_mod

    forwarded = ["--epochs", str(args.epochs), "--batch-size", str(args.batch_size)]
    if args.backbone:
        forwarded += ["--backbone", args.backbone]
    if args.no_augment:
        forwarded += ["--no-augment"]
    if args.fine_tune_mobilenet:
        forwarded += ["--fine-tune-mobilenet"]
    if args.patience is not None:
        forwarded += ["--patience", str(args.patience)]
    train_mod.main(forwarded)
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from src import evaluate as eval_mod

    forwarded: list[str] = []
    if args.model_path:
        forwarded += ["--model-path", args.model_path]
    if args.threshold is not None:
        forwarded += ["--threshold", str(args.threshold)]
    if args.pairs_per_writer is not None:
        forwarded += ["--pairs-per-writer", str(args.pairs_per_writer)]
    if args.use_saved_split:
        forwarded += ["--use-saved-split"]
    eval_mod.main(forwarded)
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    from src import predict as predict_mod

    forwarded = [args.image_a, args.image_b]
    if args.model_path:
        forwarded += ["--model-path", args.model_path]
    if args.threshold is not None:
        forwarded += ["--threshold", str(args.threshold)]
    predict_mod.main(forwarded)
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from app.app import create_app

    app = create_app(model_path=args.model_path)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


def _cmd_all(args: argparse.Namespace) -> int:
    _cmd_train(args)
    args.model_path = None  # use freshly trained model
    args.threshold = None
    args.pairs_per_writer = 20
    args.use_saved_split = True
    _cmd_evaluate(args)
    return 0


# ---------------------------------------------------------------------------
# CLI assembly
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="signet",
        description="Signature Fraud Detection — orchestrator CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- train ---------------------------------------------------------
    p_train = subparsers.add_parser("train", help="Train the Siamese model.")
    p_train.add_argument("--epochs", type=int, default=77)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--backbone", choices=("custom", "mobilenetv2"), default="custom")
    p_train.add_argument("--no-augment", action="store_true")
    p_train.add_argument("--fine-tune-mobilenet", action="store_true")
    p_train.add_argument("--patience", type=int, default=10)
    p_train.set_defaults(func=_cmd_train)

    # ---- evaluate -------------------------------------------------------
    p_eval = subparsers.add_parser("evaluate", help="Evaluate a trained model.")
    p_eval.add_argument("--model-path", type=str, default=None)
    p_eval.add_argument("--threshold", type=float, default=None)
    p_eval.add_argument("--pairs-per-writer", type=int, default=20)
    p_eval.add_argument("--use-saved-split", action="store_true")
    p_eval.set_defaults(func=_cmd_evaluate)

    # ---- predict --------------------------------------------------------
    p_pred = subparsers.add_parser("predict", help="Compare two signature images.")
    p_pred.add_argument("image_a", type=str)
    p_pred.add_argument("image_b", type=str)
    p_pred.add_argument("--model-path", type=str, default=None)
    p_pred.add_argument("--threshold", type=float, default=None)
    p_pred.set_defaults(func=_cmd_predict)

    # ---- serve ----------------------------------------------------------
    p_serve = subparsers.add_parser("serve", help="Run the Flask web app.")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=5000)
    p_serve.add_argument("--debug", action="store_true")
    p_serve.add_argument("--model-path", type=str, default=None)
    p_serve.set_defaults(func=_cmd_serve)

    # ---- all (train + evaluate) ---------------------------------------
    p_all = subparsers.add_parser(
        "all", help="Run the full pipeline: train then evaluate."
    )
    p_all.add_argument("--epochs", type=int, default=77)
    p_all.add_argument("--batch-size", type=int, default=32)
    p_all.add_argument("--backbone", choices=("custom", "mobilenetv2"), default="custom")
    p_all.add_argument("--no-augment", action="store_true")
    p_all.add_argument("--fine-tune-mobilenet", action="store_true")
    p_all.add_argument("--patience", type=int, default=10)
    p_all.set_defaults(func=_cmd_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
