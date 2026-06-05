"""
Flask web application exposing the Siamese signature verifier.

Routes
------
GET  /            -> Bootstrap form to upload two signatures.
POST /api/verify  -> JSON endpoint that returns the prediction.
POST /verify      -> HTML form handler (re-renders index with the result).
GET  /static/uploads/<file>  -> serves the uploaded image previews.
GET  /health      -> simple health-check endpoint.

The model is loaded once at startup. Uploaded images live in
``app/static/uploads/`` and are kept only briefly (the folder is created at
boot; old files older than 1 hour are pruned on each request).
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path
from typing import Tuple

# Allow `python app/app.py` to find the `src` package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import (  # noqa: E402
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from werkzeug.utils import secure_filename  # noqa: E402

from src.predict import SignatureVerifier  # noqa: E402
from src.utils import (  # noqa: E402
    CONFIG,
    MODELS_DIR,
    UPLOADS_DIR,
    ensure_directories,
    get_logger,
    load_eer_threshold,
)

logger = get_logger("signet.web")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}
MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB per upload
UPLOAD_TTL_SECONDS = 60 * 60  # 1 hour


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _save_upload(file_storage, prefix: str) -> Path:
    """Save an uploaded file safely under uploads/ and return its full path."""
    if not file_storage or not file_storage.filename:
        raise ValueError("No file provided.")
    if not _allowed_file(file_storage.filename):
        raise ValueError(
            f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
        )

    name = secure_filename(file_storage.filename)
    ext = name.rsplit(".", 1)[1].lower()
    unique = f"{prefix}_{uuid.uuid4().hex[:10]}.{ext}"
    target = UPLOADS_DIR / unique
    file_storage.save(target)
    return target


def _prune_old_uploads() -> None:
    """Best-effort cleanup of stale uploads."""
    cutoff = time.time() - UPLOAD_TTL_SECONDS
    if not UPLOADS_DIR.exists():
        return
    for path in UPLOADS_DIR.iterdir():
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
        except OSError:
            # Non-fatal — keep serving requests even if a file is locked.
            pass


def _relative_static_url(absolute_path: Path) -> str:
    """Turn an absolute file path under static/uploads into a /static URL."""
    rel = absolute_path.relative_to(PROJECT_ROOT / "app" / "static")
    return url_for("static", filename=rel.as_posix())


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(model_path: str | None = None) -> Flask:
    ensure_directories()

    app = Flask(
        __name__,
        static_folder=str(PROJECT_ROOT / "app" / "static"),
        template_folder=str(PROJECT_ROOT / "app" / "templates"),
    )
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
    app.config["SECRET_KEY"] = os.environ.get(
        "SIGNET_SECRET_KEY", uuid.uuid4().hex
    )

    # Resolve the model path once.
    resolved_model_path = Path(model_path or os.environ.get(
        "SIGNET_MODEL_PATH", str(MODELS_DIR / CONFIG.model_name)
    ))

    verifier: SignatureVerifier | None = None
    if resolved_model_path.exists():
        try:
            verifier = SignatureVerifier(resolved_model_path)
            logger.info("Model loaded — web app is ready.")
        except Exception as exc:  # pragma: no cover
            logger.exception("Failed to load the model: %s", exc)
    else:
        logger.warning(
            "Model file %s not found. Train the model first; the web app "
            "will start but predictions will return an error.",
            resolved_model_path,
        )

    # Load the EER threshold saved by the last training run.
    _threshold = load_eer_threshold()

    # -- routes ------------------------------------------------------------

    @app.get("/")
    def index():
        _prune_old_uploads()
        return render_template(
            "index.html",
            result=None,
            image_a_url=None,
            image_b_url=None,
            model_ready=verifier is not None,
            threshold=_threshold,
        )

    @app.post("/verify")
    def verify():
        if verifier is None:
            flash("Model is not available. Please train the model first.", "danger")
            return redirect(url_for("index"))

        try:
            path_a = _save_upload(request.files.get("signature_a"), "a")
            path_b = _save_upload(request.files.get("signature_b"), "b")
        except ValueError as exc:
            flash(str(exc), "warning")
            return redirect(url_for("index"))

        try:
            result = verifier.predict(path_a, path_b)
        except Exception as exc:
            logger.exception("Prediction failed: %s", exc)
            flash("Prediction failed. Check the server logs.", "danger")
            return redirect(url_for("index"))

        return render_template(
            "index.html",
            result=result,
            image_a_url=_relative_static_url(path_a),
            image_b_url=_relative_static_url(path_b),
            model_ready=True,
            threshold=verifier.threshold,
        )

    @app.post("/api/verify")
    def api_verify():
        if verifier is None:
            return jsonify({"error": "Model not loaded."}), 503
        try:
            path_a = _save_upload(request.files.get("signature_a"), "a")
            path_b = _save_upload(request.files.get("signature_b"), "b")
            result = verifier.predict(path_a, path_b)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:  # pragma: no cover
            logger.exception("API prediction failed")
            return jsonify({"error": "Internal prediction error."}), 500
        return jsonify(result.to_dict())

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "model_loaded": verifier is not None})

    @app.errorhandler(413)
    def too_large(_):
        return (
            jsonify(
                {"error": f"File too large. Max size is {MAX_CONTENT_LENGTH // 1024 // 1024} MB."}
            ),
            413,
        )

    return app


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the Signature Verifier Flask app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--model-path", type=str, default=None)
    args = parser.parse_args()

    app = create_app(model_path=args.model_path)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
