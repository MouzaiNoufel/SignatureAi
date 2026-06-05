# Signature Fraud Detection — Siamese Neural Network

> End-to-end deep-learning system that decides whether two handwritten signature images belong to the **same person** (`GENUINE`) or were **forged** (`FORGED`).  
> Trained on the CEDAR signature dataset — **85.6 % accuracy · AUC 0.943 · EER 14.4 %**.

---

## Table of Contents

1. [What We Built](#what-we-built)
2. [Architecture](#architecture)
3. [Dataset](#dataset)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [Usage — CLI](#usage--cli)
7. [Training Results](#training-results)
8. [Web App and REST API](#web-app-and-rest-api)
9. [Technical Notes — GPU / Keras Backend](#technical-notes--gpu--keras-backend)
10. [Future Improvements](#future-improvements)

---

## What We Built

Starting from a raw CEDAR-style signature image dump (55 writers, 24 genuine + 24 forged each), we designed and trained a complete **Siamese Neural Network** verification pipeline from scratch on a Windows machine with an **NVIDIA RTX 3070 Ti**.

### Everything that was implemented

| Component | Description |
|-----------|-------------|
| **`src/utils.py`** | `Config` dataclass (all hyper-parameters in one place), automatic dataset-path resolver, writer-disjoint split generator (39 / 8 / 8), reproducible-seed helper, structured logger |
| **`src/preprocess.py`** | OpenCV pipeline: BGR to RGB, resize to 155x220, float32 [0,1]; signature-specific augmentation (small rotation, translation, brightness jitter, Gaussian noise) |
| **`src/pair_generator.py`** | `build_pair_index()` for static evaluation pairs; `SignaturePairSequence` streaming balanced genuine/forged batches as a Keras 3 `PyDataset` (no full materialisation in RAM) |
| **`src/model.py`** | Siamese model builder using **Keras 3 pure API** (`keras.ops.*`): two shared-weight 4-block CNN branches, L2-normalised 128-dim embeddings, Euclidean distance, contrastive loss; optional MobileNetV2 transfer-learning variant |
| **`src/train.py`** | Full training loop with `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`, `TerminateOnNaN`; saves history plot + JSON summary |
| **`src/evaluate.py`** | Accuracy, Precision, Recall, F1, AUC-ROC, FAR/FRR sweep, EER computation; saves confusion matrix, ROC curve, FAR/FRR curve, `eval_summary.json` |
| **`src/predict.py`** | `SignatureVerifier` class for programmatic use; CLI for single-pair prediction |
| **`app/app.py`** | Flask 3 web app: Bootstrap 5 drag-and-drop HTML UI + `/api/verify` JSON endpoint + `/health`; auto-prunes old uploads |
| **`main.py`** | Unified CLI: `train` / `evaluate` / `predict` / `serve` / `all` |

### Key engineering decisions

- **Keras 3 + PyTorch backend** — TensorFlow >=2.11 dropped native Windows GPU support. We switched to Keras 3 multi-backend with `KERAS_BACKEND=torch` so the RTX 3070 Ti could be used. Training went from ~5 s/step (CPU) to **~500 ms/step (GPU)** — roughly **10x faster**.
- **Writer-disjoint splits** — No writer seen during training appears in validation or test, ensuring the metrics reflect true one-shot generalisation.
- **EER-optimal threshold** — The decision threshold of **0.264** was derived analytically as the point where FAR = FRR on the validation set, rather than using an arbitrary 0.5.
- **Contrastive loss** with margin = 1.0 and L2-normalised embeddings, so distances stay on the unit sphere and are directly interpretable.

---

## Architecture

```
Input A (155x220x3)          Input B (155x220x3)
        |                              |
        +----------+  +----------------+
                   v  v
           +-------------------------+
           |      Shared CNN         |  (weights tied between both branches)
           |                         |
           |  Conv2D(32) + BN + ReLU |  --- MaxPool 2x2
           |  Conv2D(64) + BN + ReLU |  --- MaxPool 2x2
           |  Conv2D(128)+ BN + ReLU |  --- MaxPool 2x2
           |  Conv2D(256)+ BN + ReLU |  --- GlobalAvgPool
           |  Dense(128)             |
           |  L2 Normalise           |  -> 128-dim unit embedding
           +------------+------------+
                        |
           +------------+------------+
           |   Euclidean distance    |  d = sqrt( sum( (ai - bi)^2 ) )
           +-------------------------+
                        |
                 Contrastive Loss
           L = y * d^2 + (1-y) * max(0, 1.0 - d)^2

    d < 0.264  ->  GENUINE
    d >= 0.264 ->  FORGED
```

Two backbones available:

| Flag | Backbone | Notes |
|------|----------|-------|
| `--backbone custom` (default) | Lightweight 4-block CNN | Fast, trains from scratch |
| `--backbone mobilenetv2` | ImageNet-pretrained MobileNetV2 | Better for small datasets |

---

## Dataset

**CEDAR Signature Dataset** placed at `../signature dataset/signatures/` (resolved automatically at runtime).

```
signatures/
  full_org/    original_{writer}_{sample}.png   <- 1 320 genuine images (24 x 55 writers)
  full_forg/   forgeries_{writer}_{sample}.png  <- 1 320 forged images  (24 x 55 writers)
```

- **55 writers**, IDs 1-55, each with 24 genuine + 24 skilled forgeries.
- **Writer-disjoint splits** (seed = 42):
  - Train: 39 writers
  - Validation: 8 writers
  - Test: 8 writers — [12, 17, 21, 24, 31, 50, 51, 54]

The path resolver checks in order:
1. `data/raw/` inside the project folder
2. `$SIGNATURE_DATA_DIR` environment variable
3. `../signature dataset/signatures/` (original location next to the project)

---

## Project Structure

```
signature_fraud_detection/
+-- main.py                  # CLI orchestrator (train / evaluate / predict / serve / all)
+-- requirements.txt
+-- README.md
|
+-- src/
|   +-- __init__.py          # Sets KERAS_BACKEND=torch before any keras import
|   +-- utils.py             # Config dataclass, paths, dataset indexer, logger, splits
|   +-- preprocess.py        # OpenCV: BGR->RGB, resize 155x220, [0,1] float32 + augmentation
|   +-- pair_generator.py    # Pair index builder + SignaturePairSequence (Keras PyDataset)
|   +-- model.py             # Siamese model, contrastive loss, Euclidean distance
|   +-- train.py             # Training loop, callbacks, history plot, JSON summary
|   +-- evaluate.py          # Metrics + ROC / confusion matrix / FAR-FRR plots
|   +-- predict.py           # SignatureVerifier class + CLI inference
|
+-- app/
|   +-- app.py               # Flask 3 web app (REST + HTML)
|   +-- templates/
|       +-- index.html       # Bootstrap 5 drag-and-drop UI
|
+-- models/                  # Saved model weights -- excluded from git (*.h5)
|   +-- siamese_model.h5     # Best checkpoint (epoch 35, val_loss=0.1467)
|   +-- final_siamese_model.h5
|
+-- results/                 # Generated plots & JSON summaries -- excluded from git
    +-- training_history.png
    +-- confusion_matrix.png
    +-- roc_curve.png
    +-- far_frr_curve.png
    +-- eval_summary.json
    +-- training_summary.json
```

---

## Installation

### Prerequisites

- Python 3.10-3.12
- NVIDIA GPU with CUDA 12.x recommended (tested: RTX 3070 Ti, CUDA 12.1)
- PyTorch with CUDA (install **before** the other requirements):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Steps

```bash
git clone https://github.com/MouzaiNoufel/SignatureAi.git
cd SignatureAi/signature_fraud_detection
pip install -r requirements.txt
```

Place the CEDAR dataset at `../signature dataset/signatures/` or set `SIGNATURE_DATA_DIR` to its absolute path.

---

## Usage — CLI

All commands are run from inside `signature_fraud_detection/`.

### Train

```bash
python main.py train
# Key options:
#   --backbone    custom | mobilenetv2   (default: custom)
#   --epochs      int                    (default: 77, EarlyStopping usually exits earlier)
#   --batch-size  int                    (default: 32)
#   --lr          float                  (default: 0.0001)
```

### Evaluate

```bash
python main.py evaluate --use-saved-split
```

### Predict a single pair

```bash
python main.py predict path/to/signature_a.png path/to/signature_b.png
```

Example output:

```
==================================================
  Similarity Score : 38.55%
  Distance         : 0.6145
  Threshold        : 0.264
  Prediction       : FORGED
==================================================
```

### Start the Flask web server

```bash
python main.py serve --host 0.0.0.0 --port 5000
# Then open http://localhost:5000
```

### Run full pipeline (train -> evaluate -> serve)

```bash
python main.py all
```

---

## Training Results

Training ran for **45 epochs** (EarlyStopping patience = 10) on an **NVIDIA RTX 3070 Ti** using the **Keras 3 / PyTorch backend**.

### Performance metrics (test set — 8 unseen writers)

| Metric | Value |
|--------|-------|
| **Accuracy** | **85.6 %** |
| Precision | 0.856 |
| Recall | 0.856 |
| F1-Score | 0.856 |
| **AUC-ROC** | **0.943** |
| **EER** | **14.4 %** |
| FAR @ EER threshold | 14.4 % |
| FRR @ EER threshold | 14.4 % |
| **Decision threshold** | **0.264** |

### Training details

| Detail | Value |
|--------|-------|
| Best epoch | 35 |
| Best val_loss | 0.1467 |
| Optimizer | Adam (lr = 1e-4) |
| Loss | Contrastive (margin = 1.0) |
| GPU speed | ~500 ms / step |
| CPU speed (before GPU fix) | ~5 000 ms / step |
| Total training time | ~11 minutes |

### Generated artefacts

| File | Description |
|------|-------------|
| `results/training_history.png` | Train / val loss curves over 45 epochs |
| `results/confusion_matrix.png` | Confusion matrix on the 8-writer test set |
| `results/roc_curve.png` | ROC curve — AUC 0.943 |
| `results/far_frr_curve.png` | FAR / FRR sweep showing EER crossover at threshold 0.264 |
| `results/eval_summary.json` | All scalar metrics as JSON |
| `results/training_summary.json` | Full per-epoch history + hyper-parameters |

> **Note:** Model weights (`models/*.h5`, ~15 MB each) and result images are excluded from git via `.gitignore`.
> Re-run `python main.py train` then `python main.py evaluate --use-saved-split` to reproduce them.

---

## Web App and REST API

```bash
python main.py serve --host 0.0.0.0 --port 5000
```

Opens a **Bootstrap 5** UI at `http://localhost:5000` with:
- Two drag-and-drop upload tiles (reference and query signature)
- Animated similarity circle showing the score in percent
- Colour-coded verdict card: green GENUINE / red FORGED
- Distance, threshold and raw score displayed

### POST /api/verify

**Request** — `multipart/form-data`:

| Field | Type | Description |
|-------|------|-------------|
| `signature_a` | file | Reference signature (PNG / JPG) |
| `signature_b` | file | Signature to verify |

**Response** — JSON:

```json
{
  "distance":   0.6145,
  "similarity": 38.55,
  "threshold":  0.264,
  "is_genuine": false,
  "label":      "FORGED"
}
```

### GET /health

```json
{"status": "ok", "model_loaded": true}
```

---

## Technical Notes — GPU / Keras Backend

TensorFlow >=2.11 dropped native GPU support on Windows. To use an NVIDIA GPU on Windows the project uses **Keras 3 multi-backend** with the **PyTorch backend**:

```python
# src/__init__.py  -- executed before any keras import
import os
os.environ.setdefault("KERAS_BACKEND", "torch")
```

This is already configured — no manual step required. You can also override at the shell level:

```powershell
$env:KERAS_BACKEND = "torch"
python main.py train
```

**Tested environment:**

| Component | Version |
|-----------|---------|
| OS | Windows 11 |
| Python | 3.12.10 |
| PyTorch | 2.5.1+cu121 |
| Keras | 3.13.2 |
| TensorFlow | 2.20.0 |
| protobuf | 5.29.6 |
| GPU | NVIDIA RTX 3070 Ti |
| CUDA | 12.1 |

---

## Future Improvements

- Replace contrastive loss with **triplet loss** + online semi-hard mining for better embedding separation.
- Add **Grad-CAM / saliency overlays** in the web UI to highlight which stroke regions triggered the forged decision.
- Calibrate the similarity score with **isotonic regression** for well-behaved probabilities.
- Store embeddings in **FAISS** to support 1-vs-N enrolment lookups.
- Distil the network into a **TFLite / ONNX** model for edge deployment.
- Extend to **Arabic and Devanagari** scripts with domain-adaptive fine-tuning.

---

## License

MIT
