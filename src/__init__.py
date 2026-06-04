"""Signature Fraud Detection — Siamese Neural Network package.

This package uses Keras 3 with the PyTorch backend so training runs on the
local NVIDIA GPU on Windows (TensorFlow >= 2.11 has no native Windows GPU
support, but PyTorch does). The backend is selected here, before any Keras
submodule is imported.
"""

from __future__ import annotations

import os

# Honour an explicit override but otherwise default to torch so we hit the GPU.
os.environ.setdefault("KERAS_BACKEND", "torch")

__version__ = "1.0.0"
