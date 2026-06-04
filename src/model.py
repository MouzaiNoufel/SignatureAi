"""
Siamese network architecture, distance layer, contrastive loss and a
MobileNetV2-based transfer-learning variant.

Built on **Keras 3** with the backend selected in :mod:`src.__init__`
(``torch`` by default so we can hit the local NVIDIA GPU on Windows).

Public entry points:

* :func:`build_siamese_model`             — custom CNN backbone.
* :func:`build_mobilenetv2_siamese_model` — transfer-learning backbone.

Both return a compiled ``keras.Model`` that maps a pair of images to a
scalar Euclidean distance in embedding space.
"""

from __future__ import annotations

from typing import Tuple

# Make sure ``KERAS_BACKEND`` is set before keras is imported.
from . import __init__  # noqa: F401  (side-effect import)

import keras
from keras import backend as K
from keras import layers, models, ops, optimizers

from .utils import CONFIG


# ---------------------------------------------------------------------------
# Distance layer & loss
# ---------------------------------------------------------------------------


def euclidean_distance(vectors):
    """Element-wise Euclidean distance between two embedding tensors.

    Args:
        vectors: tuple/list of two tensors with shape ``(batch, dim)``.

    Returns:
        Tensor of shape ``(batch, 1)`` with non-negative distances.
    """
    x, y = vectors
    sum_squared = ops.sum(ops.square(x - y), axis=1, keepdims=True)
    return ops.sqrt(ops.maximum(sum_squared, K.epsilon()))


def euclidean_output_shape(input_shapes):
    """Static output-shape inference for the Lambda distance layer."""
    shape_a, _ = input_shapes
    return (shape_a[0], 1)


def contrastive_loss(margin: float = 1.0):
    """Hadsell-Chopra-LeCun contrastive loss.

    ``L = y * d^2 + (1 - y) * max(margin - d, 0)^2``

    Where ``y = 1`` for a *similar* (genuine) pair and ``y = 0`` for a
    *dissimilar* (forged / different writer) pair.
    """

    def loss(y_true, y_pred):
        y_true = ops.cast(y_true, y_pred.dtype)
        square_pred = ops.square(y_pred)
        margin_square = ops.square(ops.maximum(margin - y_pred, 0.0))
        return ops.mean(y_true * square_pred + (1.0 - y_true) * margin_square)

    loss.__name__ = "contrastive_loss"
    return loss


def accuracy_at_threshold(threshold: float = 0.5):
    """Binary accuracy treating ``distance < threshold`` as positive."""

    def metric(y_true, y_pred):
        y_true = ops.cast(y_true, y_pred.dtype)
        preds = ops.cast(y_pred < threshold, y_true.dtype)
        return ops.mean(ops.cast(ops.equal(preds, y_true), y_pred.dtype))

    metric.__name__ = "siamese_accuracy"
    return metric


# ---------------------------------------------------------------------------
# Backbones
# ---------------------------------------------------------------------------


def build_custom_backbone(
    input_shape: Tuple[int, int, int],
    embedding_dim: int = 128,
) -> models.Model:
    """Custom CNN feature extractor matching the project spec."""
    inputs = layers.Input(shape=input_shape, name="backbone_input")
    x = inputs

    block_filters = (32, 64, 128, 256)
    for i, filters in enumerate(block_filters, start=1):
        x = layers.Conv2D(
            filters,
            kernel_size=3,
            padding="same",
            activation="relu",
            kernel_initializer="he_normal",
            name=f"conv{i}_a",
        )(x)
        x = layers.BatchNormalization(name=f"bn{i}_a")(x)
        x = layers.Conv2D(
            filters,
            kernel_size=3,
            padding="same",
            activation="relu",
            kernel_initializer="he_normal",
            name=f"conv{i}_b",
        )(x)
        x = layers.BatchNormalization(name=f"bn{i}_b")(x)
        x = layers.MaxPooling2D(pool_size=2, name=f"pool{i}")(x)
        x = layers.Dropout(0.25, name=f"drop{i}")(x)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(
        embedding_dim * 2,
        activation="relu",
        kernel_initializer="he_normal",
        name="fc1",
    )(x)
    x = layers.Dropout(0.3, name="fc_drop")(x)
    x = layers.Dense(embedding_dim, activation=None, name="embedding")(x)
    # L2-normalised embedding -> distances live on the unit sphere.
    x = layers.UnitNormalization(axis=1, name="l2_normalize")(x)

    return models.Model(inputs=inputs, outputs=x, name="custom_backbone")


def build_mobilenetv2_backbone(
    input_shape: Tuple[int, int, int],
    embedding_dim: int = 128,
    trainable_base: bool = False,
) -> models.Model:
    """MobileNetV2 transfer-learning feature extractor."""
    base = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base.trainable = trainable_base

    inputs = layers.Input(shape=input_shape, name="mobilenet_input")
    # MobileNetV2 expects inputs in [-1, 1]; our pipeline yields [0, 1].
    x = layers.Rescaling(scale=2.0, offset=-1.0, name="mobilenet_rescale")(inputs)
    x = base(x, training=trainable_base)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3, name="drop")(x)
    x = layers.Dense(embedding_dim, activation=None, name="embedding")(x)
    x = layers.UnitNormalization(axis=1, name="l2_normalize")(x)

    return models.Model(inputs=inputs, outputs=x, name="mobilenetv2_backbone")


# ---------------------------------------------------------------------------
# Siamese head
# ---------------------------------------------------------------------------


def _wrap_siamese(backbone: models.Model, input_shape: Tuple[int, int, int]) -> models.Model:
    """Wire two shared-weight branches and a Euclidean distance head."""
    input_a = layers.Input(shape=input_shape, name="input_a")
    input_b = layers.Input(shape=input_shape, name="input_b")

    embedding_a = backbone(input_a)
    embedding_b = backbone(input_b)

    distance = layers.Lambda(
        euclidean_distance,
        output_shape=euclidean_output_shape,
        name="euclidean_distance",
    )([embedding_a, embedding_b])

    return models.Model(inputs=[input_a, input_b], outputs=distance, name="siamese_model")


def build_siamese_model(
    input_shape: Tuple[int, int, int] | None = None,
    embedding_dim: int | None = None,
    learning_rate: float | None = None,
    margin: float | None = None,
    decision_threshold: float | None = None,
) -> models.Model:
    """Build & compile the custom-CNN Siamese network."""
    input_shape = input_shape or CONFIG.input_shape
    embedding_dim = embedding_dim or CONFIG.embedding_dim
    learning_rate = learning_rate or CONFIG.learning_rate
    margin = margin if margin is not None else CONFIG.margin
    decision_threshold = (
        decision_threshold if decision_threshold is not None else CONFIG.decision_threshold
    )

    backbone = build_custom_backbone(input_shape, embedding_dim)
    model = _wrap_siamese(backbone, input_shape)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=contrastive_loss(margin=margin),
        metrics=[accuracy_at_threshold(decision_threshold)],
    )
    return model


def build_mobilenetv2_siamese_model(
    input_shape: Tuple[int, int, int] | None = None,
    embedding_dim: int | None = None,
    learning_rate: float | None = None,
    margin: float | None = None,
    decision_threshold: float | None = None,
    trainable_base: bool = False,
) -> models.Model:
    """Build & compile the MobileNetV2-backed Siamese network."""
    input_shape = input_shape or CONFIG.input_shape
    embedding_dim = embedding_dim or CONFIG.embedding_dim
    learning_rate = learning_rate or CONFIG.learning_rate
    margin = margin if margin is not None else CONFIG.margin
    decision_threshold = (
        decision_threshold if decision_threshold is not None else CONFIG.decision_threshold
    )

    backbone = build_mobilenetv2_backbone(input_shape, embedding_dim, trainable_base)
    model = _wrap_siamese(backbone, input_shape)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=contrastive_loss(margin=margin),
        metrics=[accuracy_at_threshold(decision_threshold)],
    )
    return model


# ---------------------------------------------------------------------------
# Custom-objects dict for model (de)serialisation
# ---------------------------------------------------------------------------


def get_custom_objects(margin: float | None = None, decision_threshold: float | None = None) -> dict:
    """Return the ``custom_objects`` mapping required to load saved models."""
    margin = margin if margin is not None else CONFIG.margin
    decision_threshold = (
        decision_threshold if decision_threshold is not None else CONFIG.decision_threshold
    )
    return {
        "euclidean_distance": euclidean_distance,
        "euclidean_output_shape": euclidean_output_shape,
        "contrastive_loss": contrastive_loss(margin=margin),
        "loss": contrastive_loss(margin=margin),
        "siamese_accuracy": accuracy_at_threshold(decision_threshold),
        "metric": accuracy_at_threshold(decision_threshold),
    }


__all__ = [
    "euclidean_distance",
    "euclidean_output_shape",
    "contrastive_loss",
    "accuracy_at_threshold",
    "build_custom_backbone",
    "build_mobilenetv2_backbone",
    "build_siamese_model",
    "build_mobilenetv2_siamese_model",
    "get_custom_objects",
]
