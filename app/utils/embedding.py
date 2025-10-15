"""Utilities for turning images into vector embeddings."""

from __future__ import annotations

import logging
from typing import Iterable, List

import numpy as np
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Lightweight feature extractor backed by MobileNetV2."""

    def __init__(self) -> None:
        try:
            self.model = MobileNetV2(
                weights="imagenet",
                include_top=False,
                pooling="avg",
                input_shape=(224, 224, 3),
            )
            self.model.trainable = False
            logger.info("MobileNetV2 weights loaded for embedding extraction")
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to initialize MobileNetV2: %s", exc)
            raise

    @staticmethod
    def _prepare(image: Image.Image) -> np.ndarray:
        """Prepare an image for inference."""
        if image.mode != "RGB":
            image = image.convert("RGB")

        # MobileNet expects 224x224 RGB inputs
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        array = img_to_array(image)
        batch = np.expand_dims(array, axis=0)
        return preprocess_input(batch)

    def embed(self, image: Image.Image) -> np.ndarray:
        """Return a single L2-normalised embedding for the provided image."""
        processed = self._prepare(image)
        vector = self.model.predict(processed, verbose=0)[0]
        return self._normalise(vector)

    def embed_many(self, images: Iterable[Image.Image]) -> np.ndarray:
        """Return embeddings for a collection of images."""
        processed_batches: List[np.ndarray] = []
        for image in images:
            processed_batches.append(self._prepare(image)[0])

        if not processed_batches:
            return np.empty((0, 1280), dtype=np.float32)

        batch = np.stack(processed_batches, axis=0)
        vectors = self.model.predict(batch, verbose=0)
        return self._normalise(vectors)

    @staticmethod
    def _normalise(vectors: np.ndarray) -> np.ndarray:
        """Apply L2 normalisation to one or more vectors."""
        vectors = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms
