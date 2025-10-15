"""Image validation and light preprocessing helpers."""

from __future__ import annotations

import io
import logging
from typing import Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Collection of routines used before embedding extraction."""

    @staticmethod
    def is_image(data: bytes) -> bool:
        """Return ``True`` if the payload can be opened as an image."""
        try:
            candidate = Image.open(io.BytesIO(data))
            candidate.verify()
            return True
        except Exception:  # pragma: no cover - defensive validation
            return False

    @staticmethod
    def load(data: bytes, size: Tuple[int, int] = (224, 224)) -> Image.Image:
        """Load, orient, and resize an image while keeping aspect ratio."""
        try:
            image = Image.open(io.BytesIO(data))
            if image.mode != "RGB":
                image = image.convert("RGB")

            image = ImageOps.exif_transpose(image)
            image = ImagePreprocessor._scale_and_pad(image, size)
            return image
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to load image payload: %s", exc)
            raise

    @staticmethod
    def enhance(image: Image.Image, factor: float = 1.15) -> Image.Image:
        """Slightly enhance sharpness and contrast to aid the encoder."""
        sharp = ImageEnhance.Sharpness(image).enhance(factor)
        return ImageEnhance.Contrast(sharp).enhance(1.05)

    @staticmethod
    def as_numpy(image: Image.Image) -> np.ndarray:
        """Convert an image to a float32 numpy array in the [0, 1] range."""
        return np.asarray(image).astype("float32") / 255.0

    @staticmethod
    def _scale_and_pad(image: Image.Image, target: Tuple[int, int]) -> Image.Image:
        """Resize an image while preserving aspect ratio and centring it."""
        target_w, target_h = target
        width, height = image.size

        if width == 0 or height == 0:
            raise ValueError("Cannot resize an empty image")

        scale = min(target_w / width, target_h / height)
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        canvas = Image.new("RGB", target, (255, 255, 255))
        offset = ((target_w - new_w) // 2, (target_h - new_h) // 2)
        canvas.paste(resized, offset)
        return canvas
