"""Utility script to rebuild product embeddings for the catalogue."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"
IMAGE_DIR = STATIC_DIR / "images"

sys.path.append(str(BASE_DIR))

from app.utils import EmbeddingModel, ImagePreprocessor  # noqa: E402

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("build_embeddings")


def load_catalogue() -> pd.DataFrame:
    candidates = [DATA_DIR / "products.csv", DATA_DIR / "products.xlsx"]
    for path in candidates:
        if path.exists():
            LOGGER.info("Reading catalogue from %s", path)
            if path.suffix == ".csv":
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)
            break
    else:
        raise FileNotFoundError("Catalogue not found. Add data/products.csv or data/products.xlsx")

    if "image_path" not in df:
        df["image_path"] = ""
    if "image_url" not in df:
        df["image_url"] = ""

    return df


def ensure_directories() -> None:
    for directory in (DATA_DIR, STATIC_DIR, IMAGE_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def ensure_image_file(entry: Dict) -> Path:
    filename = Path(str(entry.get("image_path", "")).strip()).name
    if not filename:
        filename = f"{entry.get('id', 'product')}.jpg"
    local_path = IMAGE_DIR / filename

    if local_path.exists():
        return local_path

    url = str(entry.get("image_url", "")).strip()
    if not url:
        raise FileNotFoundError(f"No image available for product {entry.get('id', 'unknown')}")

    LOGGER.info("Downloading %s -> %s", url, local_path)
    response = requests.get(url, timeout=15)
    response.raise_for_status()

    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(response.content)
    return local_path


def build_embeddings(limit: int | None = None) -> None:
    ensure_directories()
    catalogue = load_catalogue()
    model = EmbeddingModel()

    vectors: List[np.ndarray] = []
    valid_rows: List[Dict] = []

    for index, row in catalogue.iterrows():
        if limit is not None and len(vectors) >= limit:
            LOGGER.info("Limit of %s embeddings reached", limit)
            break

        record = row.to_dict()
        try:
            image_path = ensure_image_file(record)
            payload = image_path.read_bytes()
            image = ImagePreprocessor.load(payload)
            image = ImagePreprocessor.enhance(image)
            embedding = model.embed(image)
        except Exception as exc:
            LOGGER.warning("Skipping %s: %s", record.get("id", index), exc)
            continue

        vectors.append(embedding)
        record["image_path"] = f"images/{Path(image_path).name}"
        valid_rows.append(record)

    if not vectors:
        raise RuntimeError("No embeddings were generated. Check that images are reachable.")

    matrix = np.vstack(vectors).astype(np.float32)
    embeddings_path = DATA_DIR / "embeddings.npy"
    np.save(embeddings_path, matrix)
    LOGGER.info("Embeddings stored at %s", embeddings_path)

    valid_df = pd.DataFrame(valid_rows)
    output_path = DATA_DIR / "valid_products.xlsx"
    valid_df.to_excel(output_path, index=False)
    LOGGER.info("Valid catalogue exported to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild product embeddings")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of products to process (useful for quick checks)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    try:
        build_embeddings(limit=arguments.limit)
        LOGGER.info("Embedding build completed successfully")
    except KeyboardInterrupt:
        LOGGER.info("Aborted by user")
    except Exception as exc:  # pragma: no cover - CLI feedback
        LOGGER.exception("Embedding build failed: %s", exc)
        sys.exit(1)
