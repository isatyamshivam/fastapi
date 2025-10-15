"""FastAPI application exposing visual product matching endpoints."""

from __future__ import annotations

import io
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiofiles
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .utils import EmbeddingModel, ImagePreprocessor, SimilarityIndex

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"
IMAGE_DIR = STATIC_DIR / "images"
LOG_DIR = BASE_DIR / "logs"

app = FastAPI(
    title="Visual Product Matcher API",
    version="1.0.0",
    description="Upload an image or pass a URL to discover visually similar catalogue items.",
)

# Get allowed origins from environment variable for production
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_embedding_model: Optional[EmbeddingModel] = None
_similarity_index: Optional[SimilarityIndex] = None
_catalogue: Optional[pd.DataFrame] = None


def _normalise_image_path(raw: str) -> str:
    """Map any image reference to ``images/<filename>``."""
    raw = (raw or "").strip()
    if not raw:
        return ""
    filename = Path(raw).name
    return str(Path("images") / filename)


def _catalogue_records(df: pd.DataFrame, request: Request | None = None) -> list[dict]:
    # Replace NaN values with None to avoid JSON serialization errors
    df_clean = df.fillna("")
    records = df_clean.to_dict(orient="records")
    return [_enrich_media(record, request) for record in records]


def _enrich_media(record: dict, request: Request | None = None) -> dict:
    record = dict(record)
    filename = _normalise_image_path(record.get("image_path", ""))
    record["image_path"] = filename
    if filename:
        static_relative = f"static/{filename}".replace("\\", "/")
        try:
            static_url = str(request.url_for("static", path=filename)) if request else f"/static/{filename}"
        except Exception:  # pragma: no cover - fallback when request context missing
            static_url = f"/static/{filename}"
        record["image_static_path"] = static_relative
        record["image_url_local"] = static_url.replace("\\", "/")
    else:
        record["image_static_path"] = ""
        record["image_url_local"] = ""
    return record


def _load_catalogue() -> pd.DataFrame:
    candidates = [DATA_DIR / "products.csv", DATA_DIR / "products.xlsx"]
    for path in candidates:
        if path.exists():
            LOGGER.info("Loading catalogue from %s", path)
            if path.suffix == ".csv":
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)
            break
    else:
        raise FileNotFoundError("No catalogue found in data/products.csv or data/products.xlsx")

    if "available" in df.columns:
        df["available"] = df["available"].astype(bool)
    if "price" in df.columns:
        df["price"] = df["price"].astype(float)

    if "category" in df.columns:
        df["category"] = df["category"].fillna("Uncategorised")
    else:
        df["category"] = "Uncategorised"

    if "image_path" in df.columns:
        image_series = df["image_path"].astype(str)
    else:
        image_series = pd.Series(["" for _ in range(len(df))])
    df["image_path"] = image_series.apply(_normalise_image_path)
    if "image_url" not in df.columns:
        df["image_url"] = ""

    return df


def _load_embeddings(df: pd.DataFrame) -> Optional[np.ndarray]:
    path = DATA_DIR / "embeddings.npy"
    if not path.exists():
        LOGGER.warning("Embeddings not found at %s. Run the build_embeddings script first.", path)
        return None

    LOGGER.info("Loading embeddings from %s", path)
    embeddings = np.load(path)
    if embeddings.shape[0] != len(df):
        LOGGER.warning(
            "Embeddings count (%s) does not match catalogue size (%s)",
            embeddings.shape[0],
            len(df),
        )
    return embeddings


def _ensure_directories() -> None:
    for directory in (STATIC_DIR, IMAGE_DIR, DATA_DIR, LOG_DIR):
        directory.mkdir(parents=True, exist_ok=True)


@app.on_event("startup")
async def startup() -> None:
    global _embedding_model, _similarity_index, _catalogue

    _ensure_directories()
    _embedding_model = EmbeddingModel()
    _catalogue = _load_catalogue()

    embeddings = _load_embeddings(_catalogue)
    if embeddings is not None:
        valid_path = DATA_DIR / "valid_products.xlsx"
        if valid_path.exists():
            try:
                LOGGER.info("Loading curated catalogue from %s", valid_path)
                valid_df = pd.read_excel(valid_path)
                if len(valid_df) == embeddings.shape[0]:
                    if "image_path" in valid_df.columns:
                        valid_images = valid_df["image_path"].astype(str)
                    else:
                        valid_images = pd.Series(["" for _ in range(len(valid_df))])
                    valid_df["image_path"] = valid_images.apply(_normalise_image_path)
                    if "image_url" not in valid_df.columns:
                        valid_df["image_url"] = ""
                    _catalogue = valid_df
                else:
                    LOGGER.warning(
                        "valid_products.xlsx count (%s) does not match embeddings (%s)",
                        len(valid_df),
                        embeddings.shape[0],
                    )
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning("Failed to load valid catalogue: %s", exc)
    if embeddings is not None:
        _similarity_index = SimilarityIndex(embeddings, _catalogue)
    else:
        _similarity_index = None

    LOGGER.info(
        "Startup completed. Model loaded: %s, embeddings loaded: %s",
        _embedding_model is not None,
        _similarity_index is not None,
    )


@app.get("/api/health")
def health() -> JSONResponse:
    payload = {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "model": _embedding_model is not None,
        "embeddings": _similarity_index is not None,
        "items": int(len(_catalogue) if _catalogue is not None else 0),
    }
    return JSONResponse(content=payload)


@app.get("/api/categories")
def list_categories() -> JSONResponse:
    if _catalogue is None:
        raise HTTPException(status_code=503, detail="Catalogue not ready")

    categories = sorted({str(category) for category in _catalogue["category"].tolist()})
    payload = {
        "categories": ["All"] + categories,
        "total": len(categories),
        "timestamp": datetime.utcnow().isoformat(),
    }
    return JSONResponse(content=payload)


@app.get("/api/products")
def list_products(
    request: Request,
    category: Optional[str] = None,
    available: Optional[bool] = None,
    limit: Optional[int] = None,
) -> JSONResponse:
    if _catalogue is None:
        raise HTTPException(status_code=503, detail="Catalogue not ready")

    df = _catalogue.copy()
    if category and category.lower() != "all":
        df = df[df["category"].str.lower() == category.lower()]
    if available is not None:
        df = df[df["available"].astype(bool) == bool(available)]
    if limit:
        df = df.head(limit)

    items = list(_catalogue_records(df, request))
    payload = {
        "products": items,
        "count": len(items),
        "timestamp": datetime.utcnow().isoformat(),
    }
    return JSONResponse(content=payload)


@app.post("/api/search")
async def search(
    request: Request,
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    top_k: int = Form(12),
    similarity_threshold: float = Form(0.0),
) -> JSONResponse:
    if _embedding_model is None or _similarity_index is None:
        raise HTTPException(status_code=503, detail="Service not ready. Build embeddings first.")

    image_bytes: Optional[bytes] = None
    origin: str = ""

    if file is not None:
        image_bytes = await file.read()
        origin = file.filename or "upload"
    elif image_url:
        origin = image_url
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            image_bytes = response.content
        except Exception as exc:
            LOGGER.exception("Failed to download image from %s: %s", image_url, exc)
            raise HTTPException(status_code=400, detail="Unable to download image from provided URL")
    else:
        raise HTTPException(status_code=400, detail="Provide either an image file or an image_url")

    if image_bytes is None or not ImagePreprocessor.is_image(image_bytes):
        raise HTTPException(status_code=400, detail="Supplied file is not a valid image")

    processed_image = ImagePreprocessor.load(image_bytes)
    processed_image = ImagePreprocessor.enhance(processed_image)

    embedding = _embedding_model.embed(processed_image)
    results = _similarity_index.search(
        embedding,
        limit=max(1, min(int(top_k), 50)),
        threshold=float(max(0.0, min(similarity_threshold, 1.0))),
    )

    enriched = [_enrich_media(result, request) for result in results]
    query_id = str(uuid.uuid4())
    await _persist_query(image_bytes, query_id, origin, len(enriched))

    payload = {
        "query_id": query_id,
        "results": enriched,
        "total": len(enriched),
        "timestamp": datetime.utcnow().isoformat(),
    }
    return JSONResponse(content=payload)


async def _persist_query(image_bytes: bytes, query_id: str, origin: str, results_count: int) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    queries_dir = LOG_DIR / "queries"
    queries_dir.mkdir(parents=True, exist_ok=True)

    image_path = queries_dir / f"{query_id}.jpg"
    async with aiofiles.open(image_path, "wb") as handle:
        await handle.write(image_bytes)

    log_path = LOG_DIR / "queries.log"
    entry = {
        "query_id": query_id,
        "timestamp": datetime.utcnow().isoformat(),
        "origin": origin,
        "results": results_count,
    }
    async with aiofiles.open(log_path, "a") as handle:
        await handle.write(json.dumps(entry) + "\n")


@app.get("/api/products/{product_id}/related")
def related_products(request: Request, product_id: str, limit: int = 6) -> JSONResponse:
    if _similarity_index is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    matches = _similarity_index.neighbours_for(product_id, limit=limit)
    payload = {
        "products": [_enrich_media(match, request) for match in matches],
        "count": len(matches),
        "timestamp": datetime.utcnow().isoformat(),
    }
    return JSONResponse(content=payload)
