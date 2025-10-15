"""Vector similarity helpers."""

from __future__ import annotations

import logging
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SimilarityIndex:
    """Perform cosine-similarity searches over product embeddings."""

    def __init__(self, embeddings: np.ndarray, catalogue: pd.DataFrame) -> None:
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array")
        if len(embeddings) != len(catalogue):
            logger.warning(
                "Embedding count (%s) does not match products (%s)",
                len(embeddings),
                len(catalogue),
            )

        self.embeddings = self._normalise(np.asarray(embeddings, dtype=np.float32))
        self.catalogue = catalogue.reset_index(drop=True)
        self.dimension = self.embeddings.shape[1]
        logger.info("Similarity index initialised with %s items", len(self.embeddings))

    @staticmethod
    def _normalise(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def search(
        self,
        query: np.ndarray,
        *,
        limit: int = 12,
        threshold: float = 0.0,
    ) -> List[dict]:
        """Return the closest matches to the provided query embedding."""
        if limit <= 0:
            return []

        query = np.asarray(query, dtype=np.float32)
        if query.ndim != 1:
            raise ValueError("Query embedding must be a 1D vector")

        query = query / max(np.linalg.norm(query), 1e-12)
        scores = self.embeddings @ query

        order = np.argsort(scores)[::-1]
        results: List[dict] = []
        for position in order:
            score = float(scores[position])
            if score < threshold:
                break

            product = self.catalogue.iloc[position].to_dict()
            # Replace NaN values with empty strings to avoid JSON serialization errors
            product = {k: ("" if pd.isna(v) else v) for k, v in product.items()}
            product["similarity"] = round(score, 4)
            product["similarity_percentage"] = round(score * 100, 2)
            results.append(product)

            if len(results) >= limit:
                break

        logger.debug("Similarity search returned %s records", len(results))
        return results

    def search_within(
        self,
        query: np.ndarray,
        *,
        categories: Sequence[str],
        limit: int = 12,
    ) -> List[dict]:
        """Restrict the search to a subset of categories."""
        mask = self.catalogue["category"].str.lower().isin([c.lower() for c in categories])
        indices = np.flatnonzero(mask.to_numpy())
        if len(indices) == 0:
            return []

        sub_embeddings = self.embeddings[indices]
        query = np.asarray(query, dtype=np.float32)
        query = query / max(np.linalg.norm(query), 1e-12)
        scores = sub_embeddings @ query
        order = np.argsort(scores)[::-1][:limit]

        matches: List[dict] = []
        for local_idx in order:
            score = float(scores[local_idx])
            product = self.catalogue.iloc[indices[local_idx]].to_dict()
            # Replace NaN values with empty strings to avoid JSON serialization errors
            product = {k: ("" if pd.isna(v) else v) for k, v in product.items()}
            product["similarity"] = round(score, 4)
            product["similarity_percentage"] = round(score * 100, 2)
            matches.append(product)
        return matches

    def neighbours_for(self, product_id: str, *, limit: int = 5) -> List[dict]:
        """Return similar items for a given product identifier."""
        candidates = self.catalogue.index[self.catalogue["id"] == product_id]
        if len(candidates) == 0:
            return []

        index = int(candidates[0])
        reference = self.embeddings[index]
        scores = self.embeddings @ reference
        scores[index] = -1.0

        order = np.argsort(scores)[::-1][:limit]
        neighbours: List[dict] = []
        for position in order:
            score = float(scores[position])
            if score < 0:
                continue
            product = self.catalogue.iloc[position].to_dict()
            product["similarity"] = round(score, 4)
            product["similarity_percentage"] = round(score * 100, 2)
            neighbours.append(product)
        return neighbours
