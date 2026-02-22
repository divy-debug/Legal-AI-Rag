
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from rank_bm25 import BM25Okapi

from embeddings import embed_texts


@dataclass
class RetrievedPoint:
    """A single retrieved item from hybrid search."""

    id: str
    score: float
    payload: dict


def _qdrant_client() -> QdrantClient:
    """Return a Qdrant client configured for local usage."""
    from embeddings import _qdrant_client as _base_client

    return _base_client()


def _tokenize(text: str) -> list[str]:
    """Very simple tokenizer for BM25."""
    import re

    tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
    return tokens


def _build_bm25_corpus(payloads: Sequence[dict]) -> BM25Okapi:
    """Build a BM25 index over payload original_text fields."""
    corpus_tokens: list[list[str]] = []
    for payload in payloads:
        text = str(payload.get("original_text") or "")
        tokens = _tokenize(text)
        corpus_tokens.append(tokens)
    return BM25Okapi(corpus_tokens)


def _bm25_scores(
    query: str,
    payloads: Sequence[dict],
) -> list[float]:
    """Compute BM25 scores for a query over the given payloads."""
    bm25 = _build_bm25_corpus(payloads)
    query_tokens = _tokenize(query)
    scores = bm25.get_scores(query_tokens)
    return list(scores)


def _normalize(scores: Sequence[float]) -> list[float]:
    """Normalize a list of scores into [0, 1]."""
    if not scores:
        return []
    max_score = max(scores)
    min_score = min(scores)
    if math.isclose(max_score, min_score):
        return [0.0 for _ in scores]
    return [
        (s - min_score) / (max_score - min_score)
        for s in scores
    ]


def dense_search(
    collection: str,
    query: str,
    top_k: int = 10,
) -> list[RetrievedPoint]:
    """
    Run pure dense similarity search in Qdrant for a query string.
    """
    client = _qdrant_client()
    vector = embed_texts([query])[0]

    result = client.search(
        collection_name=collection,
        query_vector=vector,
        limit=top_k,
    )

    points: list[RetrievedPoint] = []
    for r in result:
        payload = r.payload or {}
        points.append(
            RetrievedPoint(
                id=str(r.id),
                score=float(r.score or 0.0),
                payload=payload,
            )
        )
    return points


def hybrid_search(
    collection: str,
    query: str,
    top_k: int = 10,
    dense_weight: float = 0.7,
    bm25_weight: float = 0.3,
) -> list[RetrievedPoint]:
    """
    Run hybrid dense + BM25 retrieval.
    """
    dense_results = dense_search(collection, query, top_k=top_k)
    if not dense_results:
        return []

    payloads = [p.payload for p in dense_results]
    bm25 = _bm25_scores(query, payloads)

    dense_scores = [p.score for p in dense_results]
    dense_norm = _normalize(dense_scores)
    bm25_norm = _normalize(bm25)

    combined: list[RetrievedPoint] = []
    for point, d_score, b_score in zip(dense_results, dense_norm, bm25_norm):
        score = dense_weight * d_score + bm25_weight * b_score
        combined.append(
            RetrievedPoint(
                id=point.id,
                score=score,
                payload=point.payload,
            )
        )

    combined.sort(key=lambda p: p.score, reverse=True)
    return combined[:top_k]

