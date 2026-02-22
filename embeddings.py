
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from legal_ai.clause_detector import DetectedClause
from parser import ParsedDocument
from utils import CitationChunk

load_dotenv()


ElementKind = Literal["text", "table", "image"]


@dataclass
class EmbeddableItem:
    """Lightweight representation of a chunk ready for embedding."""

    id: str
    text: str
    element_type: ElementKind
    page_number: int | None
    clause_type: str | None
    original_text: str


def _openai_client() -> OpenAI:
    """Return a cached OpenAI client."""
    from llm import _client as _llm_client

    return _llm_client()


def _qdrant_client() -> QdrantClient:
    """Create a Qdrant client connected to local instance."""
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY") or None
    return QdrantClient(url=url, api_key=api_key)


def _embedding_model_name() -> str:
    """Return embedding model name from environment or default."""
    return os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


def build_embeddable_items(
    chunks: Sequence[CitationChunk],
    clauses: Sequence[DetectedClause],
) -> list[EmbeddableItem]:
    """
    Convert chunks and clause annotations into embeddable items.
    """
    clause_by_element = {c.element_index: c for c in clauses}
    items: list[EmbeddableItem] = []

    for chunk in chunks:
        element_type: ElementKind = (
            "image" if chunk.element_type == "image" else
            "table" if chunk.element_type == "table" else
            "text"
        )

        clause_type = None
        if chunk.element_indices:
            for idx in chunk.element_indices:
                clause = clause_by_element.get(idx)
                if clause:
                    clause_type = clause.clause_type.value
                    break

        original_text = chunk.content

        items.append(
            EmbeddableItem(
                id=chunk.chunk_id,
                text=chunk.content,
                element_type=element_type,
                page_number=chunk.page_number,
                clause_type=clause_type,
                original_text=original_text,
            )
        )

    return items


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Embed a list of texts using OpenAI embeddings."""
    if not texts:
        return []

    client = _openai_client()
    model = _embedding_model_name()

    response = client.embeddings.create(model=model, input=list(texts))
    vectors = [d.embedding for d in response.data]
    return vectors


def ensure_collection(collection: str, vector_size: int) -> None:
    """Create the Qdrant collection if it does not exist."""
    client = _qdrant_client()

    existing = client.get_collections()
    if any(c.name == collection for c in existing.collections):
        return

    client.create_collection(
        collection_name=collection,
        vectors_config=qm.VectorParams(
            size=vector_size,
            distance=qm.Distance.COSINE,
        ),
    )


def upsert_embeddings(
    collection: str,
    items: Sequence[EmbeddableItem],
) -> None:
    """
    Compute embeddings for items and upsert them into Qdrant.
    """
    if not items:
        return

    vectors = embed_texts([i.text for i in items])
    if not vectors:
        return

    ensure_collection(collection, vector_size=len(vectors[0]))

    client = _qdrant_client()
    points: list[qm.PointStruct] = []

    for item, vec in zip(items, vectors):
        payload = {
            "element_type": item.element_type,
            "page_number": item.page_number,
            "clause_type": item.clause_type,
            "original_text": item.original_text,
        }
        points.append(
            qm.PointStruct(id=item.id, vector=vec, payload=payload)
        )

    client.upsert(collection_name=collection, points=points)

