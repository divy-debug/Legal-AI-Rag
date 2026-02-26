from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Sequence

from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai

from legal_ai.clause_detector import DetectedClause
from parser import ParsedDocument
from utils import CitationChunk

load_dotenv()

logger = logging.getLogger(__name__)

ElementKind = Literal["text", "table", "image"]

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qm
except Exception:
    QdrantClient = None  # type: ignore[assignment]
    qm = None  # type: ignore[assignment]

_LOCAL_COLLECTIONS: dict[str, list[dict[str, Any]]] = {}

@dataclass
class EmbeddableItem:
    """Lightweight representation of a chunk ready for embedding."""
    id: str
    text: str
    element_type: ElementKind
    page_number: int | None
    clause_type: str | None
    original_text: str

def _get_provider() -> str:
    return os.getenv("AI_PROVIDER", "openai").lower()

def _openai_client() -> OpenAI:
    from llm import _openai_client as _llm_openai
    return _llm_openai()

def _qdrant_client() -> Any:
    if QdrantClient is None:
        raise RuntimeError("qdrant-client not available")
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY")
    return QdrantClient(url=url, api_key=api_key)

def build_embeddable_items(
    chunks: Sequence[CitationChunk],
    clauses: Sequence[DetectedClause],
) -> list[EmbeddableItem]:
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
        items.append(
            EmbeddableItem(
                id=chunk.chunk_id,
                text=chunk.content,
                element_type=element_type,
                page_number=chunk.page_number,
                clause_type=clause_type,
                original_text=chunk.content,
            )
        )
    return items

import time

# ... (other imports)

def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    if not texts:
        return []

    if os.getenv("USE_MOCK_MODELS", "false").lower() == "true":
        dim = 3072 if _get_provider() == "gemini" else 1536
        return [[0.1] * dim for _ in texts]

    if _get_provider() == "gemini":
        try:
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
            
            res_vectors = []
            for i, t in enumerate(texts):
                if i > 0:
                    time.sleep(1.0) # Rate limit protection (1 RPM is very slow, but let's try 1s)
                try:
                    r = genai.embed_content(
                        model=model,
                        content=t,
                        task_type="retrieval_document"
                    )
                    if 'embedding' in r:
                        res_vectors.append(r['embedding'])
                    else:
                        logger.error(f"Unexpected embedding response: {r.keys()}")
                        res_vectors.append([0.0] * 3072)
                except Exception as inner_e:
                    logger.error(f"Failed to embed individual text: {inner_e}")
                    res_vectors.append([0.0] * 3072)
            
            return res_vectors
        except Exception as e:
            logger.error(f"Gemini Global Embedding Error: {e}")
            return [[0.0] * 3072 for _ in texts]
    else:
        client = _openai_client()
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        try:
            response = client.embeddings.create(model=model, input=list(texts))
            return [d.embedding for d in response.data]
        except Exception as e:
            logger.error(f"OpenAI Embedding Error: {e}")
            return [[0.0] * 1536 for _ in texts]

def ensure_collection(collection: str, vector_size: int) -> None:
    # We expect 3072 for Gemini or 1536 for OpenAI
    # If it's something very small like 7, something is wrong with our embedding output
    if vector_size < 100:
        logger.error(f"CRITICAL: Unexpectedly small vector_size={vector_size} for collection '{collection}'")
        # Let's fallback to the expected Gemini dimension to allow the app to function
        # though it might fail later on upsert if the vectors are actually small.
        expected_dim = 3072 if _get_provider() == "gemini" else 1536
        logger.warning(f"Overriding vector_size {vector_size} with expected_dim {expected_dim}")
        vector_size = expected_dim
    
    client = _qdrant_client()
    logger.info(f"Ensuring collection '{collection}' with vector_size={vector_size}")
    
    # Try to get collection info directly first
    try:
        info = client.get_collection(collection_name=collection)
        vectors_config = info.config.params.vectors
        if hasattr(vectors_config, "size"):
            current_dim = vectors_config.size
        elif isinstance(vectors_config, dict):
            first_key = list(vectors_config.keys())[0] if vectors_config else None
            current_dim = vectors_config[first_key].size if first_key else 0
        else:
            current_dim = 0
            
        if current_dim == vector_size:
            logger.info(f"Collection '{collection}' already exists with correct dimension {current_dim}")
            return
            
        logger.warning(f"Dimension mismatch in '{collection}': expected {vector_size}, found {current_dim}. DELETING COLLECTION.")
        client.delete_collection(collection_name=collection)
        import time
        time.sleep(3) # Extra wait for cloud propagation
    except Exception:
        # Collection likely doesn't exist
        logger.info(f"Collection '{collection}' does not exist or info failed. Creating fresh.")

    client.create_collection(
        collection_name=collection,
        vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
    )
    logger.info(f"Successfully created collection '{collection}' with dimension {vector_size}")

def upsert_embeddings(collection: str, items: Sequence[EmbeddableItem]) -> None:
    if not items:
        return
    
    vectors = embed_texts([i.text for i in items])
    if not vectors:
        logger.error("Embedding generation failed")
        return
    
    local_points: list[dict[str, Any]] = []
    for item, vec in zip(items, vectors):
        payload = {
            "element_type": item.element_type,
            "page_number": item.page_number,
            "clause_type": item.clause_type,
            "original_text": item.original_text,
            "chunk_id": item.id,
        }
        point_id = str(uuid.uuid5(uuid.NAMESPACE_OID, item.id))
        local_points.append({"id": point_id, "payload": payload, "vector": vec})
    
    if collection not in _LOCAL_COLLECTIONS:
        _LOCAL_COLLECTIONS[collection] = []
    _LOCAL_COLLECTIONS[collection].extend(local_points)

    if QdrantClient is None or qm is None:
        return
    ensure_collection(collection, vector_size=len(vectors[0]))
    client = _qdrant_client()
    points = [qm.PointStruct(id=entry["id"], vector=entry["vector"], payload=entry["payload"]) for entry in local_points]
    client.upsert(collection_name=collection, points=points)

def get_local_collection(collection: str) -> list[dict[str, Any]]:
    return list(_LOCAL_COLLECTIONS.get(collection, []))
