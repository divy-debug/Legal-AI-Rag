

from __future__ import annotations

import os
from typing import Iterable, Sequence

from dotenv import load_dotenv
from openai import OpenAI

from legal_ai.clause_detector import ClauseType
from utils import CitationChunk

load_dotenv()

_CLIENT: OpenAI | None = None


def _client() -> OpenAI:
    """Return a cached OpenAI client."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OpenAI()
    return _CLIENT


def generate_image_caption(
    *,
    page_number: int | None,
    section_heading: str | None,
    model: str | None = None,
) -> str:
    """
    Generate a short generic caption for an embedded image.

    The caption is intentionally generic to avoid leaking sensitive content.
    """
    client = _client()
    model_name = model or os.getenv("OPENAI_IMAGE_CAPTION_MODEL", "gpt-4.1-mini")

    page_text = f"page {page_number}" if page_number is not None else "the document"
    heading = section_heading or "an unlabeled section"

    prompt = (
        "You are helping describe images in legal contracts without seeing the "
        "actual image. Generate a short, neutral caption of at most 12 words "
        "based only on the context.\n\n"
        f"Context:\n- Document type: legal contract\n- Page: {page_text}\n"
        f"- Section heading: {heading}\n\n"
        "The caption must not invent details about parties, amounts, dates, "
        "or jurisdictions. Use generic wording like 'embedded figure', "
        "'signature block', or 'company logo' if unsure."
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You write concise, neutral image captions."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=32,
    )

    text = response.choices[0].message.content or ""
    return text.strip()


def classify_clause_risk(
    clause_type: ClauseType,
    clause_text: str,
    model: str | None = None,
) -> tuple[str, str]:
    """
    Classify a clause as High/Medium/Low risk with explanation.

    Returns (risk_level, explanation).
    """
    client = _client()
    model_name = model or os.getenv("OPENAI_RISK_MODEL", "gpt-4.1-mini")

    system = (
        "You are a legal assistant classifying contract clauses into risk levels. "
        "Return a JSON object with keys 'risk_level' and 'explanation'. "
        "risk_level must be one of: High, Medium, Low."
    )

    user = (
        f"Clause type: {clause_type.value}\n\n"
        "Clause text:\n"
        f"{clause_text}\n\n"
        "Decide the overall risk from the perspective of a cautious but "
        "commercially reasonable customer. "
        "Explain your reasoning in 2–3 sentences, focusing on concrete wording."
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        max_tokens=256,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content or "{}"
    import json

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return "Medium", (
            "The clause presents some potential concerns, but the analysis "
            "could not be fully parsed. Manual review is recommended."
        )

    level = str(data.get("risk_level", "Medium")).strip()
    explanation = str(data.get("explanation", "")).strip()
    if level not in {"High", "Medium", "Low"}:
        level = "Medium"
    if not explanation:
        explanation = (
            "The clause has been assessed as "
            f"{level} risk based on its wording."
        )
    return level, explanation


def _format_citation(chunk: CitationChunk) -> str:
    """Return a human-readable citation string for a chunk."""
    page = f"Page {chunk.page_number}" if chunk.page_number else "Unknown page"
    heading = chunk.section_heading or "Unknown section"
    return f"{heading} ({page})"


def answer_with_citations(
    question: str,
    chunks: Sequence[CitationChunk],
    model: str | None = None,
) -> str:
    """
    Answer a question strictly using the provided chunks.

    The response includes inline citations in the form
    'According to <heading> (Page X)...'.
    """
    client = _client()
    model_name = model or os.getenv("OPENAI_QA_MODEL", "gpt-4.1-mini")

    context_blocks: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        citation = _format_citation(chunk)
        context_blocks.append(
            f"[{i}] {citation}\n"
            f"Type: {chunk.element_type}\n"
            f"Content:\n{chunk.content}\n"
        )

    context_text = "\n\n".join(context_blocks)

    system = (
        "You are a legal RAG assistant. Answer strictly from the provided "
        "context chunks. If the answer is not supported, say you do not know.\n"
        "When you cite a chunk, use the format "
        "\"According to <section> (Page N)...\" and make sure the citation "
        "actually matches the supporting text."
    )

    user = (
        f"Question:\n{question}\n\n"
        "Context chunks:\n"
        f"{context_text}\n\n"
        "Answer the question using the most relevant chunks. If multiple "
        "chunks are relevant, synthesize them, and include citations."
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        max_tokens=512,
    )

    text = response.choices[0].message.content or ""
    return text.strip()


def chunk_ids_for_logging(chunks: Iterable[CitationChunk]) -> list[str]:
    """Return chunk IDs for lightweight logging and debugging."""
    return [c.chunk_id for c in chunks]

