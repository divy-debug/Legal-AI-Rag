from __future__ import annotations

import logging
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from legal_ai.clause_detector import ClauseType, DetectedClause
from legal_ai.risk_classifier import RiskAssessment, RiskLevel
from parser import ElementType, ParsedDocument, ParsedElement

logger = logging.getLogger(__name__)


class CitationChunk(BaseModel):
    """A retrieval-ready chunk with full citation and annotation metadata."""

    chunk_id: str = Field(description="Unique identifier: <source>_chunk_<N>")
    content: str = Field(description="Chunk text, markdown table, or base64 image")
    element_type: str = Field(description="text | table | image")

    source_file: str
    page_number: int | None = None
    section_heading: str | None = None
    element_indices: list[int] = Field(
        default_factory=list,
        description="Indices of source ParsedElements that contribute to this chunk",
    )

    clause_types: list[str] = Field(
        default_factory=list,
        description="Clause types detected in this chunk",
    )
    risk_levels: list[str] = Field(
        default_factory=list,
        description="Risk levels associated with detected clauses",
    )
    risk_explanations: list[str] = Field(
        default_factory=list,
        description="Risk explanations for each detected clause",
    )

    metadata: dict[str, Any] = Field(default_factory=dict)


def _build_annotation_index(
    clauses: list[DetectedClause],
    risks: list[RiskAssessment],
) -> dict[int, list[tuple[ClauseType, RiskLevel, str]]]:
    """
    Build a mapping: element_index → list of (clause_type, risk_level, explanation).
    """
    risk_by_idx: dict[int, list[tuple[ClauseType, RiskLevel, str]]] = {}

    for i, clause in enumerate(clauses):
        if i < len(risks):
            risk = risks[i]
            entry = (clause.clause_type, risk.risk_level, risk.explanation)
        else:
            entry = (clause.clause_type, RiskLevel.MEDIUM, "Risk not assessed")
        risk_by_idx.setdefault(clause.element_index, []).append(entry)

    return risk_by_idx


def _is_atomic(element: ParsedElement) -> bool:
    """Tables and images should never be split."""
    return element.element_type in (ElementType.TABLE, ElementType.IMAGE)


DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


def build_chunks(
    parsed_doc: ParsedDocument,
    clauses: list[DetectedClause],
    risks: list[RiskAssessment],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[CitationChunk]:
    """Convert a parsed and annotated document into citation-aware chunks."""
    annotation_index = _build_annotation_index(clauses, risks)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    )

    chunks: list[CitationChunk] = []
    chunk_counter = 0

    groups = _group_elements(parsed_doc.elements)

    for group in groups:
        if len(group) == 1 and _is_atomic(group[0]):
            el = group[0]
            annotations = annotation_index.get(el.element_index, [])

            chunk_counter += 1
            chunks.append(
                CitationChunk(
                    chunk_id=f"{parsed_doc.source_file}_chunk_{chunk_counter}",
                    content=el.content,
                    element_type=el.element_type.value,
                    source_file=parsed_doc.source_file,
                    page_number=el.page_number,
                    section_heading=el.section_heading,
                    element_indices=[el.element_index],
                    clause_types=[a[0].value for a in annotations],
                    risk_levels=[a[1].value for a in annotations],
                    risk_explanations=[a[2] for a in annotations],
                )
            )
        else:
            merged_text = "\n\n".join(el.content for el in group if el.content.strip())
            if not merged_text.strip():
                continue

            all_indices = [el.element_index for el in group]
            page_numbers = [el.page_number for el in group if el.page_number]
            section_headings = [el.section_heading for el in group if el.section_heading]

            sub_texts = text_splitter.split_text(merged_text)

            for sub_text in sub_texts:
                chunk_counter += 1

                relevant_annotations = _match_annotations_to_subchunk(
                    sub_text, group, annotation_index
                )

                chunks.append(
                    CitationChunk(
                        chunk_id=f"{parsed_doc.source_file}_chunk_{chunk_counter}",
                        content=sub_text,
                        element_type="text",
                        source_file=parsed_doc.source_file,
                        page_number=page_numbers[0] if page_numbers else None,
                        section_heading=section_headings[-1] if section_headings else None,
                        element_indices=all_indices,
                        clause_types=[a[0].value for a in relevant_annotations],
                        risk_levels=[a[1].value for a in relevant_annotations],
                        risk_explanations=[a[2] for a in relevant_annotations],
                    )
                )

    logger.info(
        "Built %d chunks from %d elements (%d atomic, %d text-split)",
        len(chunks),
        len(parsed_doc.elements),
        sum(1 for c in chunks if c.element_type in ("table", "image")),
        sum(1 for c in chunks if c.element_type == "text"),
    )

    return chunks


def _group_elements(elements: list[ParsedElement]) -> list[list[ParsedElement]]:
    """Group consecutive text elements together and keep atomic elements alone."""
    if not elements:
        return []

    groups: list[list[ParsedElement]] = []
    current_text_group: list[ParsedElement] = []

    for el in elements:
        if _is_atomic(el):
            if current_text_group:
                groups.append(current_text_group)
                current_text_group = []
            groups.append([el])
        else:
            current_text_group.append(el)

    if current_text_group:
        groups.append(current_text_group)

    return groups


def _match_annotations_to_subchunk(
    sub_text: str,
    group_elements: list[ParsedElement],
    annotation_index: dict[int, list[tuple[ClauseType, RiskLevel, str]]],
) -> list[tuple[ClauseType, RiskLevel, str]]:
    relevant: list[tuple[ClauseType, RiskLevel, str]] = []
    seen: set[tuple[str, str]] = set()

    for el in group_elements:
        annotations = annotation_index.get(el.element_index, [])
        if not annotations:
            continue

        snippet = el.content[:50].strip()
        if snippet and snippet in sub_text:
            for ann in annotations:
                key = (ann[0].value, ann[1].value)
                if key not in seen:
                    seen.add(key)
                    relevant.append(ann)

    if not relevant:
        for el in group_elements:
            for ann in annotation_index.get(el.element_index, []):
                key = (ann[0].value, ann[1].value)
                if key not in seen:
                    seen.add(key)
                    relevant.append(ann)

    return relevant
