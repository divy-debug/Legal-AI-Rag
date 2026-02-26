from __future__ import annotations

import base64
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ElementType(str, Enum):
    """Types of elements extracted from a document."""

    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    TITLE = "title"


class ParsedElement(BaseModel):
    """A single extracted element."""

    element_type: ElementType
    content: str = Field(description="Text content, table text, or image caption")
    page_number: int | None = Field(default=None)
    element_index: int = Field(description="0-indexed position")
    section_heading: str | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    """Complete parsed representation of a document."""

    source_file: str
    file_type: str
    elements: list[ParsedElement] = Field(default_factory=list)

    @property
    def text_count(self) -> int:
        return sum(1 for e in self.elements if e.element_type != ElementType.IMAGE)


def _rows_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    col_count = max(len(r) for r in rows)
    header = "| " + " | ".join(rows[0]) + " |"
    sep = "| " + " | ".join(["---"] * col_count) + " |"
    body = ["| " + " | ".join([(str(c) or "") for c in r]) + " |" for r in rows[1:]]
    return "\n".join([header, sep, *body])


def load_document(file_path: str | Path) -> ParsedDocument:
    """Load a PDF using pdfplumber (lightweight alternative to Unstructured)."""
    import pdfplumber

    path = Path(file_path).resolve()
    elements: list[ParsedElement] = []
    el_idx = 0

    image_dir = path.parent / "extracted_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    try:
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # 1. Extract Text
                text = page.extract_text()
                if text:
                    for block in text.split("\n\n"):
                        if not block.strip():
                            continue
                        elements.append(
                            ParsedElement(
                                element_type=ElementType.TEXT,
                                content=block.strip(),
                                page_number=page_num,
                                element_index=el_idx,
                            )
                        )
                        el_idx += 1

                # 2. Extract Tables
                tables = page.extract_tables()
                for table in tables:
                    if not table:
                        continue
                    md = _rows_to_markdown(table)
                    elements.append(
                        ParsedElement(
                            element_type=ElementType.TABLE,
                            content=md,
                            page_number=page_num,
                            element_index=el_idx,
                            metadata={"table_rows": table},
                        )
                    )
                    el_idx += 1

                # 3. Extract Images
                for img_idx, img in enumerate(page.images):
                    img_name = f"{path.stem}_p{page_num}_i{img_idx}.png"
                    img_path = image_dir / img_name
                    
                    try:
                        # Extract image using page cropping
                        # We use a small padding to ensure we capture the whole image
                        bbox = (img["x0"], img["top"], img["x1"], img["bottom"])
                        cropped = page.within_bbox(bbox)
                        img_obj = cropped.to_image(resolution=150)
                        img_obj.save(str(img_path))
                        
                        elements.append(
                            ParsedElement(
                                element_type=ElementType.IMAGE,
                                content=f"Image {img_idx} on page {page_num}",
                                page_number=page_num,
                                element_index=el_idx,
                                metadata={"image_path": str(img_path)},
                            )
                        )
                        el_idx += 1
                    except Exception as img_e:
                        logger.warning(f"Failed to extract image {img_idx} on page {page_num}: {img_e}")

    except Exception as e:
        logger.error(f"Failed to parse {path}: {e}")

    return ParsedDocument(
        source_file=str(path), file_type=path.suffix.lstrip("."), elements=elements
    )
