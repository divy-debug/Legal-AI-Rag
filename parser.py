from __future__ import annotations

import base64
import logging
import os
import re
import shutil
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ElementType(str, Enum):
    """Types of elements extracted from a document."""

    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    TITLE = "title"
    HEADER = "header"
    FOOTER = "footer"
    LIST_ITEM = "list_item"
    NARRATIVE_TEXT = "narrative_text"


class ParsedElement(BaseModel):
    """A single extracted element (text block, table, or image)."""

    element_type: ElementType
    content: str = Field(
        description="Primary text content, table text, or image caption"
    )
    page_number: int | None = Field(
        default=None,
        description="1-indexed page number where this element appears",
    )
    element_index: int = Field(
        description="Position of this element in the document (0-indexed)"
    )
    section_heading: str | None = Field(
        default=None,
        description="Nearest preceding heading or section title, if available",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Raw text, structured table JSON, image path, and extra metadata",
    )


class ParsedDocument(BaseModel):
    """Complete parsed representation of a single document."""

    source_file: str
    file_type: str
    elements: list[ParsedElement] = Field(default_factory=list)

    @property
    def text_count(self) -> int:
        """Return number of non-table, non-image elements."""
        return sum(
            1
            for e in self.elements
            if e.element_type not in (ElementType.TABLE, ElementType.IMAGE)
        )

    @property
    def table_count(self) -> int:
        """Return number of table elements."""
        return sum(1 for e in self.elements if e.element_type == ElementType.TABLE)

    @property
    def image_count(self) -> int:
        """Return number of image elements."""
        return sum(1 for e in self.elements if e.element_type == ElementType.IMAGE)


_CATEGORY_MAP: dict[str, ElementType] = {
    "Title": ElementType.TITLE,
    "Header": ElementType.HEADER,
    "Footer": ElementType.FOOTER,
    "NarrativeText": ElementType.NARRATIVE_TEXT,
    "ListItem": ElementType.LIST_ITEM,
    "Table": ElementType.TABLE,
    "Image": ElementType.IMAGE,
    "FigureCaption": ElementType.TEXT,
    "Text": ElementType.TEXT,
    "UncategorizedText": ElementType.TEXT,
    "Formula": ElementType.TEXT,
    "Address": ElementType.TEXT,
    "EmailAddress": ElementType.TEXT,
    "PageBreak": ElementType.TEXT,
}


_SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md", ".rtf"}
_IMAGE_DIR_NAME = "extracted_images"


def _resolve_element_type(category: str) -> ElementType:
    """Map an Unstructured category string to our ElementType enum."""
    return _CATEGORY_MAP.get(category, ElementType.TEXT)


def _rows_to_markdown(rows: list[list[str]]) -> str:
    """Convert a list of rows into a markdown table string."""
    if not rows:
        return ""

    col_count = max(len(r) for r in rows)
    for row in rows:
        while len(row) < col_count:
            row.append("")

    header = "| " + " | ".join(rows[0]) + " |"
    separator = "| " + " | ".join(["---"] * col_count) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join([header, separator, *body])


def _html_table_to_text_and_rows(html: str) -> tuple[str, list[list[str]]]:
    """Convert an HTML table to markdown text and structured rows."""
    try:
        rows: list[list[str]] = []
        for tr_match in re.finditer(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL):
            cells = re.findall(
                r"<t[hd][^>]*>(.*?)</t[hd]>", tr_match.group(1), re.DOTALL
            )
            clean_cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
            rows.append(clean_cells)
        if not rows:
            return html, []
        return _rows_to_markdown(rows), rows
    except Exception:
        return html, []


def _extract_pdf_tables(file_path: Path) -> list[tuple[int, str, list[list[str]]]]:
    """Extract tables from a PDF using pdfplumber."""
    import pdfplumber

    tables: list[tuple[int, str, list[list[str]]]] = []

    try:
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_tables = page.extract_tables()
                if not page_tables:
                    continue

                for table_data in page_tables:
                    if not table_data or len(table_data) < 2:
                        continue

                    cleaned_rows = [
                        [(cell or "").strip().replace("\n", " ") for cell in row]
                        for row in table_data
                    ]
                    md_table = _rows_to_markdown(cleaned_rows)
                    if md_table:
                        tables.append((page_num, md_table, cleaned_rows))

        logger.info(
            "pdfplumber extracted %d table(s) from %s",
            len(tables),
            file_path.name,
        )
    except Exception as exc:
        logger.warning("pdfplumber table extraction failed: %s", exc)

    return tables


def _ensure_image_dir(base_dir: Path) -> Path:
    """Ensure the image output directory exists and return its path."""
    target = base_dir / _IMAGE_DIR_NAME
    target.mkdir(parents=True, exist_ok=True)
    return target


def _save_image_to_disk(
    image_bytes: bytes | None,
    source_path: str | None,
    target_dir: Path,
    stem: str,
    page_number: int | None,
    index: int,
) -> str | None:
    """Persist an image to the extracted_images directory and return its path."""
    if not image_bytes and not source_path:
        return None

    page_suffix = f"_p{page_number}" if page_number is not None else ""
    filename = f"{stem}{page_suffix}_img{index}.png"
    target_path = target_dir / filename

    try:
        if image_bytes:
            with open(target_path, "wb") as f:
                f.write(image_bytes)
        elif source_path:
            if os.path.abspath(source_path) != os.fspath(target_path):
                shutil.copy2(source_path, target_path)
        return os.fspath(target_path)
    except Exception as exc:
        logger.warning("Failed to save image to %s: %s", target_path, exc)
        return None


def _generate_image_caption(
    page_number: int | None,
    section_heading: str | None,
) -> str:
    """Generate a short caption for an image using the LLM helper."""
    try:
        from llm import generate_image_caption
    except Exception:
        heading = section_heading or "contract"
        page_text = f"page {page_number}" if page_number is not None else "the document"
        return f"Embedded image in {heading} on {page_text}"

    return generate_image_caption(page_number=page_number, section_heading=section_heading)


def _load_with_unstructured(file_path: Path) -> list[Any]:
    """Load raw elements from Unstructured via LangChain integration."""
    from langchain_unstructured import UnstructuredLoader

    loader = UnstructuredLoader(
        str(file_path),
        mode="elements",
        strategy="fast",
        unstructured_kwargs={"infer_table_structure": True},
    )
    raw_docs = loader.load()
    logger.info("Extracted %d raw elements from %s", len(raw_docs), file_path.name)
    return raw_docs


def _base_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Return metadata dictionary excluding keys stored separately."""
    excluded = {
        "category",
        "page_number",
        "text_as_html",
        "image_base64",
        "image_path",
        "table",
    }
    return {k: v for k, v in meta.items() if k not in excluded}


def _iter_with_index(items: Iterable[Any]) -> Iterable[tuple[int, Any]]:
    """Yield (index, item) pairs for an iterable."""
    for idx, item in enumerate(items):
        yield idx, item


def _build_elements_from_raw(
    raw_docs: list[Any],
    file_path: Path,
) -> list[ParsedElement]:
    """Convert raw Unstructured elements into ParsedElement instances."""
    elements: list[ParsedElement] = []
    current_heading: str | None = None
    image_dir = _ensure_image_dir(file_path.parent)
    image_counter = 0

    for idx, doc in _iter_with_index(raw_docs):
        meta = doc.metadata or {}
        category = meta.get("category", "Text")
        el_type = _resolve_element_type(category)

        if el_type in (ElementType.TITLE, ElementType.HEADER):
            current_heading = doc.page_content.strip() or current_heading

        page_number = meta.get("page_number")
        content = doc.page_content or ""
        metadata = _base_metadata(meta)

        if el_type == ElementType.TABLE:
            html_table = meta.get("text_as_html", "")
            if html_table:
                table_text, table_rows = _html_table_to_text_and_rows(html_table)
                content = table_text
                metadata["table_rows"] = table_rows
                metadata["raw_html"] = html_table
            metadata["raw_text"] = doc.page_content

        elif el_type == ElementType.IMAGE:
            image_base64 = meta.get("image_base64")
            image_path = meta.get("image_path")
            image_bytes: bytes | None = None

            if image_base64:
                try:
                    image_bytes = base64.b64decode(image_base64)
                except Exception:
                    image_bytes = None
            elif image_path and os.path.exists(image_path):
                try:
                    with open(image_path, "rb") as f:
                        image_bytes = f.read()
                except Exception:
                    image_bytes = None

            image_counter += 1
            saved_path = _save_image_to_disk(
                image_bytes=image_bytes,
                source_path=image_path,
                target_dir=image_dir,
                stem=file_path.stem,
                page_number=page_number,
                index=image_counter,
            )

            caption = _generate_image_caption(
                page_number=page_number,
                section_heading=current_heading,
            )

            content = caption
            metadata["image_path"] = saved_path
            metadata["raw_text"] = doc.page_content

        else:
            metadata["raw_text"] = doc.page_content

        elements.append(
            ParsedElement(
                element_type=el_type,
                content=content,
                page_number=page_number,
                element_index=idx,
                section_heading=current_heading,
                metadata=metadata,
            )
        )

    return elements


def _merge_pdf_tables_into_elements(
    elements: list[ParsedElement],
    pdf_tables: list[tuple[int, str, list[list[str]]]],
) -> None:
    """Insert pdfplumber tables into the ParsedElement list in page order."""
    if not pdf_tables:
        return

    idx = len(elements)
    for page_number, table_md, rows in pdf_tables:
        insert_pos = len(elements)
        for i, el in enumerate(elements):
            if el.page_number is not None and el.page_number > page_number:
                insert_pos = i
                break

        heading_ctx = None
        for el in reversed(elements[:insert_pos]):
            if el.section_heading:
                heading_ctx = el.section_heading
                break

        table_element = ParsedElement(
            element_type=ElementType.TABLE,
            content=table_md,
            page_number=page_number,
            element_index=idx,
            section_heading=heading_ctx,
            metadata={
                "source": "pdfplumber",
                "table_rows": rows,
            },
        )
        elements.insert(insert_pos, table_element)
        idx += 1

    for i, el in enumerate(elements):
        el.element_index = i


def load_document(file_path: str | Path) -> ParsedDocument:
    """Load and parse a document into structured elements."""
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Supported: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"
        )

    logger.info("Loading document: %s", path)
    raw_docs = _load_with_unstructured(path)

    pdf_tables: list[tuple[int, str, list[list[str]]]] = []
    if suffix == ".pdf":
        pdf_tables = _extract_pdf_tables(path)

    elements = _build_elements_from_raw(raw_docs, path)
    _merge_pdf_tables_into_elements(elements, pdf_tables)

    parsed = ParsedDocument(
        source_file=str(path),
        file_type=suffix.lstrip("."),
        elements=elements,
    )

    logger.info(
        "Parsed %s: %d text, %d tables, %d images",
        path.name,
        parsed.text_count,
        parsed.table_count,
        parsed.image_count,
    )
    return parsed
