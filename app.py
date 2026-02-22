
from __future__ import annotations

from pathlib import Path

import streamlit as st

from embeddings import EmbeddableItem, build_embeddable_items, upsert_embeddings
from legal_ai import ClauseDetector, RiskClassifier
from llm import answer_with_citations
from parser import ParsedDocument, load_document
from retrieval import hybrid_search
from utils import CitationChunk, build_chunks


def _run_pipeline(file_path: Path) -> tuple[ParsedDocument, list[ClauseDetector], list, list[CitationChunk]]:
    """Run parsing, clause detection, risk assessment, and chunking."""
    parsed = load_document(str(file_path))

    detector = ClauseDetector()
    clauses = detector.detect(parsed)

    classifier = RiskClassifier()
    risks = classifier.classify(clauses)

    chunks = build_chunks(parsed, clauses, risks)
    return parsed, clauses, risks, chunks


def _index_in_qdrant(
    collection: str,
    chunks: list[CitationChunk],
    clauses,
) -> None:
    """Build embeddings and store them in Qdrant."""
    items = build_embeddable_items(chunks, clauses)
    upsert_embeddings(collection, items)


def _display_images(parsed: ParsedDocument) -> None:
    """Show extracted images using metadata paths."""
    image_elements = [
        el for el in parsed.elements
        if el.element_type == el.element_type.IMAGE
        and el.metadata.get("image_path")
    ]
    if not image_elements:
        st.info("No images detected in this document.")
        return

    for el in image_elements:
        path = el.metadata.get("image_path")
        if not path:
            continue
        st.image(path, caption=el.content)


def main() -> None:
    """Render the Streamlit interface."""
    st.set_page_config(page_title="Legal AI RAG", layout="wide")
    st.title("Privacy-First Legal AI RAG")

    uploaded = st.file_uploader("Upload a PDF contract", type=["pdf"])
    collection_name = st.text_input("Qdrant collection", value="legal_ai_documents")

    if not uploaded:
        st.stop()

    temp_dir = Path("data/uploads")
    temp_dir.mkdir(parents=True, exist_ok=True)
    file_path = temp_dir / uploaded.name
    with open(file_path, "wb") as f:
        f.write(uploaded.read())

    with st.spinner("Parsing document and detecting clauses..."):
        parsed, clauses, risks, chunks = _run_pipeline(file_path)

    _index_in_qdrant(collection_name, chunks, clauses)

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Question Answering")
        question = st.text_input("Ask a question about this contract")
        if question:
            with st.spinner("Retrieving relevant chunks and generating answer..."):
                results = hybrid_search(collection_name, question, top_k=5)

                id_to_chunk = {c.chunk_id: c for c in chunks}
                retrieved_chunks: list[CitationChunk] = []
                for r in results:
                    chunk = id_to_chunk.get(r.id)
                    if chunk:
                        retrieved_chunks.append(chunk)

                answer = answer_with_citations(question, retrieved_chunks)
            st.markdown(answer)

            with st.expander("Retrieved chunks"):
                for c in retrieved_chunks:
                    citation = c.section_heading or "Unknown section"
                    page = c.page_number or "?"
                    st.markdown(f"**{citation} (Page {page})**")
                    st.write(c.content)

    with col_right:
        st.subheader("Detected Clauses")
        if not clauses:
            st.write("No clauses detected.")
        else:
            for clause, risk in zip(clauses, risks):
                page = clause.page_number or "?"
                st.markdown(
                    f"**{clause.clause_type.value}** — "
                    f"{risk.risk_level.value.upper()} (Page {page})"
                )
                st.caption(risk.explanation)

        st.subheader("Extracted Images")
        _display_images(parsed)


if __name__ == "__main__":
    main()
