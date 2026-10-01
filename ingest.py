"""PDF ingestion: extract -> chunk -> embed -> persist in Chroma.

The embedding model and the vector store are expensive to build, so both are
wrapped in ``@st.cache_resource`` and shared across Streamlit reruns.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import streamlit as st
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from config import settings
from utils import get_logger, sha256_bytes

log = get_logger(__name__)

# A page with fewer than this many characters is treated as "no extractable
# text" - typically a scan, a full-page figure, or a cover page.
MIN_CHARS_PER_PAGE = 30


@dataclass
class IngestResult:
    """Outcome of ingesting a single PDF."""

    filename: str
    pages_read: int = 0
    pages_with_text: int = 0
    chunks_created: int = 0
    skipped: bool = False
    warning: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        """Whether the file produced at least one stored chunk."""
        return self.error is None and (self.chunks_created > 0 or self.skipped)


@dataclass
class DocumentInfo:
    """Summary of one already-ingested document, for the sidebar."""

    source: str
    chunks: int = 0
    pages: set[int] = field(default_factory=set)
    file_hash: str = ""

    @property
    def page_count(self) -> int:
        """Number of distinct pages that contributed chunks."""
        return len(self.pages)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the embedding model (first run downloads it)...")
def get_embeddings() -> Any:
    """Load the local sentence-transformers embedding model.

    The model runs on the CPU and needs no API key. It is downloaded once to the
    Hugging Face cache and reused afterwards.

    Returns:
        A LangChain embeddings object exposing ``embed_documents``/``embed_query``.
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:  # pragma: no cover - fallback for older installs
        from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource(show_spinner="Opening the vector store...")
def get_vectorstore() -> Chroma:
    """Open (or create) the persistent Chroma collection.

    Returns:
        A :class:`~langchain_chroma.Chroma` instance backed by ``settings.chroma_dir``.
    """
    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.chroma_path),
    )


# ---------------------------------------------------------------------------
# Pure helpers (no Streamlit, no network - these are what the tests exercise)
# ---------------------------------------------------------------------------
def extract_pdf_pages(data: bytes, filename: str) -> tuple[list[Document], int]:
    """Extract one :class:`Document` per non-empty page of a PDF.

    Args:
        data: Raw PDF bytes.
        filename: Display name used as the ``source`` metadata value.

    Returns:
        A tuple of (documents, total_pages_in_pdf). Pages whose extracted text is
        shorter than :data:`MIN_CHARS_PER_PAGE` are skipped but still counted.

    Raises:
        ValueError: If the bytes cannot be read as a PDF.
    """
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:  # pypdf raises several unrelated types
        raise ValueError(f"'{filename}' could not be opened as a PDF: {exc}") from exc

    documents: list[Document] = []
    total_pages = len(reader.pages)
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # a single corrupt page should not kill the file
            log.warning("%s p.%s: extraction failed (%s)", filename, page_number, exc)
            continue
        text = text.strip()
        if len(text) < MIN_CHARS_PER_PAGE:
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={"source": filename, "page": page_number},
            )
        )
    return documents, total_pages


def chunk_documents(
    documents: Sequence[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split page-level documents into overlapping retrieval chunks.

    Args:
        documents: Page-level documents produced by :func:`extract_pdf_pages`.
        chunk_size: Characters per chunk; defaults to ``settings.chunk_size``.
        chunk_overlap: Overlap between neighbours; defaults to ``settings.chunk_overlap``.

    Returns:
        Chunk documents, each inheriting the ``source`` and ``page`` metadata of
        the page it came from.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    return splitter.split_documents(list(documents))


# ---------------------------------------------------------------------------
# Store-facing operations
# ---------------------------------------------------------------------------
def _all_metadatas(vectorstore: Chroma) -> list[dict[str, Any]]:
    """Return every stored chunk's metadata.

    Args:
        vectorstore: The Chroma collection.

    Returns:
        A list of metadata dictionaries (empty when the store is empty).
    """
    try:
        record = vectorstore.get(include=["metadatas"])
    except Exception as exc:  # pragma: no cover - store not yet created
        log.warning("Could not read the collection: %s", exc)
        return []
    return [m or {} for m in (record.get("metadatas") or [])]


def stored_hashes(vectorstore: Chroma) -> set[str]:
    """Collect the SHA-256 hashes of every PDF already ingested.

    Args:
        vectorstore: The Chroma collection.

    Returns:
        A set of hex digests used to skip duplicate uploads.
    """
    return {
        str(meta["file_hash"])
        for meta in _all_metadatas(vectorstore)
        if meta.get("file_hash")
    }


def list_documents(vectorstore: Chroma) -> list[DocumentInfo]:
    """Summarise the documents currently in the vector store.

    Args:
        vectorstore: The Chroma collection.

    Returns:
        One :class:`DocumentInfo` per distinct ``source``, sorted by name.
    """
    by_source: dict[str, DocumentInfo] = {}
    for meta in _all_metadatas(vectorstore):
        source = str(meta.get("source", "unknown"))
        info = by_source.setdefault(source, DocumentInfo(source=source))
        info.chunks += 1
        page = meta.get("page")
        if isinstance(page, int):
            info.pages.add(page)
        if not info.file_hash and meta.get("file_hash"):
            info.file_hash = str(meta["file_hash"])
    return [by_source[key] for key in sorted(by_source)]


def ingest_pdf(
    vectorstore: Chroma,
    data: bytes,
    filename: str,
    known_hashes: Iterable[str] = (),
    progress_cb: Callable[[float, str], None] | None = None,
) -> IngestResult:
    """Ingest a single PDF into the vector store, skipping duplicates.

    Args:
        vectorstore: The Chroma collection to write to.
        data: Raw PDF bytes.
        filename: Display name, stored as ``source`` metadata.
        known_hashes: Hashes already present in the store.
        progress_cb: Optional callback receiving ``(fraction, message)`` updates.

    Returns:
        An :class:`IngestResult` describing what happened. Failures are reported
        on the result object rather than raised, so one bad file cannot abort a
        multi-file upload.
    """

    def report(fraction: float, message: str) -> None:
        if progress_cb is not None:
            progress_cb(fraction, message)

    file_hash = sha256_bytes(data)
    if file_hash in set(known_hashes):
        report(1.0, f"{filename}: already ingested")
        return IngestResult(
            filename=filename,
            skipped=True,
            warning=f"'{filename}' is already in your library - skipped.",
        )

    report(0.1, f"Reading {filename}...")
    try:
        pages, total_pages = extract_pdf_pages(data, filename)
    except ValueError as exc:
        return IngestResult(filename=filename, error=str(exc))

    if not pages:
        return IngestResult(
            filename=filename,
            pages_read=total_pages,
            warning=(
                f"No selectable text found in '{filename}' ({total_pages} page(s)). "
                "It is probably a scan or an image-only export - run OCR on it first "
                "(for example with OCRmyPDF) and upload it again."
            ),
        )

    report(0.45, f"Splitting {filename} into chunks...")
    chunks = chunk_documents(pages)
    for chunk in chunks:
        chunk.metadata["file_hash"] = file_hash

    if not chunks:
        return IngestResult(
            filename=filename,
            pages_read=total_pages,
            pages_with_text=len(pages),
            warning=f"'{filename}' produced no usable chunks.",
        )

    report(0.65, f"Embedding {len(chunks)} chunks from {filename}...")
    try:
        ids = [f"{file_hash}:{i}" for i in range(len(chunks))]
        vectorstore.add_documents(documents=chunks, ids=ids)
    except Exception as exc:
        log.exception("Failed to write '%s' to Chroma", filename)
        return IngestResult(
            filename=filename,
            pages_read=total_pages,
            pages_with_text=len(pages),
            error=f"Could not store '{filename}' in the vector database: {exc}",
        )

    report(1.0, f"{filename}: done")
    return IngestResult(
        filename=filename,
        pages_read=total_pages,
        pages_with_text=len(pages),
        chunks_created=len(chunks),
        warning=(
            f"{total_pages - len(pages)} page(s) of '{filename}' had no selectable "
            "text and were skipped."
            if total_pages > len(pages)
            else None
        ),
    )


def clear_all(vectorstore: Chroma) -> None:
    """Delete every document in the collection and reset cached resources.

    Args:
        vectorstore: The Chroma collection to wipe.
    """
    try:
        vectorstore.delete_collection()
    except Exception as exc:  # pragma: no cover - already gone
        log.warning("delete_collection failed: %s", exc)
    finally:
        get_vectorstore.clear()


__all__ = [
    "DocumentInfo",
    "IngestResult",
    "MIN_CHARS_PER_PAGE",
    "chunk_documents",
    "clear_all",
    "extract_pdf_pages",
    "get_embeddings",
    "get_vectorstore",
    "ingest_pdf",
    "list_documents",
    "stored_hashes",
]
