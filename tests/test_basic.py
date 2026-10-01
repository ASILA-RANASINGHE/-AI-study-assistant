"""Unit tests for chunking and JSON parsing/validation.

These cover the parts of StudyMate AI that must not break silently: the text
splitter contract and the tolerant-but-strict handling of model JSON. No API key
and no network access are required.

Run with::

    pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document  # noqa: E402

from ingest import chunk_documents  # noqa: E402
from utils import (  # noqa: E402
    JSONParseError,
    SchemaValidationError,
    flashcards_to_anki_csv,
    format_citations,
    parse_json_object,
    sha256_bytes,
    strip_code_fences,
    validate_flashcards,
    validate_quiz,
)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def make_page(text: str, page: int = 1, source: str = "lecture1.pdf") -> Document:
    """Build a page-level document for tests."""
    return Document(page_content=text, metadata={"source": source, "page": page})


def test_chunking_splits_long_pages() -> None:
    """A page longer than the chunk size becomes several chunks."""
    page = make_page("word " * 1200)  # ~6000 characters
    chunks = chunk_documents([page], chunk_size=1000, chunk_overlap=150)
    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 1000 for chunk in chunks)


def test_chunking_preserves_metadata() -> None:
    """Source and page metadata survive the split, which is what citations need."""
    pages = [make_page("alpha " * 400, page=3), make_page("beta " * 400, page=7)]
    chunks = chunk_documents(pages, chunk_size=500, chunk_overlap=50)
    pages_seen = {chunk.metadata["page"] for chunk in chunks}
    assert pages_seen == {3, 7}
    assert all(chunk.metadata["source"] == "lecture1.pdf" for chunk in chunks)


def test_chunking_short_page_stays_single_chunk() -> None:
    """A page shorter than the chunk size is not split."""
    chunks = chunk_documents([make_page("A short note.")], chunk_size=1000, chunk_overlap=150)
    assert len(chunks) == 1
    assert chunks[0].page_content == "A short note."


def test_chunking_overlap_repeats_text() -> None:
    """Consecutive chunks share text, so a sentence is never cut in half."""
    page = make_page(" ".join(f"token{i}" for i in range(800)))
    chunks = chunk_documents([page], chunk_size=400, chunk_overlap=100)
    assert len(chunks) >= 2
    tail = chunks[0].page_content[-50:]
    assert any(word in chunks[1].page_content for word in tail.split())


def test_sha256_is_stable_and_distinct() -> None:
    """Hashing is deterministic, which is what makes duplicate detection work."""
    assert sha256_bytes(b"pdf-bytes") == sha256_bytes(b"pdf-bytes")
    assert sha256_bytes(b"pdf-bytes") != sha256_bytes(b"other-bytes")


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------
def test_strip_code_fences_removes_json_fence() -> None:
    """Fenced replies are unwrapped."""
    assert strip_code_fences('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_parse_json_object_plain() -> None:
    """A clean JSON object parses unchanged."""
    assert parse_json_object('{"cards": []}') == {"cards": []}


def test_parse_json_object_with_fence_and_prose() -> None:
    """Prose around the JSON is discarded rather than failing the request."""
    raw = 'Sure! Here you go:\n```json\n{"questions": [1, 2]}\n```\nHope that helps.'
    assert parse_json_object(raw) == {"questions": [1, 2]}


def test_parse_json_object_rejects_garbage() -> None:
    """Unparseable output raises, which triggers the one-shot retry upstream."""
    with pytest.raises(JSONParseError):
        parse_json_object("I cannot produce JSON right now.")


def test_parse_json_object_rejects_array_top_level() -> None:
    """A top-level array is not the agreed schema."""
    with pytest.raises(JSONParseError):
        parse_json_object("[1, 2, 3]")


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------
def valid_question(**overrides: object) -> dict:
    """Build a schema-valid quiz question, with optional field overrides."""
    question = {
        "question": "Why does gradient descent diverge with a large learning rate?",
        "options": ["It overshoots", "It underflows", "It caches", "It normalises"],
        "correct_index": 0,
        "explanation": "Steps overshoot the minimum.",
        "source_page": 4,
    }
    question.update(overrides)
    return question


def test_validate_quiz_accepts_valid_payload() -> None:
    """A well-formed payload is returned normalised."""
    questions = validate_quiz({"questions": [valid_question()]})
    assert len(questions) == 1
    assert questions[0]["correct_index"] == 0
    assert questions[0]["source_page"] == 4


def test_validate_quiz_rejects_wrong_option_count() -> None:
    """Exactly four options are required."""
    with pytest.raises(SchemaValidationError):
        validate_quiz({"questions": [valid_question(options=["a", "b", "c"])]})


def test_validate_quiz_rejects_out_of_range_index() -> None:
    """correct_index must point at one of the four options."""
    with pytest.raises(SchemaValidationError):
        validate_quiz({"questions": [valid_question(correct_index=9)]})


def test_validate_quiz_rejects_empty_list() -> None:
    """An empty question list is a failure, not an empty quiz."""
    with pytest.raises(SchemaValidationError):
        validate_quiz({"questions": []})


def test_validate_quiz_coerces_string_page() -> None:
    """A page given as "p. 12" is still usable."""
    questions = validate_quiz({"questions": [valid_question(source_page="p. 12")]})
    assert questions[0]["source_page"] == 12


def test_validate_flashcards_drops_duplicates() -> None:
    """Cards with the same front (ignoring case) are de-duplicated."""
    cards = validate_flashcards(
        {
            "cards": [
                {"front": "What is entropy?", "back": "A measure of disorder.", "source_page": 2},
                {"front": "what is ENTROPY?", "back": "Duplicate.", "source_page": 2},
                {"front": "What is enthalpy?", "back": "Heat content.", "source_page": 3},
            ]
        }
    )
    assert len(cards) == 2
    assert [card["front"] for card in cards] == ["What is entropy?", "What is enthalpy?"]


def test_validate_flashcards_rejects_missing_back() -> None:
    """A card without an answer is rejected."""
    with pytest.raises(SchemaValidationError):
        validate_flashcards({"cards": [{"front": "Term", "back": "  "}]})


# ---------------------------------------------------------------------------
# Presentation helpers
# ---------------------------------------------------------------------------
def test_format_citations_groups_pages_by_source() -> None:
    """Citations are grouped per file and sorted by page."""
    line = format_citations(
        [
            {"source": "lecture3.pdf", "page": 7},
            {"source": "lecture3.pdf", "page": 4},
            {"source": "lecture3.pdf", "page": 4},
        ]
    )
    assert line == "Sources: lecture3.pdf (p. 4, p. 7)"


def test_format_citations_empty_input() -> None:
    """No metadata means no citation line."""
    assert format_citations([]) == ""


def test_anki_csv_has_two_columns() -> None:
    """The Anki export keeps only front and back."""
    csv_text = flashcards_to_anki_csv(
        [{"front": "Q", "back": "A", "source_page": 3}]
    )
    lines = csv_text.strip().splitlines()
    assert lines[0] == "front,back"
    assert lines[1] == "Q,A"
