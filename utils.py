"""Shared helpers: logging, hashing, JSON extraction/validation, CSV export.

These functions are deliberately free of Streamlit and LangChain imports so
they can be unit tested without a UI or an API key.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import re
from typing import Any, Iterable, Mapping, Sequence

from config import settings

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def get_logger(name: str = "studymate") -> logging.Logger:
    """Return a configured logger, attaching a stream handler only once.

    Args:
        name: Logger name, typically the module name.

    Returns:
        A :class:`logging.Logger` ready for use.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
        logger.propagate = False
    return logger


log = get_logger(__name__)


class JSONParseError(ValueError):
    """Raised when the model's reply cannot be parsed as a JSON object."""


class SchemaValidationError(ValueError):
    """Raised when parsed JSON does not match the expected schema."""


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------
def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 digest of ``data``.

    Args:
        data: Raw bytes, e.g. the contents of an uploaded PDF.

    Returns:
        A 64-character lowercase hex digest, used to de-duplicate ingestion.
    """
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# JSON handling
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*|\s*```\s*$")


def strip_code_fences(text: str) -> str:
    """Remove surrounding Markdown code fences from a model reply.

    Args:
        text: Raw model output, possibly wrapped in fenced code blocks.

    Returns:
        The text with leading/trailing fences removed and whitespace trimmed.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = _FENCE_RE.sub("", cleaned)
    return cleaned.strip()


def extract_json_object(text: str) -> str:
    """Extract the outermost brace-delimited block from a string.

    Useful when the model prefixes its JSON with prose despite instructions.

    Args:
        text: Text that contains a JSON object somewhere inside it.

    Returns:
        The substring spanning the first opening brace to the last closing brace.

    Raises:
        JSONParseError: If no braces are present.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise JSONParseError("No JSON object found in the model response.")
    return text[start : end + 1]


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a model reply into a JSON object, tolerating fences and prose.

    Args:
        text: Raw model output.

    Returns:
        The decoded JSON object as a dictionary.

    Raises:
        JSONParseError: If the text cannot be decoded, or decodes to a non-object.
    """
    if not text or not text.strip():
        raise JSONParseError("The model returned an empty response.")

    candidates = [strip_code_fences(text)]
    try:
        candidates.append(extract_json_object(candidates[0]))
    except JSONParseError:
        pass

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as exc:  # try the next candidate
            last_error = exc
            continue
        if not isinstance(data, dict):
            raise JSONParseError(f"Expected a JSON object, got {type(data).__name__}.")
        return data

    raise JSONParseError(f"Could not decode JSON: {last_error}")


def _coerce_page(value: Any) -> int:
    """Best-effort conversion of a model-supplied page number to ``int``.

    Args:
        value: Whatever the model put in ``source_page``.

    Returns:
        The page number, or ``0`` when it is missing or unparseable.
    """
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float):
        return max(int(value), 0)
    if isinstance(value, str):
        match = re.search(r"\d+", value)
        if match:
            return int(match.group())
    return 0


def validate_quiz(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate the quiz JSON schema and normalise its fields.

    Expected shape::

        {"questions": [{"question": str, "options": [4 strings],
                        "correct_index": int, "explanation": str,
                        "source_page": int}]}

    Args:
        data: Parsed JSON object from the model.

    Returns:
        A list of validated question dictionaries.

    Raises:
        SchemaValidationError: If the payload is structurally wrong.
    """
    questions = data.get("questions")
    if not isinstance(questions, list) or not questions:
        raise SchemaValidationError(
            "Expected a non-empty 'questions' array at the top level."
        )

    validated: list[dict[str, Any]] = []
    for index, item in enumerate(questions, start=1):
        if not isinstance(item, Mapping):
            raise SchemaValidationError(f"Question {index} is not an object.")

        question = str(item.get("question", "")).strip()
        if not question:
            raise SchemaValidationError(f"Question {index} has an empty 'question'.")

        raw_options = item.get("options")
        if not isinstance(raw_options, Sequence) or isinstance(raw_options, (str, bytes)):
            raise SchemaValidationError(f"Question {index} has no 'options' array.")
        options = [str(opt).strip() for opt in raw_options if str(opt).strip()]
        if len(options) != 4:
            raise SchemaValidationError(
                f"Question {index} must have exactly 4 options, got {len(options)}."
            )

        correct_index = item.get("correct_index")
        if isinstance(correct_index, bool) or not isinstance(correct_index, (int, float)):
            raise SchemaValidationError(
                f"Question {index} has a non-numeric 'correct_index'."
            )
        correct_index = int(correct_index)
        if not 0 <= correct_index <= 3:
            raise SchemaValidationError(
                f"Question {index} has 'correct_index' {correct_index}, expected 0-3."
            )

        validated.append(
            {
                "question": question,
                "options": options,
                "correct_index": correct_index,
                "explanation": str(item.get("explanation", "")).strip()
                or "No explanation provided.",
                "source_page": _coerce_page(item.get("source_page")),
            }
        )
    return validated


def validate_flashcards(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate the flashcard JSON schema and drop duplicate fronts.

    Expected shape::

        {"cards": [{"front": str, "back": str, "source_page": int}]}

    Args:
        data: Parsed JSON object from the model.

    Returns:
        A list of validated, de-duplicated card dictionaries.

    Raises:
        SchemaValidationError: If the payload is structurally wrong.
    """
    cards = data.get("cards")
    if not isinstance(cards, list) or not cards:
        raise SchemaValidationError("Expected a non-empty 'cards' array at the top level.")

    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(cards, start=1):
        if not isinstance(item, Mapping):
            raise SchemaValidationError(f"Card {index} is not an object.")
        front = str(item.get("front", "")).strip()
        back = str(item.get("back", "")).strip()
        if not front or not back:
            raise SchemaValidationError(f"Card {index} is missing 'front' or 'back'.")
        key = front.casefold()
        if key in seen:
            continue
        seen.add(key)
        validated.append(
            {
                "front": front,
                "back": back,
                "source_page": _coerce_page(item.get("source_page")),
            }
        )

    if not validated:
        raise SchemaValidationError("No usable cards after de-duplication.")
    return validated


# ---------------------------------------------------------------------------
# Presentation helpers
# ---------------------------------------------------------------------------
def format_citations(metadatas: Iterable[Mapping[str, Any]]) -> str:
    """Build a human-readable citation line from chunk metadata.

    Args:
        metadatas: Metadata mappings containing ``source`` and ``page`` keys.

    Returns:
        A string such as ``"Sources: lecture3.pdf (p. 4, p. 7)"``, or an empty
        string when no metadata was supplied.
    """
    pages_by_source: dict[str, set[int]] = {}
    for meta in metadatas:
        source = str(meta.get("source", "unknown"))
        page = _coerce_page(meta.get("page"))
        pages_by_source.setdefault(source, set()).add(page)

    if not pages_by_source:
        return ""

    parts: list[str] = []
    for source in sorted(pages_by_source):
        pages = sorted(p for p in pages_by_source[source] if p > 0)
        if pages:
            parts.append(f"{source} (" + ", ".join(f"p. {p}" for p in pages) + ")")
        else:
            parts.append(source)
    return "Sources: " + "; ".join(parts)


def rows_to_csv(rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> str:
    """Serialise dictionaries to CSV text with a header row.

    Args:
        rows: The records to write.
        fieldnames: Column order; keys missing from a row are written blank.

    Returns:
        The CSV document as a string.
    """
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=list(fieldnames), extrasaction="ignore", lineterminator="\n"
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
    return buffer.getvalue()


def flashcards_to_anki_csv(cards: Sequence[Mapping[str, Any]]) -> str:
    """Export flashcards as a two-column CSV that Anki can import directly.

    Args:
        cards: Validated flashcard dictionaries.

    Returns:
        CSV text with ``front,back`` columns and no extra fields.
    """
    return rows_to_csv(
        [{"front": c["front"], "back": c["back"]} for c in cards], ["front", "back"]
    )


def friendly_api_error(exc: BaseException) -> str:
    """Translate an exception from an Anthropic call into user-facing advice.

    Args:
        exc: The exception raised while calling the model.

    Returns:
        A short, actionable message safe to show in the UI.
    """
    try:  # pragma: no cover - guarded so tests run without the SDK installed
        import anthropic
    except ImportError:  # pragma: no cover
        anthropic = None  # type: ignore[assignment]

    if anthropic is not None:
        if isinstance(exc, anthropic.AuthenticationError):
            return (
                "Anthropic rejected your API key. Check `ANTHROPIC_API_KEY` in your "
                ".env file and that the key is still active."
            )
        if isinstance(exc, anthropic.NotFoundError):
            return (
                f"The model `{settings.model}` was not found for this account. "
                "Set a valid `ANTHROPIC_MODEL` in your .env file."
            )
        if isinstance(exc, anthropic.RateLimitError):
            return (
                "Rate limit reached. Wait a few seconds and try again, or ask for "
                "fewer questions/cards."
            )
        if isinstance(exc, anthropic.APIConnectionError):
            return (
                "Could not reach the Anthropic API. Check your internet connection or "
                "proxy settings and try again."
            )
        if isinstance(exc, anthropic.APIStatusError):
            status = getattr(exc, "status_code", "unknown")
            return f"The Anthropic API returned an error (HTTP {status}): {exc}"

    log.exception("Unhandled error during model call")
    return f"Something went wrong while contacting the model: {exc}"


__all__ = [
    "JSONParseError",
    "SchemaValidationError",
    "extract_json_object",
    "flashcards_to_anki_csv",
    "format_citations",
    "friendly_api_error",
    "get_logger",
    "parse_json_object",
    "rows_to_csv",
    "sha256_bytes",
    "strip_code_fences",
    "validate_flashcards",
    "validate_quiz",
]
