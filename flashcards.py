"""Flashcard generation and export."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from langchain_chroma import Chroma

import prompts
from rag import LLMError, gather_context, generate_json
from utils import flashcards_to_anki_csv, get_logger, rows_to_csv, validate_flashcards

log = get_logger(__name__)

MIN_CARDS = 5
MAX_CARDS = 30

# Review status values used by the flip-card UI.
STATUS_UNSEEN = "unseen"
STATUS_KNOWN = "known"
STATUS_REVIEW = "review"


def context_budget(num_cards: int) -> int:
    """Choose how many chunks to feed the model for a given deck size.

    Args:
        num_cards: Number of cards requested.

    Returns:
        A chunk count large enough to source distinct concepts without flooding
        the prompt.
    """
    return max(10, min(30, num_cards))


def generate_flashcards(
    vectorstore: Chroma,
    topic: str | None,
    num_cards: int,
    sources: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Generate a validated, de-duplicated flashcard deck from the notes.

    Args:
        vectorstore: The Chroma collection to draw material from.
        topic: Optional topic; blank means "cover the selected documents broadly".
        num_cards: How many cards to request (clamped to 5-30).
        sources: Filenames selected in the sidebar.

    Returns:
        A list of validated card dictionaries with ``front``, ``back`` and
        ``source_page`` keys.

    Raises:
        LLMError: If no notes matched, or the model failed to return valid JSON.
    """
    num_cards = max(MIN_CARDS, min(MAX_CARDS, int(num_cards)))

    context, chunks = gather_context(
        vectorstore, topic, sources, limit=context_budget(num_cards)
    )
    if not context:
        raise LLMError(
            "I couldn't find anything to make flashcards from. Upload a PDF, select "
            "at least one document in the sidebar, or try a different topic."
        )
    log.info("Flashcards: %s chunks, %s cards requested", len(chunks), num_cards)

    system_prompt, user_prompt = prompts.render_flashcard_prompts(
        context=context, num_cards=num_cards, topic=topic
    )
    cards = generate_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        validator=validate_flashcards,
        schema_hint=prompts.FLASHCARD_SCHEMA_HINT,
    )
    return cards[:num_cards]


def status_counts(statuses: Mapping[int, str], total: int) -> dict[str, int]:
    """Count how many cards are marked known, for review, or still unseen.

    Args:
        statuses: Mapping of card index to a ``STATUS_*`` value.
        total: Total number of cards in the deck.

    Returns:
        A dictionary with ``known``, ``review`` and ``unseen`` counts.
    """
    known = sum(1 for value in statuses.values() if value == STATUS_KNOWN)
    review = sum(1 for value in statuses.values() if value == STATUS_REVIEW)
    return {"known": known, "review": review, "unseen": max(total - known - review, 0)}


def cards_to_anki_csv(cards: Sequence[Mapping[str, Any]]) -> str:
    """Export a deck as a two-column CSV that Anki imports directly.

    Args:
        cards: Validated card dictionaries.

    Returns:
        CSV text with a ``front,back`` header row.
    """
    return flashcards_to_anki_csv(cards)


def cards_to_full_csv(
    cards: Sequence[Mapping[str, Any]], statuses: Mapping[int, str] | None = None
) -> str:
    """Export a deck with page numbers and review status for personal records.

    Args:
        cards: Validated card dictionaries.
        statuses: Optional mapping of card index to a ``STATUS_*`` value.

    Returns:
        CSV text with ``front,back,source_page,status`` columns.
    """
    statuses = statuses or {}
    rows = [
        {
            "front": card["front"],
            "back": card["back"],
            "source_page": card.get("source_page", 0),
            "status": statuses.get(index, STATUS_UNSEEN),
        }
        for index, card in enumerate(cards)
    ]
    return rows_to_csv(rows, ["front", "back", "source_page", "status"])


__all__ = [
    "MAX_CARDS",
    "MIN_CARDS",
    "STATUS_KNOWN",
    "STATUS_REVIEW",
    "STATUS_UNSEEN",
    "cards_to_anki_csv",
    "cards_to_full_csv",
    "context_budget",
    "generate_flashcards",
    "status_counts",
]
