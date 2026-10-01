"""Retrieval, grounded question answering, and the shared JSON generation loop.

This module owns every call to the Claude API. UI code calls these functions and
renders the dataclasses they return; it never talks to the model directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

import prompts
from config import settings
from utils import (
    JSONParseError,
    SchemaValidationError,
    format_citations,
    friendly_api_error,
    get_logger,
    parse_json_object,
)

log = get_logger(__name__)


class LLMError(RuntimeError):
    """A model call failed; the message is safe to show to the user."""


@dataclass
class Answer:
    """A grounded answer plus the evidence used to produce it."""

    text: str
    citations: str = ""
    chunks: list[Document] = field(default_factory=list)
    grounded: bool = True


# ---------------------------------------------------------------------------
# Model access
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_llm(model: str, max_tokens: int) -> ChatAnthropic:
    """Build a cached Claude chat model.

    Args:
        model: Anthropic model id, e.g. ``claude-sonnet-5``.
        max_tokens: Maximum tokens the model may generate per reply.

    Returns:
        A configured :class:`~langchain_anthropic.ChatAnthropic` client.

    Raises:
        LLMError: If no API key is configured.

    Note:
        ``temperature`` is intentionally not set. Current Claude models reject
        sampling parameters, and deterministic-by-default output is what we want
        for quizzes and flashcards anyway.
    """
    if not settings.has_api_key:
        raise LLMError(
            "No Anthropic API key found. Add ANTHROPIC_API_KEY to your .env file "
            "and restart the app."
        )
    return ChatAnthropic(
        model=model,
        api_key=settings.anthropic_api_key,
        max_tokens=max_tokens,
        timeout=120,
        max_retries=2,
    )


def default_llm() -> ChatAnthropic:
    """Return the chat model configured in ``.env``.

    Returns:
        The cached :class:`ChatAnthropic` instance for the configured model.

    Raises:
        LLMError: If no API key is configured.
    """
    return get_llm(settings.model, settings.max_tokens)


def invoke_text(llm: ChatAnthropic, messages: Sequence[BaseMessage]) -> str:
    """Call the model and return its reply as plain text.

    Args:
        llm: The chat model to call.
        messages: The full message list, system prompt first.

    Returns:
        The assistant reply as a string.

    Raises:
        LLMError: Wrapping any API, auth, rate-limit or network failure with a
            message that can be shown directly in the UI.
    """
    try:
        response = llm.invoke(list(messages))
    except Exception as exc:  # noqa: BLE001 - re-raised as a friendly LLMError
        raise LLMError(friendly_api_error(exc)) from exc

    content = response.content
    if isinstance(content, str):
        return content
    # Claude may return a list of content blocks; keep the text ones.
    parts: list[str] = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, Mapping) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "\n".join(parts).strip()


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def build_filter(sources: Sequence[str] | None) -> dict[str, Any] | None:
    """Build a Chroma metadata filter restricting retrieval to chosen documents.

    Args:
        sources: Filenames the student selected in the sidebar, or ``None``/empty
            to search everything.

    Returns:
        A Chroma ``where`` clause, or ``None`` when no filtering is needed.
    """
    if not sources:
        return None
    if len(sources) == 1:
        return {"source": sources[0]}
    return {"source": {"$in": list(sources)}}


def retrieve(
    vectorstore: Chroma,
    query: str,
    sources: Sequence[str] | None = None,
    k: int | None = None,
) -> list[Document]:
    """Fetch the most relevant chunks for a query.

    Args:
        vectorstore: The Chroma collection.
        query: The student's question or topic.
        sources: Optional filenames to restrict the search to.
        k: Number of chunks to return; defaults to ``settings.top_k``.

    Returns:
        The retrieved chunks, most similar first (empty on failure).
    """
    try:
        return vectorstore.similarity_search(
            query, k=k or settings.top_k, filter=build_filter(sources)
        )
    except Exception as exc:  # pragma: no cover - store errors are non-fatal
        log.exception("Retrieval failed")
        raise LLMError(f"Could not search your notes: {exc}") from exc


def sample_chunks(
    vectorstore: Chroma, sources: Sequence[str] | None = None, limit: int = 20
) -> list[Document]:
    """Sample chunks spread evenly across the selected documents.

    Used for whole-document quiz and flashcard generation, where there is no
    topic to search for and even coverage matters more than similarity.

    Args:
        vectorstore: The Chroma collection.
        sources: Optional filenames to restrict sampling to.
        limit: Maximum number of chunks to return.

    Returns:
        Chunks sampled at a regular stride through each document, in page order.
    """
    try:
        record = vectorstore.get(
            where=build_filter(sources), include=["documents", "metadatas"]
        )
    except Exception as exc:  # pragma: no cover
        raise LLMError(f"Could not read your notes: {exc}") from exc

    texts = record.get("documents") or []
    metadatas = record.get("metadatas") or []
    documents = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(texts, metadatas)
    ]
    if not documents:
        return []

    documents.sort(
        key=lambda d: (str(d.metadata.get("source", "")), int(d.metadata.get("page", 0)))
    )
    if len(documents) <= limit:
        return documents

    stride = len(documents) / limit
    return [documents[int(i * stride)] for i in range(limit)]


def context_from_documents(documents: Sequence[Document]) -> str:
    """Render retrieved chunks into a labelled context block for the prompt.

    Args:
        documents: Retrieved or sampled chunks.

    Returns:
        The formatted context string, with a ``[source, p. N]`` header per chunk.
    """
    return prompts.format_context(
        [
            (
                doc.page_content,
                str(doc.metadata.get("source", "unknown")),
                int(doc.metadata.get("page", 0) or 0),
            )
            for doc in documents
        ]
    )


# ---------------------------------------------------------------------------
# Grounded question answering
# ---------------------------------------------------------------------------
def history_messages(
    history: Sequence[Mapping[str, str]], turns: int | None = None
) -> list[BaseMessage]:
    """Convert recent chat history into LangChain messages.

    Args:
        history: Chat entries with ``role`` (``user``/``assistant``) and ``content``.
        turns: How many user/assistant pairs to keep; defaults to
            ``settings.history_turns``.

    Returns:
        A list of messages, oldest first, so follow-up questions have context.
    """
    keep = (turns if turns is not None else settings.history_turns) * 2
    messages: list[BaseMessage] = []
    for entry in list(history)[-keep:] if keep > 0 else []:
        role = entry.get("role")
        content = str(entry.get("content", "")).strip()
        if not content:
            continue
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def answer_question(
    vectorstore: Chroma,
    question: str,
    sources: Sequence[str] | None = None,
    history: Sequence[Mapping[str, str]] | None = None,
    k: int | None = None,
) -> Answer:
    """Answer a question strictly from the student's notes.

    Args:
        vectorstore: The Chroma collection to search.
        question: The student's question.
        sources: Filenames selected in the sidebar, or ``None`` for all.
        history: Recent chat entries used for follow-up context.
        k: Number of chunks to retrieve; defaults to ``settings.top_k``.

    Returns:
        An :class:`Answer` with the reply text, a citation line and the chunks used.

    Raises:
        LLMError: If retrieval or the model call fails.
    """
    chunks = retrieve(vectorstore, question, sources=sources, k=k)
    if not chunks:
        return Answer(text=prompts.NO_CONTEXT_ANSWER, grounded=False)

    messages: list[BaseMessage] = [
        SystemMessage(content=prompts.render_qa_system(context_from_documents(chunks)))
    ]
    messages.extend(history_messages(history or []))
    messages.append(HumanMessage(content=prompts.render_qa_user(question)))

    text = invoke_text(default_llm(), messages)
    return Answer(
        text=text or prompts.NO_CONTEXT_ANSWER,
        citations=format_citations([doc.metadata for doc in chunks]),
        chunks=list(chunks),
        grounded=True,
    )


# ---------------------------------------------------------------------------
# Structured (JSON) generation shared by the quiz and flashcard tabs
# ---------------------------------------------------------------------------
def generate_json(
    system_prompt: str,
    user_prompt: str,
    validator: Callable[[Mapping[str, Any]], list[dict[str, Any]]],
    schema_hint: str,
    llm: ChatAnthropic | None = None,
) -> list[dict[str, Any]]:
    """Ask the model for strict JSON, validate it, and retry once on failure.

    Args:
        system_prompt: The rendered system prompt.
        user_prompt: The rendered user prompt.
        validator: A function that validates the parsed object and returns the
            normalised list of items (raises on a schema mismatch).
        schema_hint: A compact schema description included in the retry prompt.
        llm: Optional model override; defaults to the configured model.

    Returns:
        The validated list of items.

    Raises:
        LLMError: If the model call fails, or if the second attempt is still not
            valid JSON matching the schema.
    """
    model = llm or default_llm()
    messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    raw = invoke_text(model, messages)
    try:
        return validator(parse_json_object(raw))
    except (JSONParseError, SchemaValidationError) as exc:
        # `exc` is unbound once the except block ends, so keep the text.
        first_error = str(exc)
        log.warning("First JSON attempt failed: %s", first_error)

    retry_messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
        HumanMessage(
            content=prompts.render_retry_prompt(
                previous=raw[:4000], error=first_error, schema=schema_hint
            )
        ),
    ]
    retried = invoke_text(model, retry_messages)
    try:
        return validator(parse_json_object(retried))
    except (JSONParseError, SchemaValidationError) as second_error:
        log.error("Retry also failed: %s", second_error)
        raise LLMError(
            "The model did not return valid JSON, even after a correction attempt "
            f"({second_error}). Try again, or reduce the number of items requested."
        ) from second_error


def gather_context(
    vectorstore: Chroma,
    topic: str | None,
    sources: Sequence[str] | None,
    limit: int,
) -> tuple[str, list[Document]]:
    """Collect the notes used for quiz or flashcard generation.

    A topic triggers similarity search; a blank topic samples evenly across the
    selected documents so the whole file is represented.

    Args:
        vectorstore: The Chroma collection.
        topic: Optional topic supplied by the student.
        sources: Filenames selected in the sidebar.
        limit: Maximum number of chunks to include.

    Returns:
        A ``(context_string, chunks)`` tuple; the string is empty when nothing
        was found.

    Raises:
        LLMError: If reading the vector store fails.
    """
    topic = (topic or "").strip()
    chunks = (
        retrieve(vectorstore, topic, sources=sources, k=limit)
        if topic
        else sample_chunks(vectorstore, sources=sources, limit=limit)
    )
    if not chunks:
        return "", []
    return context_from_documents(chunks), list(chunks)


__all__ = [
    "Answer",
    "LLMError",
    "answer_question",
    "build_filter",
    "context_from_documents",
    "default_llm",
    "gather_context",
    "generate_json",
    "get_llm",
    "history_messages",
    "invoke_text",
    "retrieve",
    "sample_chunks",
]
