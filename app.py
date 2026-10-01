"""StudyMate AI - Streamlit entry point.

Run with::

    streamlit run app.py
"""

from __future__ import annotations

from html import escape
from typing import Any, Sequence

import streamlit as st

import flashcards as fc
import ingest
import quiz as qz
from config import APP_ICON, APP_TAGLINE, APP_TITLE, MISSING_KEY_MESSAGE, settings
from rag import LLMError, answer_question
from utils import get_logger, sha256_bytes

log = get_logger(__name__)

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .block-container {padding-top: 2.2rem; max-width: 1200px;}
    .sm-hero h1 {margin-bottom: 0.1rem;}
    .sm-hero p {color: #6b7280; margin-top: 0;}
    .sm-citation {color: #6b7280; font-size: 0.85rem; margin-top: 0.4rem;}
    .sm-card {
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        min-height: 190px;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        font-size: 1.25rem;
        line-height: 1.5;
        background: rgba(128, 128, 128, 0.06);
    }
    .sm-card-back {background: rgba(16, 185, 129, 0.10);}
    .sm-chunk {
        border-left: 3px solid rgba(128, 128, 128, 0.35);
        padding-left: 0.8rem;
        margin-bottom: 0.9rem;
        font-size: 0.88rem;
        white-space: pre-wrap;
    }
</style>
"""


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_state() -> None:
    """Create every session-state key the UI relies on, once per session."""
    defaults: dict[str, Any] = {
        "messages": [],
        "quiz_questions": [],
        "quiz_meta": {},
        "quiz_submitted": False,
        "quiz_result": None,
        "cards": [],
        "card_index": 0,
        "card_revealed": False,
        "card_status": {},
        "ingest_report": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_quiz() -> None:
    """Clear the current quiz, including the per-question radio selections.

    The radio widgets are keyed ``quiz_answer_<n>``; without dropping those keys
    a new quiz would open with the previous quiz's answers pre-selected.
    """
    st.session_state.quiz_questions = []
    st.session_state.quiz_meta = {}
    st.session_state.quiz_submitted = False
    st.session_state.quiz_result = None
    for key in [k for k in st.session_state if str(k).startswith("quiz_answer_")]:
        del st.session_state[key]


def reset_cards() -> None:
    """Clear the current flashcard deck and its review progress."""
    st.session_state.cards = []
    st.session_state.card_index = 0
    st.session_state.card_revealed = False
    st.session_state.card_status = {}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar(vectorstore: Any) -> list[str]:
    """Render upload, document selection and maintenance controls.

    Args:
        vectorstore: The Chroma collection.

    Returns:
        The filenames the student selected for retrieval.
    """
    with st.sidebar:
        st.markdown(f"## {APP_ICON} {APP_TITLE}")
        st.caption(f"Model: `{settings.model}`")

        if not settings.has_api_key:
            st.error(MISSING_KEY_MESSAGE, icon="\N{NO ENTRY SIGN}")

        st.divider()
        st.markdown("### 1. Add your notes")
        uploads = st.file_uploader(
            "PDF notes",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            help="Text-based PDFs only. Scanned pages need OCR first.",
        )
        if uploads and st.button("Add to library", type="primary", use_container_width=True):
            run_ingestion(vectorstore, uploads)

        for line in st.session_state.ingest_report:
            level, message = line
            getattr(st, level)(message)

        st.divider()
        st.markdown("### 2. Choose what to study")
        documents = ingest.list_documents(vectorstore)
        if not documents:
            st.info("No documents yet. Upload a PDF to get started.")
            return []

        options = [doc.source for doc in documents]
        selected = st.multiselect(
            "Search within these documents",
            options=options,
            default=options,
            help="Retrieval is filtered to the documents you select here.",
        )
        for doc in documents:
            if doc.source in selected:
                st.caption(f"\N{PAGE FACING UP} **{doc.source}** - {doc.page_count} pages, {doc.chunks} chunks")

        st.divider()
        with st.expander("Danger zone"):
            confirm = st.checkbox("I understand this deletes everything")
            if st.button("Clear all data", disabled=not confirm, use_container_width=True):
                ingest.clear_all(vectorstore)
                st.session_state.messages = []
                st.session_state.ingest_report = []
                reset_quiz()
                reset_cards()
                st.success("Library cleared.")
                st.rerun()

        st.divider()
        st.caption(
            f"chunk_size={settings.chunk_size} · overlap={settings.chunk_overlap} "
            f"· top_k={settings.top_k}"
        )
        return selected


def run_ingestion(vectorstore: Any, uploads: Sequence[Any]) -> None:
    """Ingest uploaded PDFs with a progress bar and a per-file summary.

    Args:
        vectorstore: The Chroma collection to write to.
        uploads: Files returned by ``st.file_uploader``.
    """
    known = ingest.stored_hashes(vectorstore)
    report: list[tuple[str, str]] = []
    progress = st.progress(0.0, text="Starting...")
    total = len(uploads)

    for position, upload in enumerate(uploads):
        def on_progress(fraction: float, message: str, _pos: int = position) -> None:
            progress.progress((_pos + fraction) / total, text=message)

        try:
            data = upload.getvalue()
        except Exception as exc:  # pragma: no cover - Streamlit upload failure
            report.append(("error", f"Could not read '{upload.name}': {exc}"))
            continue

        result = ingest.ingest_pdf(
            vectorstore,
            data=data,
            filename=upload.name,
            known_hashes=known,
            progress_cb=on_progress,
        )
        if result.chunks_created:
            known.add(sha256_bytes(data))
            report.append(
                (
                    "success",
                    f"**{result.filename}** - {result.pages_with_text}/{result.pages_read} "
                    f"pages read, {result.chunks_created} chunks created.",
                )
            )
        if result.error:
            report.append(("error", result.error))
        elif result.warning:
            report.append(("warning", result.warning))

    progress.progress(1.0, text="Done")
    st.session_state.ingest_report = report
    st.rerun()


# ---------------------------------------------------------------------------
# Chat tab
# ---------------------------------------------------------------------------
def render_context_expander(chunks: Sequence[Any]) -> None:
    """Show the retrieved chunks behind an answer.

    Args:
        chunks: The documents that were sent to the model.
    """
    if not chunks:
        return
    with st.expander(f"View retrieved context ({len(chunks)} chunks)"):
        for position, doc in enumerate(chunks, start=1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            st.markdown(f"**{position}. {source} - p. {page}**")
            st.markdown(
                f"<div class='sm-chunk'>{escape(doc.page_content.strip())}</div>",
                unsafe_allow_html=True,
            )


def render_chat_tab(vectorstore: Any, sources: Sequence[str]) -> None:
    """Render the grounded chat interface.

    Args:
        vectorstore: The Chroma collection.
        sources: Selected document filenames.
    """
    st.subheader("Ask your notes")
    st.caption(
        "Answers come only from the documents selected in the sidebar, with page "
        "citations. If something is not in your notes, StudyMate says so."
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("citations"):
                st.markdown(
                    f"<div class='sm-citation'>{message['citations']}</div>",
                    unsafe_allow_html=True,
                )
            render_context_expander(message.get("chunks", []))

    question = st.chat_input("e.g. Explain backpropagation from lecture 3")
    if not question:
        return

    if not sources:
        st.warning("Select at least one document in the sidebar first.")
        return
    if not settings.has_api_key:
        st.error(MISSING_KEY_MESSAGE)
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your notes..."):
            try:
                answer = answer_question(
                    vectorstore,
                    question=question,
                    sources=list(sources),
                    history=st.session_state.messages[:-1],
                )
            except LLMError as exc:
                st.error(str(exc))
                st.session_state.messages.pop()
                return

        st.markdown(answer.text)
        if answer.citations:
            st.markdown(
                f"<div class='sm-citation'>{answer.citations}</div>",
                unsafe_allow_html=True,
            )
        render_context_expander(answer.chunks)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer.text,
            "citations": answer.citations,
            "chunks": answer.chunks,
        }
    )


# ---------------------------------------------------------------------------
# Quiz tab
# ---------------------------------------------------------------------------
def render_quiz_tab(vectorstore: Any, sources: Sequence[str]) -> None:
    """Render quiz generation, the interactive quiz, and its results.

    Args:
        vectorstore: The Chroma collection.
        sources: Selected document filenames.
    """
    st.subheader("Test yourself")

    with st.form("quiz_setup"):
        col1, col2, col3 = st.columns([3, 1, 1])
        topic = col1.text_input(
            "Topic (optional)", placeholder="Leave blank to cover the whole document"
        )
        count = col2.number_input(
            "Questions",
            min_value=qz.MIN_QUESTIONS,
            max_value=qz.MAX_QUESTIONS,
            value=5,
            step=1,
        )
        difficulty = col3.selectbox("Difficulty", qz.DIFFICULTIES, index=1)
        generate = st.form_submit_button("Generate quiz", type="primary")

    if generate:
        if not sources:
            st.warning("Select at least one document in the sidebar first.")
        elif not settings.has_api_key:
            st.error(MISSING_KEY_MESSAGE)
        else:
            reset_quiz()
            with st.spinner("Writing your quiz..."):
                try:
                    st.session_state.quiz_questions = qz.generate_quiz(
                        vectorstore,
                        topic=topic,
                        num_questions=int(count),
                        difficulty=difficulty,
                        sources=list(sources),
                    )
                    st.session_state.quiz_meta = {
                        "topic": topic.strip() or "whole document",
                        "difficulty": difficulty,
                    }
                except LLMError as exc:
                    st.error(str(exc))

    questions = st.session_state.quiz_questions
    if not questions:
        return

    meta = st.session_state.quiz_meta
    st.caption(
        f"{len(questions)} questions · {meta.get('difficulty', '')} · "
        f"{meta.get('topic', '')}"
    )

    if not st.session_state.quiz_submitted:
        with st.form("quiz_answers"):
            for index, question in enumerate(questions):
                st.markdown(f"**{index + 1}. {question['question']}**")
                st.radio(
                    "Choose one",
                    options=list(range(len(question["options"]))),
                    format_func=lambda i, q=question: (
                        f"{qz.OPTION_LABELS[i]}. {q['options'][i]}"
                    ),
                    key=f"quiz_answer_{index}",
                    index=None,
                    label_visibility="collapsed",
                )
                st.divider()
            if st.form_submit_button("Submit answers", type="primary"):
                answers = {
                    index: st.session_state.get(f"quiz_answer_{index}")
                    for index in range(len(questions))
                }
                st.session_state.quiz_result = qz.score_quiz(questions, answers)
                st.session_state.quiz_submitted = True
                st.rerun()
        return

    result: qz.QuizResult = st.session_state.quiz_result
    st.metric("Score", result.headline)
    st.progress(result.percentage / 100)

    for index, row in enumerate(result.rows):
        correct = row["result"] == "correct"
        icon = "\N{WHITE HEAVY CHECK MARK}" if correct else "\N{CROSS MARK}"
        with st.expander(f"{icon} {row['number']}. {row['question']}", expanded=not correct):
            st.markdown(f"**Your answer:** {row['your_answer']}")
            if not correct:
                st.markdown(f"**Correct answer:** {row['correct_answer']}")
            st.info(row["explanation"])
            if row.get("source_page"):
                st.caption(f"From p. {row['source_page']} of your notes")

    col1, col2 = st.columns(2)
    col1.download_button(
        "Download results (CSV)",
        data=qz.quiz_to_csv(result),
        file_name="studymate_quiz_results.csv",
        mime="text/csv",
        use_container_width=True,
    )
    if col2.button("New quiz", use_container_width=True):
        reset_quiz()
        st.rerun()


# ---------------------------------------------------------------------------
# Flashcards tab
# ---------------------------------------------------------------------------
def render_flashcards_tab(vectorstore: Any, sources: Sequence[str]) -> None:
    """Render flashcard generation, the flip-card reviewer, and exports.

    Args:
        vectorstore: The Chroma collection.
        sources: Selected document filenames.
    """
    st.subheader("Revise with flashcards")

    with st.form("flashcard_setup"):
        col1, col2 = st.columns([3, 1])
        topic = col1.text_input(
            "Topic (optional)", placeholder="Leave blank to cover the whole document"
        )
        count = col2.number_input(
            "Cards",
            min_value=fc.MIN_CARDS,
            max_value=fc.MAX_CARDS,
            value=10,
            step=1,
        )
        generate = st.form_submit_button("Generate flashcards", type="primary")

    if generate:
        if not sources:
            st.warning("Select at least one document in the sidebar first.")
        elif not settings.has_api_key:
            st.error(MISSING_KEY_MESSAGE)
        else:
            reset_cards()
            with st.spinner("Writing your deck..."):
                try:
                    st.session_state.cards = fc.generate_flashcards(
                        vectorstore,
                        topic=topic,
                        num_cards=int(count),
                        sources=list(sources),
                    )
                except LLMError as exc:
                    st.error(str(exc))

    cards = st.session_state.cards
    if not cards:
        return

    index = min(st.session_state.card_index, len(cards) - 1)
    st.session_state.card_index = index
    card = cards[index]

    counts = fc.status_counts(st.session_state.card_status, len(cards))
    col1, col2, col3 = st.columns(3)
    col1.metric("Know it", counts["known"])
    col2.metric("Review again", counts["review"])
    col3.metric("Not reviewed", counts["unseen"])

    st.caption(f"Card {index + 1} of {len(cards)}")
    if st.session_state.card_revealed:
        st.markdown(
            f"<div class='sm-card sm-card-back'>{escape(card['back'])}</div>",
            unsafe_allow_html=True,
        )
        if card.get("source_page"):
            st.caption(f"From p. {card['source_page']} of your notes")
    else:
        st.markdown(
            f"<div class='sm-card'>{escape(card['front'])}</div>", unsafe_allow_html=True
        )

    nav1, nav2, nav3 = st.columns(3)
    if nav1.button("← Previous", use_container_width=True, disabled=index == 0):
        st.session_state.card_index = max(index - 1, 0)
        st.session_state.card_revealed = False
        st.rerun()
    if nav2.button(
        "Hide answer" if st.session_state.card_revealed else "Reveal answer",
        use_container_width=True,
        type="primary",
    ):
        st.session_state.card_revealed = not st.session_state.card_revealed
        st.rerun()
    if nav3.button(
        "Next →", use_container_width=True, disabled=index >= len(cards) - 1
    ):
        st.session_state.card_index = min(index + 1, len(cards) - 1)
        st.session_state.card_revealed = False
        st.rerun()

    mark1, mark2 = st.columns(2)
    if mark1.button("\N{WHITE HEAVY CHECK MARK} Know it", use_container_width=True):
        st.session_state.card_status[index] = fc.STATUS_KNOWN
        st.session_state.card_index = min(index + 1, len(cards) - 1)
        st.session_state.card_revealed = False
        st.rerun()
    if mark2.button("\N{ANTICLOCKWISE OPEN CIRCLE ARROW} Review again", use_container_width=True):
        st.session_state.card_status[index] = fc.STATUS_REVIEW
        st.session_state.card_index = min(index + 1, len(cards) - 1)
        st.session_state.card_revealed = False
        st.rerun()

    st.divider()
    exp1, exp2 = st.columns(2)
    exp1.download_button(
        "Export for Anki (front,back)",
        data=fc.cards_to_anki_csv(cards),
        file_name="studymate_anki.csv",
        mime="text/csv",
        use_container_width=True,
    )
    exp2.download_button(
        "Export with pages + status",
        data=fc.cards_to_full_csv(cards, st.session_state.card_status),
        file_name="studymate_flashcards.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Compose the page: hero, sidebar, and the three feature tabs."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    init_state()
    st.markdown(
        f"<div class='sm-hero'><h1>{APP_ICON} {APP_TITLE}</h1><p>{APP_TAGLINE}</p></div>",
        unsafe_allow_html=True,
    )

    try:
        vectorstore = ingest.get_vectorstore()
    except Exception as exc:  # pragma: no cover - fatal, show and stop
        st.error(
            "Could not open the vector database at "
            f"`{settings.chroma_dir}`: {exc}\n\n"
            "Delete the folder and restart if it is corrupted."
        )
        st.stop()
        return

    sources = render_sidebar(vectorstore)

    chat_tab, quiz_tab, cards_tab = st.tabs(
        ["\N{SPEECH BALLOON} Chat", "\N{MEMO} Quiz", "\N{BRAIN} Flashcards"]
    )
    with chat_tab:
        render_chat_tab(vectorstore, sources)
    with quiz_tab:
        render_quiz_tab(vectorstore, sources)
    with cards_tab:
        render_flashcards_tab(vectorstore, sources)


if __name__ == "__main__":
    main()
