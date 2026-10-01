"""Quiz generation, scoring and CSV export."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from langchain_chroma import Chroma

import prompts
from rag import LLMError, gather_context, generate_json
from utils import get_logger, rows_to_csv, validate_quiz

log = get_logger(__name__)

DIFFICULTIES: tuple[str, ...] = ("easy", "medium", "hard")
MIN_QUESTIONS = 3
MAX_QUESTIONS = 15
OPTION_LABELS = ("A", "B", "C", "D")


@dataclass
class QuizResult:
    """The outcome of grading a submitted quiz."""

    score: int = 0
    total: int = 0
    rows: list[dict[str, Any]] = field(default_factory=list)

    @property
    def percentage(self) -> float:
        """Score as a percentage, or ``0.0`` for an empty quiz."""
        return round(100 * self.score / self.total, 1) if self.total else 0.0

    @property
    def headline(self) -> str:
        """A short summary line such as ``"7 / 10 correct (70.0%)"``."""
        return f"{self.score} / {self.total} correct ({self.percentage}%)"


def context_budget(num_questions: int) -> int:
    """Choose how many chunks to feed the model for a given quiz length.

    Args:
        num_questions: Number of questions requested.

    Returns:
        A chunk count that gives the model enough material without flooding it.
    """
    return max(8, min(24, num_questions * 2))


def generate_quiz(
    vectorstore: Chroma,
    topic: str | None,
    num_questions: int,
    difficulty: str,
    sources: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Generate a validated multiple-choice quiz from the student's notes.

    Args:
        vectorstore: The Chroma collection to draw material from.
        topic: Optional topic; blank means "cover the selected documents broadly".
        num_questions: How many questions to request (clamped to 3-15).
        difficulty: One of ``easy``, ``medium`` or ``hard``.
        sources: Filenames selected in the sidebar.

    Returns:
        A list of validated question dictionaries.

    Raises:
        LLMError: If no notes matched, or the model failed to return valid JSON.
    """
    num_questions = max(MIN_QUESTIONS, min(MAX_QUESTIONS, int(num_questions)))
    difficulty = difficulty if difficulty in DIFFICULTIES else "medium"

    context, chunks = gather_context(
        vectorstore, topic, sources, limit=context_budget(num_questions)
    )
    if not context:
        raise LLMError(
            "I couldn't find anything to build a quiz from. Upload a PDF, select at "
            "least one document in the sidebar, or try a different topic."
        )
    log.info("Quiz: %s chunks, %s questions, %s", len(chunks), num_questions, difficulty)

    system_prompt, user_prompt = prompts.render_quiz_prompts(
        context=context,
        num_questions=num_questions,
        difficulty=difficulty,
        topic=topic,
    )
    questions = generate_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        validator=validate_quiz,
        schema_hint=prompts.QUIZ_SCHEMA_HINT,
    )
    return questions[:num_questions]


def score_quiz(
    questions: Sequence[Mapping[str, Any]], answers: Mapping[int, int | None]
) -> QuizResult:
    """Grade the student's answers.

    Args:
        questions: The validated questions that were displayed.
        answers: Mapping of question index to the option index chosen, where
            ``None`` means unanswered.

    Returns:
        A :class:`QuizResult` with the score and per-question rows suitable for
        both on-screen review and CSV export.
    """
    result = QuizResult(total=len(questions))
    for index, question in enumerate(questions):
        chosen = answers.get(index)
        correct_index = int(question["correct_index"])
        options = list(question["options"])
        is_correct = chosen is not None and int(chosen) == correct_index
        if is_correct:
            result.score += 1
        result.rows.append(
            {
                "number": index + 1,
                "question": question["question"],
                "your_answer": options[chosen] if chosen is not None else "(no answer)",
                "correct_answer": options[correct_index],
                "result": "correct" if is_correct else "incorrect",
                "explanation": question.get("explanation", ""),
                "source_page": question.get("source_page", 0),
            }
        )
    return result


def quiz_to_csv(result: QuizResult) -> str:
    """Serialise graded quiz rows to CSV.

    Args:
        result: The graded quiz.

    Returns:
        CSV text with one row per question.
    """
    return rows_to_csv(
        result.rows,
        [
            "number",
            "question",
            "your_answer",
            "correct_answer",
            "result",
            "explanation",
            "source_page",
        ],
    )


__all__ = [
    "DIFFICULTIES",
    "MAX_QUESTIONS",
    "MIN_QUESTIONS",
    "OPTION_LABELS",
    "QuizResult",
    "context_budget",
    "generate_quiz",
    "quiz_to_csv",
    "score_quiz",
]
