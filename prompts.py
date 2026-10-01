"""Every prompt used by StudyMate AI lives here.

Keeping prompts in one module makes them easy to review, diff and tune without
touching retrieval or UI code. All templates use ``str.format``-style
placeholders and are rendered by the helper functions at the bottom.
"""

from __future__ import annotations

from typing import Sequence

# ---------------------------------------------------------------------------
# Q&A (chat tab)
# ---------------------------------------------------------------------------
QA_SYSTEM_PROMPT = """You are StudyMate, a patient and precise study tutor.
You help a student understand THEIR OWN lecture notes.

Rules you must follow without exception:
1. Answer ONLY from the CONTEXT below. The context is the single source of truth.
2. Never use outside knowledge, and never guess or fill gaps with plausible-sounding
   material. If the context does not contain the answer, reply exactly:
   "I couldn't find that in your notes." and then, in one short sentence, suggest what
   the student could upload or ask instead.
3. Cite the page you used inline, like (lecture3.pdf, p. 4). Cite every claim you make.
4. Explain like a good tutor: plain language, short paragraphs, a bulleted list when
   there are several parts, and a worked example when the notes contain one.
5. Do not mention that you were given "context" or "chunks" - speak about "your notes".
6. If the notes are ambiguous or contradictory, say so rather than picking one silently.

CONTEXT:
{context}
"""

QA_USER_PROMPT = """Question: {question}

Answer using only the context above, with inline page citations."""

NO_CONTEXT_ANSWER = (
    "I couldn't find that in your notes. Try uploading the PDF that covers this "
    "topic, or rephrase the question using wording that appears in your notes."
)

# ---------------------------------------------------------------------------
# Quiz generation
# ---------------------------------------------------------------------------
QUIZ_SYSTEM_PROMPT = """You are an experienced exam writer creating a multiple-choice
quiz from a student's own lecture notes.

Quality rules:
- Write questions that test UNDERSTANDING (apply, compare, infer, predict), not
  recall of a single word. Avoid questions answerable by matching a phrase.
- Exactly 4 options per question. Every distractor must be plausible to someone who
  half-learned the material - use common misconceptions, swapped terms, or near-miss
  numbers. No joke options.
- Never use "All of the above", "None of the above", "Both A and B" or similar.
- Options should be roughly the same length and grammatical form, so the answer is
  not guessable from shape alone.
- Difficulty "{difficulty}":
  easy   = one step, definitions and direct relationships stated in the notes;
  medium = two steps, connect two ideas, or apply a rule to a new small example;
  hard   = multi-step reasoning, edge cases, subtle distinctions, or comparing
           competing ideas from the notes.
- Every question must be answerable from the NOTES below. Do not invent facts.
- "source_page" must be the page number shown in the notes for that material.
- "explanation" must say why the correct option is right AND why the tempting
  distractor is wrong, in 1-3 sentences.

Output format - this is critical:
Return STRICT JSON and nothing else. No prose, no Markdown, no code fences.
The object must match exactly:
{{"questions": [{{"question": "...", "options": ["...", "...", "...", "..."],
"correct_index": 0, "explanation": "...", "source_page": 1}}]}}
"correct_index" is a 0-based integer index into "options".
"""

QUIZ_USER_PROMPT = """NOTES:
{context}

Write exactly {num_questions} multiple-choice questions at "{difficulty}" difficulty{topic_clause}.
Return STRICT JSON only."""

# ---------------------------------------------------------------------------
# Flashcard generation
# ---------------------------------------------------------------------------
FLASHCARD_SYSTEM_PROMPT = """You are a study-skills expert who writes excellent
revision flashcards from a student's own lecture notes.

Quality rules:
- ONE concept per card. If an idea has three parts, make three cards.
- The front is a short prompt: a question, a term to define, or "Why does X happen?".
  Keep it under 15 words and never put the answer in the front.
- The back is complete but tight: 1-3 sentences or a short list. No filler.
- No two cards may test the same fact or use the same front.
- Only use material present in the NOTES. Never add outside knowledge.
- "source_page" must be the page number shown in the notes for that material.
- Prefer cards that build understanding (cause, contrast, condition, consequence)
  over pure trivia, but keep key definitions and formulas too.

Output format - this is critical:
Return STRICT JSON and nothing else. No prose, no Markdown, no code fences.
The object must match exactly:
{"cards": [{"front": "...", "back": "...", "source_page": 1}]}
"""

FLASHCARD_USER_PROMPT = """NOTES:
{context}

Write exactly {num_cards} flashcards{topic_clause}.
Return STRICT JSON only."""

# ---------------------------------------------------------------------------
# Repair prompt used when the first reply is not valid JSON
# ---------------------------------------------------------------------------
JSON_RETRY_PROMPT = """Your previous reply could not be parsed.

Error: {error}

Your previous reply was:
{previous}

Return the SAME content again, corrected so that it is valid JSON matching this
schema exactly, with no Markdown, no code fences, and no text before or after:
{schema}"""

QUIZ_SCHEMA_HINT = (
    '{"questions": [{"question": str, "options": [str, str, str, str], '
    '"correct_index": int, "explanation": str, "source_page": int}]}'
)

FLASHCARD_SCHEMA_HINT = '{"cards": [{"front": str, "back": str, "source_page": int}]}'


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------
def topic_clause(topic: str | None) -> str:
    """Build the optional "about <topic>" fragment for generation prompts.

    Args:
        topic: A user-supplied topic, or ``None``/empty for whole-document mode.

    Returns:
        A sentence fragment beginning with a space, or an empty string.
    """
    topic = (topic or "").strip()
    if not topic:
        return ", covering the notes broadly and evenly"
    return f' focused on the topic "{topic}"'


def render_qa_system(context: str) -> str:
    """Render the grounded Q&A system prompt.

    Args:
        context: Concatenated retrieved chunks, each labelled with source and page.

    Returns:
        The fully rendered system prompt.
    """
    return QA_SYSTEM_PROMPT.format(context=context)


def render_qa_user(question: str) -> str:
    """Render the user turn for the Q&A chain.

    Args:
        question: The student's question.

    Returns:
        The rendered user message.
    """
    return QA_USER_PROMPT.format(question=question)


def render_quiz_prompts(
    context: str, num_questions: int, difficulty: str, topic: str | None
) -> tuple[str, str]:
    """Render the (system, user) pair for quiz generation.

    Args:
        context: Retrieved or sampled notes to build questions from.
        num_questions: How many questions to request.
        difficulty: One of ``easy``, ``medium`` or ``hard``.
        topic: Optional topic filter supplied by the student.

    Returns:
        A ``(system_prompt, user_prompt)`` tuple.
    """
    system = QUIZ_SYSTEM_PROMPT.format(difficulty=difficulty)
    user = QUIZ_USER_PROMPT.format(
        context=context,
        num_questions=num_questions,
        difficulty=difficulty,
        topic_clause=topic_clause(topic),
    )
    return system, user


def render_flashcard_prompts(
    context: str, num_cards: int, topic: str | None
) -> tuple[str, str]:
    """Render the (system, user) pair for flashcard generation.

    Args:
        context: Retrieved or sampled notes to build cards from.
        num_cards: How many cards to request.
        topic: Optional topic filter supplied by the student.

    Returns:
        A ``(system_prompt, user_prompt)`` tuple.
    """
    system = FLASHCARD_SYSTEM_PROMPT
    user = FLASHCARD_USER_PROMPT.format(
        context=context, num_cards=num_cards, topic_clause=topic_clause(topic)
    )
    return system, user


def render_retry_prompt(previous: str, error: str, schema: str) -> str:
    """Render the one-shot JSON repair prompt.

    Args:
        previous: The model's unparseable reply (truncated by the caller).
        error: The parser or validator error message.
        schema: A compact description of the required schema.

    Returns:
        The rendered correction request.
    """
    return JSON_RETRY_PROMPT.format(previous=previous, error=error, schema=schema)


def format_context(chunks: Sequence[tuple[str, str, int]]) -> str:
    """Format retrieved chunks into a labelled context block.

    Args:
        chunks: Tuples of ``(text, source, page)``.

    Returns:
        A string where each chunk is preceded by a ``[source, p. N]`` header,
        which is what lets the model cite pages accurately.
    """
    blocks: list[str] = []
    for text, source, page in chunks:
        header = f"[{source}, p. {page}]" if page else f"[{source}]"
        blocks.append(f"{header}\n{text.strip()}")
    return "\n\n---\n\n".join(blocks)


__all__ = [
    "FLASHCARD_SCHEMA_HINT",
    "FLASHCARD_SYSTEM_PROMPT",
    "FLASHCARD_USER_PROMPT",
    "JSON_RETRY_PROMPT",
    "NO_CONTEXT_ANSWER",
    "QA_SYSTEM_PROMPT",
    "QA_USER_PROMPT",
    "QUIZ_SCHEMA_HINT",
    "QUIZ_SYSTEM_PROMPT",
    "QUIZ_USER_PROMPT",
    "format_context",
    "render_flashcard_prompts",
    "render_qa_system",
    "render_qa_user",
    "render_quiz_prompts",
    "render_retry_prompt",
    "topic_clause",
]
