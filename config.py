"""Central configuration for StudyMate AI.

All tunables and secrets are read from environment variables (usually via a
local ``.env`` file). Nothing in this project hardcodes a key or a model name.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env once, at import time. ``override=False`` means real environment
# variables win over the file, which is what you want in deployment.
load_dotenv(override=False)

PROJECT_ROOT: Path = Path(__file__).resolve().parent


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable, falling back to ``default``.

    Args:
        name: Environment variable name.
        default: Value to use when the variable is unset or not an integer.

    Returns:
        The parsed integer value.
    """
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable application settings resolved from the environment."""

    # --- Anthropic ---------------------------------------------------------
    anthropic_api_key: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "").strip()
    )
    model: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5").strip()
        or "claude-sonnet-5"
    )
    max_tokens: int = field(default_factory=lambda: _env_int("MAX_TOKENS", 4096))

    # --- Retrieval / chunking ---------------------------------------------
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 1000))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 150))
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 5))
    history_turns: int = field(default_factory=lambda: _env_int("HISTORY_TURNS", 4))

    # --- Storage -----------------------------------------------------------
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ).strip()
    )
    chroma_dir: str = field(
        default_factory=lambda: os.getenv(
            "CHROMA_DIR", str(PROJECT_ROOT / "chroma_db")
        ).strip()
    )
    collection_name: str = field(
        default_factory=lambda: os.getenv("CHROMA_COLLECTION", "studymate_notes").strip()
    )

    # --- Misc --------------------------------------------------------------
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").strip())

    @property
    def has_api_key(self) -> bool:
        """Whether an Anthropic API key is configured."""
        return bool(self.anthropic_api_key)

    @property
    def chroma_path(self) -> Path:
        """The Chroma persistence directory as a :class:`~pathlib.Path`."""
        return Path(self.chroma_dir)


settings = Settings()

APP_TITLE = "StudyMate AI"
APP_ICON = "\N{BOOKS}"
APP_TAGLINE = "Upload your notes. Ask, quiz, and revise \u2014 grounded in your own PDFs."

MISSING_KEY_MESSAGE = (
    "No `ANTHROPIC_API_KEY` found. Copy `.env.example` to `.env`, paste your key "
    "from https://console.anthropic.com/settings/keys, then restart the app."
)

__all__ = [
    "APP_ICON",
    "APP_TAGLINE",
    "APP_TITLE",
    "MISSING_KEY_MESSAGE",
    "PROJECT_ROOT",
    "Settings",
    "settings",
]
