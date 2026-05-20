"""
REMI Vocabulary Mapping - Inverted vocabulary for token-level heuristics.

This module provides an independent inverted REMI vocabulary mapping
that maps TokenId (int) -> EventName (str) for fast token-level evaluation.

Architecture:
    - Pure domain module (no external dependencies beyond pickle for loading)
    - Provides inverted vocabulary for heuristics evaluation
    - Loads from the canonical REMI vocabulary file at initialization

The inverted vocabulary enables TokenHeuristics to parse tokens in memory
without disk I/O, dramatically improving evaluation speed during beam search.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

# Default path to the REMI vocabulary file
_DEFAULT_VOCAB_PATH = Path(__file__).parent.parent.parent / "models" / "text2midi" / "vocab_remi.pkl"


def _load_inverted_vocab(vocab_path: Path = _DEFAULT_VOCAB_PATH) -> dict[int, str]:
    """
    Load and invert the REMI vocabulary from the pickle file.

    Args:
        vocab_path: Path to the REMI vocabulary pickle file.

    Returns:
        A dict mapping TokenId (int) -> EventName (str).
    """
    if not vocab_path.exists():
        raise FileNotFoundError(f"REMI vocabulary file not found: {vocab_path}")

    with open(vocab_path, "rb") as f:
        remi_tokenizer = pickle.load(f)

    # The REMI tokenizer object has a .vocab attribute that is a dict
    # mapping EventName -> TokenId. We need to invert it.
    vocab = remi_tokenizer.vocab

    # Invert: TokenId -> EventName
    return {token_id: event_name for event_name, token_id in vocab.items()}


# Load the inverted vocabulary at module initialization
INVERTED_VOCAB: dict[int, str] = _load_inverted_vocab()

# Special token identifiers (tokens that don't represent musical events)
SPECIAL_TOKENS: set[str] = {
    "PAD_None",
    "BOS_None",
    "EOS_None",
    "MASK_None",
    "Bar_None",
}


def get_event_name(token_id: int) -> Optional[str]:
    """
    Get the event name for a given token ID.

    Args:
        token_id: The TokenId to look up.

    Returns:
        The event name string, or None if the token ID is not in the vocabulary.
    """
    return INVERTED_VOCAB.get(token_id)


def is_pitch_token(token_id: int) -> bool:
    """
    Check if a token ID represents a Pitch event.

    Args:
        token_id: The TokenId to check.

    Returns:
        True if the token is a Pitch_X event, False otherwise.
    """
    event_name = INVERTED_VOCAB.get(token_id)
    return event_name is not None and event_name.startswith("Pitch_")


def is_program_token(token_id: int) -> bool:
    """
    Check if a token ID represents a Program (instrument) event.

    Args:
        token_id: The TokenId to check.

    Returns:
        True if the token is a Program_X event, False otherwise.
    """
    event_name = INVERTED_VOCAB.get(token_id)
    return event_name is not None and event_name.startswith("Program_")


def is_timesig_token(token_id: int) -> bool:
    """
    Check if a token ID represents a TimeSig event.

    Args:
        token_id: The TokenId to check.

    Returns:
        True if the token is a TimeSig_X/Y event, False otherwise.
    """
    event_name = INVERTED_VOCAB.get(token_id)
    return event_name is not None and event_name.startswith("TimeSig_")


def is_special_token(token_id: int) -> bool:
    """
    Check if a token ID represents a special token (non-musical).

    Args:
        token_id: The TokenId to check.

    Returns:
        True if the token is a special token (PAD, BOS, EOS, etc.), False otherwise.
    """
    event_name = INVERTED_VOCAB.get(token_id)
    return event_name is not None and event_name in SPECIAL_TOKENS
