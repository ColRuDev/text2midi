"""
REMI Vocabulary Mapping - Inverted vocabulary for token-level heuristics.

This module provides an independent inverted REMI vocabulary mapping
that maps TokenId (int) -> EventName (str) for fast token-level evaluation.

Architecture:
    - Pure domain module (no I/O, no external dependencies)
    - Vocabulary mapping is injected from adapter layer
    - Provides helper functions for token classification

The inverted vocabulary enables TokenHeuristics to parse tokens in memory
without disk I/O, dramatically improving evaluation speed during beam search.
"""

from __future__ import annotations

from typing import Optional


# Module-level vocabulary (set by adapter layer via set_inverted_vocab)
_INVERTED_VOCAB: Optional[dict[int, str]] = None


def set_inverted_vocab(vocab: dict[int, str]) -> None:
    """
    Set the inverted vocabulary mapping from the adapter layer.

    This MUST be called during application initialization before any
    token heuristics evaluation. The adapter layer is responsible for
    loading the vocabulary from disk.

    Args:
        vocab: Dict mapping TokenId (int) -> EventName (str).
    """
    global _INVERTED_VOCAB
    _INVERTED_VOCAB = vocab


def get_inverted_vocab() -> dict[int, str]:
    """
    Get the inverted vocabulary mapping.

    Returns:
        Dict mapping TokenId (int) -> EventName (str).

    Raises:
        RuntimeError: If vocabulary has not been initialized via set_inverted_vocab.
    """
    if _INVERTED_VOCAB is None:
        raise RuntimeError(
            "REMI vocabulary not initialized. Call set_inverted_vocab() from adapter layer."
        )
    return _INVERTED_VOCAB

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
    return get_inverted_vocab().get(token_id)


def is_pitch_token(token_id: int) -> bool:
    """
    Check if a token ID represents a Pitch event.

    Args:
        token_id: The TokenId to check.

    Returns:
        True if the token is a Pitch_X event, False otherwise.
    """
    event_name = get_inverted_vocab().get(token_id)
    return event_name is not None and event_name.startswith("Pitch_")


def is_program_token(token_id: int) -> bool:
    """
    Check if a token ID represents a Program (instrument) event.

    Args:
        token_id: The TokenId to check.

    Returns:
        True if the token is a Program_X event, False otherwise.
    """
    event_name = get_inverted_vocab().get(token_id)
    return event_name is not None and event_name.startswith("Program_")


def is_timesig_token(token_id: int) -> bool:
    """
    Check if a token ID represents a TimeSig event.

    Args:
        token_id: The TokenId to check.

    Returns:
        True if the token is a TimeSig_X/Y event, False otherwise.
    """
    event_name = get_inverted_vocab().get(token_id)
    return event_name is not None and event_name.startswith("TimeSig_")


def is_special_token(token_id: int) -> bool:
    """
    Check if a token ID represents a special token (non-musical).

    Args:
        token_id: The TokenId to check.

    Returns:
        True if the token is a special token (PAD, BOS, EOS, etc.), False otherwise.
    """
    event_name = get_inverted_vocab().get(token_id)
    return event_name is not None and event_name in SPECIAL_TOKENS
