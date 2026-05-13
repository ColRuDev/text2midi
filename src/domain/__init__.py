"""
Domain layer - Core entities and interfaces.

This package contains the pure domain model with no external dependencies.
Import entities and interfaces from this module.
"""

from .entities import (
    Intent,
    GenerationProfile,
    MidiSequence,
    ClapPromptSource,
    TokenId,
    MidiBytes,
    PromptText,
)
from .interfaces import (
    LLMTranslator,
    MidiGenerator,
    Evaluator,
    AudioRenderer,
    AudioSamples,
)

__all__ = [
    # Entities
    "Intent",
    "GenerationProfile",
    "MidiSequence",
    "ClapPromptSource",
    # Type aliases
    "TokenId",
    "MidiBytes",
    "PromptText",
    "AudioSamples",
    # Interfaces
    "LLMTranslator",
    "MidiGenerator",
    "Evaluator",
    "AudioRenderer",
]
