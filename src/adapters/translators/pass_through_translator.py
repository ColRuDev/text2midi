"""
Pass-Through Translator Adapter - Implements LLMTranslator as a null object.

This adapter implements the Null Object Pattern, passing the original
intent text through without any LLM translation. It's useful for:
- Bypassing translation when no LLM is desired
- Testing the pipeline without API calls
- Using the original intent text directly with downstream components

Architecture:
    - Implements domain.interfaces.LLMTranslator
    - No external dependencies (no API calls, no API keys)
    - Returns [intent.text] * num_variations
"""

from __future__ import annotations

from typing import List

from domain.entities import Intent, PromptText
from domain.interfaces import LLMTranslator


class PassThroughTranslator(LLMTranslator):
    """
    Null Object implementation of LLMTranslator.

    This translator passes the original intent text through unchanged,
    returning multiple copies as specified by num_variations.

    Use cases:
    - Bypass LLM translation when desired
    - Test pipeline without API calls
    - Use original intent text directly with MidiLLM (batch generation)

    Example:
        >>> translator = PassThroughTranslator()
        >>> prompts = translator.translate(Intent("A sunrise"), 3)
        >>> prompts
        ['A sunrise', 'A sunrise', 'A sunrise']
    """

    def translate(self, intent: Intent, num_variations: int) -> List[PromptText]:
        """
        Return the intent text unaltered, repeated num_variations times.

        Args:
            intent: The user's creative intent in natural language.
            num_variations: Number of prompt variations to generate.

        Returns:
            A list of the original intent text, repeated num_variations times.
            Empty list if num_variations=0.

        Raises:
            ValueError: If num_variations is negative.

        Example:
            >>> translator = PassThroughTranslator()
            >>> translator.translate(Intent("Jazz piano"), 2)
            ['Jazz piano', 'Jazz piano']
        """
        if num_variations < 0:
            raise ValueError(
                f"num_variations must be >= 0, got {num_variations}"
            )
        if num_variations == 0:
            return []

        return [intent.text] * num_variations
