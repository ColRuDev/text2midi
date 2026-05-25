"""
Unit tests for PassThroughTranslator adapter.

These tests verify the pass-through behavior where the translator
returns the original intent text unaltered.
"""

import pytest

from adapters.translators.pass_through_translator import PassThroughTranslator
from domain.entities import Intent


class TestPassThroughTranslator:
    """Tests for PassThroughTranslator."""

    def test_translate_returns_list_with_intent_text(self):
        """
        Pass-through translation: translate MUST return a list containing
        exactly the original text intent.
        
        GIVEN a text intent
        WHEN the PassThroughTranslator is invoked
        THEN it MUST return a list containing exactly the original text intent
        """
        translator = PassThroughTranslator()
        intent = Intent(text="A peaceful sunrise at the beach")

        result = translator.translate(intent, num_variations=1)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == "A peaceful sunrise at the beach"

    def test_translate_returns_num_variations_copies(self):
        """
        Design decision: translate returns [intent.text] * num_variations.
        
        GIVEN num_variations is 3
        WHEN translate is called
        THEN it MUST return a list of length 3 with all elements equal to intent.text
        """
        translator = PassThroughTranslator()
        intent = Intent(text="Jazz piano melody")

        result = translator.translate(intent, num_variations=3)

        assert len(result) == 3
        assert all(p == "Jazz piano melody" for p in result)

    def test_translate_zero_variations_returns_empty_list(self):
        """
        Edge case: num_variations=0 returns empty list.
        
        This matches the contract of LLMTranslator.
        """
        translator = PassThroughTranslator()
        intent = Intent(text="Test intent")

        result = translator.translate(intent, num_variations=0)

        assert result == []

    def test_translate_negative_variations_raises_valueerror(self):
        """
        Edge case: num_variations < 0 raises ValueError.
        
        This matches the contract of LLMTranslator.
        """
        translator = PassThroughTranslator()
        intent = Intent(text="Test intent")

        with pytest.raises(ValueError) as exc_info:
            translator.translate(intent, num_variations=-1)

        assert "num_variations must be >= 0" in str(exc_info.value)

    def test_translator_is_subclass_of_llm_translator(self):
        """
        Design: PassThroughTranslator MUST implement LLMTranslator interface.
        """
        from domain.interfaces import LLMTranslator

        translator = PassThroughTranslator()

        assert isinstance(translator, LLMTranslator)

    def test_translate_preserves_intent_text_exactly(self):
        """
        Triangulation: verify exact text preservation with special characters.
        
        GIVEN an intent with special characters
        WHEN translate is called
        THEN the exact text MUST be preserved without modification
        """
        translator = PassThroughTranslator()
        special_text = "Café mélancolique à Paris 🎹\n\tTabbed"

        result = translator.translate(Intent(text=special_text), num_variations=1)

        assert result[0] == special_text
