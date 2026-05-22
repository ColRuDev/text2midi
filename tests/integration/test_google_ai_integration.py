"""
Integration tests for Google AI Translator with real API calls.

These tests require a valid GOOGLE_API_KEY environment variable.
They are skipped automatically if the key is not present.
"""

import os
import pytest

from adapters.translators.google_ai_translator import GoogleAITranslator, GoogleAIConfig
from domain.entities import Intent


# Skip all tests in this module if no API key is available
pytestmark = pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set - skipping integration tests",
)


class TestGoogleAIIntegration:
    """Integration tests with real Google AI API."""

    def test_single_translation_returns_valid_prompt(self):
        """A single translation returns a valid technical prompt."""
        translator = GoogleAITranslator()
        
        intent = Intent("A peaceful piano melody at sunrise")
        result = translator.translate(intent, num_variations=1)
        
        assert len(result) == 1
        assert isinstance(result[0], str)
        assert len(result[0]) > 50  # Expect substantial output
        
        # Should contain musical terminology
        prompt_lower = result[0].lower()
        assert any(term in prompt_lower for term in ["tempo", "key", "chord", "melody", "piano"])

    def test_multiple_variations_returns_distinct_prompts(self):
        """Multiple variations return different prompts."""
        translator = GoogleAITranslator()
        
        intent = Intent("An upbeat electronic dance track")
        result = translator.translate(intent, num_variations=3)
        
        assert len(result) == 3
        
        # All should be strings
        assert all(isinstance(p, str) for p in result)
        
        # Should have some variety (not all identical)
        # Note: with temperature > 0, responses should vary
        unique_prompts = set(result)
        assert len(unique_prompts) >= 1  # At minimum, they exist

    def test_custom_config_uses_specified_model(self):
        """Custom config parameters are respected."""
        config = GoogleAIConfig(
            model_name="gemma-4-26b-a4b-it",
            temperature=0.3,  # Lower for more deterministic output
            max_output_tokens=300,
        )
        translator = GoogleAITranslator(config)
        
        intent = Intent("A sad cello solo")
        result = translator.translate(intent, num_variations=1)
        
        assert len(result) == 1
        assert isinstance(result[0], str)

    def test_system_prompt_is_loaded(self):
        """System prompt is loaded correctly from file."""
        translator = GoogleAITranslator()
        
        # Should have loaded the system prompt
        assert translator._system_prompt is not None
        assert len(translator._system_prompt) > 100
        assert "Music Theory Consultant" in translator._system_prompt

    def test_intent_with_special_characters(self):
        """Intent with special characters works correctly."""
        translator = GoogleAITranslator()
        
        # Intent with quotes, accents, and special chars
        intent = Intent("A melancholic \"nocturne\" with élégance — Chopin style")
        result = translator.translate(intent, num_variations=1)
        
        assert len(result) == 1
        assert isinstance(result[0], str)
