"""
Translators - LLM adapters for intent-to-prompt translation.

This package contains adapters that implement the LLMTranslator interface
for various LLM providers (Google AI, OpenAI, etc.).
"""

from .google_ai_translator import GoogleAITranslator, GoogleAIConfig

__all__ = ["GoogleAITranslator", "GoogleAIConfig"]
