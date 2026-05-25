"""
Translators - LLM adapters for intent-to-prompt translation.

This package contains adapters that implement the LLMTranslator interface
for various LLM providers (Google AI, OpenAI, etc.) and the PassThroughTranslator
for bypassing LLM translation.
"""

from .google_ai_translator import GoogleAITranslator, GoogleAIConfig
from .pass_through_translator import PassThroughTranslator

__all__ = ["GoogleAITranslator", "GoogleAIConfig", "PassThroughTranslator"]
