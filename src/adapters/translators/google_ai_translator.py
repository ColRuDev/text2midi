"""
Google AI Translator Adapter - Implements LLMTranslator using Google AI SDK.

This adapter wraps the google-genai SDK to translate natural language
intents into structured technical prompts for MIDI generation.

Architecture:
    - Implements domain.interfaces.LLMTranslator
    - Uses google.genai SDK for API calls
    - Loads system prompt from external file at initialization
    - Wraps all SDK exceptions in adapters.exceptions.LLMTranslationError
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from google import genai
from google.genai import types

from adapters.exceptions import LLMTranslationError
from domain.entities import Intent, PromptText
from domain.interfaces import LLMTranslator


@dataclass
class GoogleAIConfig:
    """
    Configuration for GoogleAI translator.

    All parameters have sensible defaults for the Gemma model.

    Attributes:
        model_name: The model identifier to use for generation.
        temperature: Sampling temperature (0.0-1.0). Higher = more creative.
        top_p: Nucleus sampling parameter (0.0-1.0).
        max_output_tokens: Maximum tokens per response.
        system_prompt_path: Path to the system prompt file.

    Example:
        >>> config = GoogleAIConfig(temperature=0.5, max_output_tokens=2048)
        >>> translator = GoogleAITranslator(config)
    """

    model_name: str = "gemma-4-31b-it"
    temperature: float = 0.7
    top_p: float = 0.9
    max_output_tokens: int = 1024
    system_prompt_path: Path = field(
        default_factory=lambda: Path(__file__).parent / "prompts" / "system_prompt.md"
    )


class GoogleAITranslator(LLMTranslator):
    """
    Adapter implementing LLMTranslator using Google AI SDK.

    This translator converts natural language intents into structured
    technical prompts using Google's models via the google-genai SDK.

    The adapter:
    - Loads system prompt from file at initialization (fail-fast)
    - Validates GOOGLE_API_KEY environment variable
    - Makes sequential API calls for multiple variations
    - Wraps all SDK exceptions in LLMTranslationError

    Example:
        >>> config = GoogleAIConfig()
        >>> translator = GoogleAITranslator(config)
        >>> prompts = translator.translate(Intent("A sunrise"), 3)
        >>> len(prompts)
        3
    """

    def __init__(self, config: GoogleAIConfig | None = None):
        """
        Initialize the translator with configuration.

        Args:
            config: Configuration parameters. Uses defaults if None.

        Raises:
            FileNotFoundError: If system prompt file doesn't exist.
            LLMTranslationError: If GOOGLE_API_KEY is not set.
        """
        self._config = config or GoogleAIConfig()
        self._system_prompt = self._load_system_prompt()
        self._client = self._create_client()

    def translate(self, intent: Intent, num_variations: int) -> List[PromptText]:
        """
        Translate intent to technical prompts via Google AI.

        Makes num_variations sequential API calls to ensure each
        variation is independent. Returns raw LLM output without
        validation.

        Args:
            intent: The user's creative intent in natural language.
            num_variations: Number of prompt variations to generate.

        Returns:
            List of technical prompt strings. Empty list if num_variations=0.

        Raises:
            LLMTranslationError: If any API call fails.
        """
        if num_variations < 0:
            raise ValueError(f"num_variations must be >= 0, got {num_variations}")
        if num_variations == 0:
            return []

        prompts: List[PromptText] = []
        for _ in range(num_variations):
            prompt = self._generate_single_prompt(intent)
            prompts.append(prompt)

        return prompts

    def _load_system_prompt(self) -> str:
        """
        Load and cache system prompt from file.

        Returns:
            The content of the system prompt file.

        Raises:
            FileNotFoundError: If the file doesn't exist.
        """
        path = self._config.system_prompt_path
        if not path.exists():
            raise FileNotFoundError(
                f"System prompt not found: {path}. "
                f"Create the file or set a different system_prompt_path in config."
            )
        return path.read_text(encoding="utf-8")

    def _create_client(self) -> genai.Client:
        """
        Create and configure the genai client.

        Returns:
            Configured genai.Client instance.

        Raises:
            LLMTranslationError: If GOOGLE_API_KEY is not set.
        """
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise LLMTranslationError(
                "GOOGLE_API_KEY environment variable not set. "
                "Set it before initializing GoogleAITranslator."
            )
        return genai.Client(api_key=api_key)

    def _generate_single_prompt(self, intent: Intent) -> PromptText:
        """
        Generate a single prompt variation.

        Args:
            intent: The user's creative intent.

        Returns:
            Raw LLM output as a prompt string.

        Raises:
            LLMTranslationError: If the API call fails.
        """
        try:
            response = self._client.models.generate_content(
                model=self._config.model_name,
                contents=intent.text,
                config=types.GenerateContentConfig(
                    system_instruction=self._system_prompt,
                    temperature=self._config.temperature,
                    top_p=self._config.top_p,
                    max_output_tokens=self._config.max_output_tokens,
                ),
            )
            if not response.text:
                raise LLMTranslationError("Empty or blocked response from LLM")
            return response.text
        except LLMTranslationError:
            raise
        except Exception as e:
            raise LLMTranslationError(
                f"LLM translation failed: {type(e).__name__}: {e}"
            ) from e
