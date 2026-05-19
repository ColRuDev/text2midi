"""
Text2Midi Pipeline - Dependency Injection Container and Orchestration.

This module provides the main entry point for text-to-MIDI generation.
The Text2MidiPipeline acts as a DI container, instantiating heavy adapters
once during initialization and reusing them across generate calls.

Architecture:
    - Pipeline as DI Container: Heavy adapters instantiated once in __init__
    - Uses ProgressiveSearch for the generation algorithm
    - Clean separation between construction and execution
    - CLI-friendly interface with simple generate() method

Data Flow:
    CLI -> Pipeline (Loads Adapters) -> ProgressiveSearch -> MIDI bytes
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from adapters.audio.fluidsynth_memory import InMemoryFluidSynthEngine
from adapters.evaluators.composite import CompositeEvaluator
from adapters.generators.text2midi_generator import (
    Text2MidiGenerator,
    Text2MidiGeneratorConfig,
)
from adapters.translators.google_ai_translator import (
    GoogleAIConfig,
    GoogleAITranslator,
)
from domain.entities import GenerationProfile, GenerationResult, Intent, MidiBytes
from use_cases.progressive_search import ProgressiveSearch

if TYPE_CHECKING:
    from domain.interfaces import Evaluator

logger = logging.getLogger(__name__)


class Text2MidiPipeline:
    """
    Dependency Injection container for text-to-MIDI generation.

    This pipeline:
    - Instantiates heavy adapters once during initialization
    - Reuses the same adapters across all generate() calls
    - Orchestrates ProgressiveSearch for MIDI generation

    Attributes:
        _translator: LLMTranslator for intent-to-prompt translation.
        _generator: MidiGenerator for token generation and decoding.
        _evaluator: Evaluator for scoring MIDI sequences.
        _audio_renderer: AudioRenderer for audio synthesis.

    Example:
        >>> pipeline = Text2MidiPipeline()
        >>> midi_bytes = pipeline.generate("A peaceful sunrise", BALANCED)
        >>> with open("output.mid", "wb") as f:
        ...     f.write(midi_bytes)
    """

    def __init__(
        self,
        translator_config: GoogleAIConfig | None = None,
        generator_config: Text2MidiGeneratorConfig | None = None,
        evaluator: "Evaluator | None" = None,
        sample_rate: int = 48000,
    ) -> None:
        """
        Initialize the pipeline with all heavy adapters.

        Adapters are instantiated once and reused for all generate() calls.
        This mitigates the cold-start problem by loading models once.

        Args:
            translator_config: Optional config for GoogleAITranslator.
            generator_config: Optional config for Text2MidiGenerator.
            evaluator: Optional custom evaluator (uses CompositeEvaluator by default).
            sample_rate: Audio sample rate for FluidSynth (default: 48000 for CLAP).

        Raises:
            FileNotFoundError: If model files or prompts are not found.
            LLMTranslationError: If GOOGLE_API_KEY is not set.
            GeneratorError: If model loading fails.
        """
        logger.info("Initializing Text2MidiPipeline...")

        # Instantiate heavy adapters - these are loaded ONCE
        self._translator = GoogleAITranslator(translator_config)
        logger.debug("Translator initialized")

        self._generator = Text2MidiGenerator(
            generator_config or Text2MidiGeneratorConfig()
        )
        logger.debug("Generator initialized")

        # Audio renderer for evaluation
        self._audio_renderer = InMemoryFluidSynthEngine(sample_rate=sample_rate)
        logger.debug("Audio renderer initialized")

        # Evaluator (use provided or create default CompositeEvaluator)
        if evaluator is not None:
            self._evaluator = evaluator
        else:
            self._evaluator = CompositeEvaluator(
                clap_evaluator=None,  # CLAP optional
                heuristics_evaluator=None,  # Heuristics optional
                profile=None,
            )
        logger.debug("Evaluator initialized")

        # Create the ProgressiveSearch orchestrator
        self._search = ProgressiveSearch(
            translator=self._translator,
            generator=self._generator,
            evaluator=self._evaluator,
            audio_renderer=self._audio_renderer,
        )
        logger.info("Pipeline initialization complete")

    def generate(
        self,
        text: str,
        profile: GenerationProfile,
    ) -> GenerationResult:
        """
        Generate MIDI from natural language text.

        Orchestrates the full text-to-MIDI pipeline using the provided
        configuration profile. Reuses pre-loaded adapters for efficiency.

        Args:
            text: Natural language description of desired music.
            profile: GenerationProfile controlling search and evaluation.

        Returns:
            GenerationResult containing the MIDI bytes and technical prompt
            of the best generated sequence.

        Raises:
            RuntimeError: If all generation branches fail.
            LLMTranslationError: If LLM translation fails.
            GeneratorError: If MIDI generation fails.

        Example:
            >>> from config.profiles import BALANCED
            >>> result = pipeline.generate("A peaceful sunrise", BALANCED)
            >>> len(result.midi_bytes) > 0
            True
        """
        logger.info(f"Generating MIDI for: '{text[:50]}...' with profile {profile}")

        # Create Intent from text
        intent = Intent(text=text)

        # Execute ProgressiveSearch
        result = self._search.execute(intent=intent, profile=profile)

        logger.info(f"Generation complete: {len(result.midi_bytes)} bytes")
        return result
