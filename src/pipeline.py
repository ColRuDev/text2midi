"""
Text2Midi Pipeline - Dependency Injection Container and Orchestration.

This module provides the main entry point for text-to-MIDI generation.
The Text2MidiPipeline acts as a DI container, instantiating heavy adapters
once during initialization and reusing them across generate calls.

Architecture:
    - Pipeline as DI Container: Heavy adapters instantiated once in __init__
    - Factory Method for Strategy Selection: Selects ProgressiveSearch or BestOfNSearch
    - Clean separation between construction and execution
    - CLI-friendly interface with simple generate() method

Data Flow:
    CLI -> Pipeline (Loads Adapters + Selects Strategy) -> Search -> MIDI bytes
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from adapters.audio.fluidsynth_memory import InMemoryFluidSynthEngine
from adapters.evaluators.composite import CompositeEvaluator
from adapters.generators.midillm_generator import (
    MidiLLMGenerator,
    MidiLLMGeneratorConfig,
)
from adapters.generators.text2midi_generator import (
    Text2MidiGenerator,
    Text2MidiGeneratorConfig,
)
from adapters.translators.google_ai_translator import (
    GoogleAIConfig,
    GoogleAITranslator,
)
from adapters.translators.pass_through_translator import PassThroughTranslator
from domain.entities import GenerationProfile, GenerationResult, Intent, MidiBytes
from use_cases.best_of_n_search import BestOfNSearch
from use_cases.progressive_search import ProgressiveSearch

if TYPE_CHECKING:
    from domain.interfaces import BatchMidiGenerator, Evaluator, MidiGenerator

logger = logging.getLogger(__name__)


class Text2MidiPipeline:
    """
    Dependency Injection container for text-to-MIDI generation.

    This pipeline:
    - Instantiates heavy adapters once during initialization
    - Reuses the same adapters across all generate() calls
    - Uses a factory method to select between ProgressiveSearch and BestOfNSearch
    - Orchestrates generation strategy based on profile configuration

    Attributes:
        _translator: LLMTranslator for intent-to-prompt translation.
        _generator: MidiGenerator or BatchMidiGenerator for token generation.
        _evaluator: Evaluator for scoring MIDI sequences.
        _audio_renderer: AudioRenderer for audio synthesis.
        _search: ProgressiveSearch or BestOfNSearch orchestrator.

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
        midillm_config: MidiLLMGeneratorConfig | None = None,
        evaluator: "Evaluator | None" = None,
        sample_rate: int = 48000,
        profile: GenerationProfile | None = None,
    ) -> None:
        """
        Initialize the pipeline with all heavy adapters.

        Adapters are instantiated once and reused for all generate() calls.
        This mitigates the cold-start problem by loading models once.

        Args:
            translator_config: Optional config for GoogleAITranslator.
            generator_config: Optional config for Text2MidiGenerator.
            midillm_config: Optional config for MidiLLMGenerator.
            evaluator: Optional custom evaluator (uses CompositeEvaluator by default).
            sample_rate: Audio sample rate for FluidSynth (default: 48000 for CLAP).
            profile: Optional profile to determine initial strategy.
                If not provided, defaults to text2midi strategy.

        Raises:
            FileNotFoundError: If model files or prompts are not found.
            LLMTranslationError: If GOOGLE_API_KEY is not set.
            GeneratorError: If model loading fails.
        """
        logger.info("Initializing Text2MidiPipeline...")

        # Store configs for later use
        self._translator_config = translator_config
        self._generator_config = generator_config
        self._midillm_config = midillm_config
        self._sample_rate = sample_rate

        # Instantiate translator based on config presence
        # PassThroughTranslator when config is None (no API key validation)
        # GoogleAITranslator when config is provided (validates API key)
        if translator_config is None:
            self._translator = PassThroughTranslator()
            logger.debug("PassThroughTranslator initialized (no translation)")
        else:
            self._translator = GoogleAITranslator(translator_config)
            logger.debug("GoogleAITranslator initialized")

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

        # Cache generators by type to avoid re-instantiation (GPU memory leak prevention)
        self._generators: dict[str, "MidiGenerator | BatchMidiGenerator"] = {}
        # Lock for thread-safe generator cache mutations
        self._generators_lock = threading.Lock()

        # Generator and Search strategy - instantiated lazily by _get_search_for_profile
        self._generator: "MidiGenerator | BatchMidiGenerator | None" = None
        self._search: ProgressiveSearch | BestOfNSearch | None = None

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

        Thread-safety: Uses local search instance to avoid mutating shared state.

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

        # Use local search instance for thread-safety (don't mutate self._search)
        search = self._get_search_for_profile(profile)

        # Create Intent from text
        intent = Intent(text=text)

        # Execute search strategy
        result = search.execute(intent=intent, profile=profile)

        logger.info(f"Generation complete: {len(result.midi_bytes)} bytes")
        return result

    def _get_search_for_profile(
        self, profile: GenerationProfile
    ) -> ProgressiveSearch | BestOfNSearch:
        """
        Get or create the appropriate search strategy for the profile.

        Thread-safety: Returns a search instance without mutating instance state.
        Uses cached generators to prevent memory leaks.

        Args:
            profile: Configuration determining strategy selection.

        Returns:
            The appropriate search instance for the profile.
        """
        # Check if we already have the right search initialized
        current_is_batch = isinstance(self._search, BestOfNSearch)
        requested_is_batch = profile.generator_type == "midillm"

        if self._search is not None and current_is_batch == requested_is_batch:
            return self._search

        # Need different strategy - create locally without mutating instance
        if profile.generator_type == "midillm":
            # Check cache first (thread-safe)
            with self._generators_lock:
                if "midillm" in self._generators:
                    generator = self._generators["midillm"]
                elif self._midillm_config:
                    generator = MidiLLMGenerator(self._midillm_config)
                    self._generators["midillm"] = generator
                else:
                    generator = MidiLLMGenerator(MidiLLMGeneratorConfig())
                    self._generators["midillm"] = generator

            search = BestOfNSearch(
                translator=self._translator,
                generator=generator,
                evaluator=self._evaluator,
                audio_renderer=self._audio_renderer,
            )
            self._search = search
            return search
        else:
            # text2midi strategy
            with self._generators_lock:
                if "text2midi" in self._generators:
                    generator = self._generators["text2midi"]
                else:
                    generator = Text2MidiGenerator(
                        self._generator_config or Text2MidiGeneratorConfig()
                    )
                    self._generators["text2midi"] = generator

            search = ProgressiveSearch(
                translator=self._translator,
                generator=generator,
                evaluator=self._evaluator,
                audio_renderer=self._audio_renderer,
            )
            self._search = search
            return search


