"""
Best-of-N Search Use Case - Batch Generation Orchestration.

This module implements the core algorithm that orchestrates batch MIDI generation
using Best-of-N selection. It generates N complete sequences and returns the
highest-scoring one.

Architecture:
    - Pure domain logic operating on domain interfaces
    - Uses BatchMidiGenerator for batch token generation
    - Uses Evaluator for scoring sequences
    - No knowledge of infrastructure concerns (APIs, PyTorch, etc.)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from domain.entities import (
    GenerationProfile,
    GenerationResult,
    Intent,
    MidiSequence,
)
from domain.interfaces import (
    AudioRenderer,
    AudioSamples,
    BatchMidiGenerator,
    Evaluator,
    LLMTranslator,
)

if TYPE_CHECKING:
    from domain.entities import MidiBytes
    from domain.interfaces import PromptText

logger = logging.getLogger(__name__)


class BestOfNSearch:
    """
    Best-of-N Search orchestrator for batch MIDI generation.

    This use case implements the core algorithm that:
    1. Translates user intent into a technical prompt
    2. Generates N complete MIDI sequences in a single batch
    3. Evaluates all sequences using reward signals
    4. Returns the highest-scoring MIDI result

    The orchestrator is pure domain logic - it knows nothing about
    APIs, PyTorch, or other infrastructure concerns.
    """

    def __init__(
        self,
        translator: LLMTranslator,
        generator: BatchMidiGenerator,
        evaluator: Evaluator,
        audio_renderer: AudioRenderer,
    ):
        """
        Initialize the orchestrator with domain interface implementations.

        Args:
            translator: Converts natural language intent to technical prompts.
            generator: Generates batch MIDI tokens and decodes them.
            evaluator: Scores sequences against quality criteria.
            audio_renderer: Renders MIDI tokens to audio for evaluation.
        """
        self.translator = translator
        self.generator = generator
        self.evaluator = evaluator
        self.audio_renderer = audio_renderer

    def execute(
        self,
        intent: Intent,
        profile: GenerationProfile,
    ) -> GenerationResult:
        """
        Execute the Best-of-N search algorithm.

        Args:
            intent: The user's creative intent in natural language.
            profile: Configuration for generation parameters.

        Returns:
            GenerationResult containing the MIDI bytes and technical prompt
            of the best generated sequence.

        Raises:
            RuntimeError: If batch generation fails completely.
        """
        # Step 1: Get technical prompt
        # If using midillm, it expects natural language directly, so bypass the translator
        # if the profile specifies it, or just use the intent text.
        if profile.generator_type == "midillm":
            technical_prompt = intent.text
        else:
            technical_prompts = self.translator.translate(
                intent=intent,
                num_variations=1,
            )
            
            # Validate that we received at least one prompt
            if not technical_prompts:
                raise RuntimeError(
                    "Translator returned empty prompts list. "
                    "Cannot proceed with generation."
                )
            
            technical_prompt = technical_prompts[0]

        logger.info(
            f"Starting Best-of-N search with num_outputs={profile.num_outputs}, "
            f"prompt='{technical_prompt[:50]}...'"
        )

        # Step 2: Generate batch of N complete sequences
        sequences = self.generator.generate_batch(
            technical_prompt=technical_prompt,
            num_outputs=profile.num_outputs,
        )

        logger.info(f"Generated {len(sequences)} sequences")

        # Step 3: Evaluate all sequences and find the best
        best_sequence: MidiSequence | None = None
        best_reward = float("-inf")

        for i, tokens in enumerate(sequences):
            # Create MidiSequence for evaluation
            sequence = MidiSequence(
                technical_prompt=technical_prompt,
                tokens=tokens,
            )

            # Render to audio and evaluate
            # Wrap in try/except to handle individual sequence failures gracefully
            try:
                # First, ensure the tokens can be decoded into a valid MIDI file
                midi_bytes = self.generator.decode_to_midi(tokens)
                
                # Use synthesize_from_bytes if available (preferred as it uses the real decoded MIDI)
                if hasattr(self.audio_renderer, "synthesize_from_bytes"):
                    audio_data: AudioSamples = self.audio_renderer.synthesize_from_bytes(midi_bytes)
                else:
                    audio_data: AudioSamples = self.audio_renderer.render(tokens)
                    
                reward = self.evaluator.evaluate(
                    sequence=sequence,
                    audio_data=audio_data,
                    intent=intent,
                )
                
                # Temporarily store the decoded bytes to avoid double decoding
                setattr(sequence, "_temp_midi_bytes", midi_bytes)
            except Exception as e:
                logger.warning(
                    f"Sequence {i} evaluation failed: {e}. Assigning -inf reward."
                )
                reward = float("-inf")

            sequence.reward = reward

            logger.debug(f"Sequence {i}: {len(tokens)} tokens, reward={reward:.3f}")

            # Track best sequence (handle case where all rewards are -inf)
            if best_sequence is None or reward > best_reward:
                best_reward = reward
                best_sequence = sequence

        if best_sequence is None:
            raise RuntimeError(
                "Best-of-N search failed to produce any valid sequences. "
                "Check infrastructure logs for details."
            )

        logger.info(
            f"Best-of-N search complete: {len(best_sequence.tokens)} tokens, "
            f"reward={best_sequence.reward:.3f}"
        )

        # Step 4: Return best result
        midi_bytes = getattr(best_sequence, "_temp_midi_bytes", None)
        if midi_bytes is None:
            midi_bytes = self.generator.decode_to_midi(best_sequence.tokens)

        return GenerationResult(
            midi_bytes=midi_bytes,
            technical_prompt=best_sequence.technical_prompt,
        )
