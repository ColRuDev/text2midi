"""
Progressive Search Use Case - Reward-Guided Beam Search Orchestration.

This module implements the core algorithm that orchestrates MIDI generation
using beam search guided by reward signals. It operates purely on domain
interfaces and has no knowledge of implementation details.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import List

from domain.entities import (
    GenerationProfile,
    GenerationResult,
    Intent,
    MidiSequence,
)
from domain.interfaces import (
    AudioRenderer,
    AudioSamples,
    Evaluator,
    LLMTranslator,
    MidiBytes,
    MidiGenerator,
)

logger = logging.getLogger(__name__)


@dataclass
class BranchResult:
    """Result of expanding a single branch during beam search."""

    sequence: MidiSequence
    is_alive: bool = True
    error: str | None = None


class ProgressiveSearch:
    """
    Reward-Guided Beam Search orchestrator for MIDI generation.

    This use case implements the core algorithm that:
    1. Translates user intent into technical prompts
    2. Explores multiple generation branches in parallel
    3. Evaluates branches using reward signals
    4. Prunes low-reward branches at each step
    5. Returns the best MIDI result

    The orchestrator is pure domain logic - it knows nothing about
    APIs, PyTorch, or other infrastructure concerns.
    """

    def __init__(
        self,
        translator: LLMTranslator,
        generator: MidiGenerator,
        evaluator: Evaluator,
        audio_renderer: AudioRenderer,
    ):
        """
        Initialize the orchestrator with domain interface implementations.

        Args:
            translator: Converts natural language intent to technical prompts.
            generator: Generates and decodes MIDI tokens.
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
        Execute the progressive search algorithm.

        Args:
            intent: The user's creative intent in natural language.
            profile: Configuration for search and evaluation parameters.

        Returns:
            GenerationResult containing the MIDI bytes and technical prompt
            of the best generated sequence.

        Raises:
            RuntimeError: If all branches fail before any successful generation.
        """
        # Step 1: Generate technical prompt variations
        technical_prompts = self.translator.translate(
            intent=intent,
            num_variations=profile.num_beams,
        )

        # Step 2: Initialize branches for each variation
        branches: List[MidiSequence] = [
            MidiSequence(technical_prompt=prompt) for prompt in technical_prompts
        ]

        # Track the best surviving sequence for graceful degradation
        best_survivor: MidiSequence | None = None
        tokens_generated = 0
        
        # Initialize RNG once for consistent sampling across all iterations
        rng = random.Random(profile.random_seed) if profile.random_seed is not None else random

        logger.info(
            f"Starting beam search with {len(branches)} branches, "
            f"max_tokens={profile.max_tokens}"
        )

        # Step 3: Iterative beam search loop
        while branches and tokens_generated < profile.max_tokens:
            # Expand each branch
            expanded_branches: List[BranchResult] = []

            for branch in branches:
                result = self._expand_branch(
                    branch=branch,
                    profile=profile,
                    intent=intent,
                )
                expanded_branches.append(result)

            # Update best survivor from successful expansions
            for result in expanded_branches:
                if result.is_alive and (
                    best_survivor is None
                    or result.sequence.reward > best_survivor.reward
                ):
                    best_survivor = result.sequence

            # Filter alive branches
            alive_branches = [r.sequence for r in expanded_branches if r.is_alive]

            if not alive_branches:
                logger.warning("All branches died in this iteration")
                branches = []  # Clear branches so final check works correctly
                break

            # Step 4: Prune to top_k
            alive_branches.sort(reverse=True)  # Sort by reward (highest first)
            branches = alive_branches[: profile.top_k]

            # Step 4b: Replenish branches with clones of survivors (PRD 07)
            # Use uniform random selection from surviving branches
            # Sample from original survivors to avoid bias from mutated list
            survivors = list(branches)
            while len(branches) < profile.num_beams:
                clone_source = rng.choice(survivors)
                branches.append(clone_source.copy())

            tokens_generated += profile.token_batch_size

            logger.debug(
                f"Iteration complete: {len(branches)} branches alive, "
                f"best_reward={branches[0].reward:.3f}"
            )

        # Step 5: Return best result
        if not branches and best_survivor is not None:
            logger.warning(
                "All branches failed, returning best surviving partial result"
            )
            return GenerationResult(
                midi_bytes=self.generator.decode_to_midi(best_survivor.tokens),
                technical_prompt=best_survivor.technical_prompt,
            )

        if not branches:
            raise RuntimeError(
                "All generation branches failed before producing any output. "
                "Check infrastructure logs for details."
            )

        # Return the highest-reward branch
        branches.sort(reverse=True)
        winner = branches[0]

        logger.info(
            f"Beam search complete: {len(winner.tokens)} tokens, "
            f"reward={winner.reward:.3f}"
        )

        return GenerationResult(
            midi_bytes=self.generator.decode_to_midi(winner.tokens),
            technical_prompt=winner.technical_prompt,
        )

    def _expand_branch(
        self,
        branch: MidiSequence,
        profile: GenerationProfile,
        intent: Intent,
    ) -> BranchResult:
        """
        Expand a single branch by generating and evaluating tokens.

        Args:
            branch: The sequence to expand.
            profile: Generation configuration.
            intent: Original user intent for evaluation.

        Returns:
            BranchResult with updated sequence and status.
        """
        try:
            # Generate next batch of tokens
            new_tokens = self.generator.generate_step(
                technical_prompt=branch.technical_prompt,
                current_tokens=branch.tokens,
                num_tokens=profile.token_batch_size,
            )

            # Update sequence with new tokens
            branch.tokens.extend(new_tokens)

            # Decode to MIDI and render to audio
            audio_data: AudioSamples = self.audio_renderer.render(branch.tokens)

            # Evaluate and update reward
            reward = self.evaluator.evaluate(
                sequence=branch,
                audio_data=audio_data,
                intent=intent,
            )
            branch.reward = reward

            return BranchResult(sequence=branch, is_alive=True)

        except Exception as e:
            logger.error(
                f"Branch died due to error: {e.__class__.__name__}: {e}. "
                f"Prompt: '{branch.technical_prompt[:50]}...', "
                f"Tokens generated: {len(branch.tokens)}"
            )
            return BranchResult(
                sequence=branch,
                is_alive=False,
                error=str(e),
            )
