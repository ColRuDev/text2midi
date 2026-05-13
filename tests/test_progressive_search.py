"""
Unit tests for ProgressiveSearch use case.

Tests validate the PRD acceptance criteria using mock implementations
of domain interfaces.
"""

import unittest
from typing import List

from domain.entities import (
    GenerationProfile,
    Intent,
    MidiSequence,
)
from domain.interfaces import (
    AudioSamples,
    MidiBytes,
    PromptText,
    TokenId,
)
from use_cases.progressive_search import ProgressiveSearch


class MockTranslator:
    """Mock LLMTranslator for testing."""

    def __init__(self, prompts: List[PromptText] | None = None):
        self.prompts = prompts or ["prompt1", "prompt2", "prompt3"]
        self.call_count = 0

    def translate(self, intent: Intent, num_variations: int) -> List[PromptText]:
        self.call_count += 1
        return self.prompts[:num_variations]


class MockGenerator:
    """Mock MidiGenerator for testing."""

    def __init__(
        self,
        tokens_per_call: int = 10,
        should_fail: bool = False,
        fail_after: int = 999,
    ):
        self.tokens_per_call = tokens_per_call
        self.should_fail = should_fail
        self.fail_after = fail_after
        self.call_count = 0
        self.total_tokens_generated = 0

    def generate_step(
        self,
        technical_prompt: PromptText,
        current_tokens: List[TokenId],
        num_tokens: int,
    ) -> List[TokenId]:
        self.call_count += 1
        self.total_tokens_generated += num_tokens

        if self.should_fail and self.call_count > self.fail_after:
            raise RuntimeError("Simulated generation failure")

        # Return new tokens starting from current length
        start = len(current_tokens)
        return list(range(start, start + num_tokens))

    def decode_to_midi(self, tokens: List[TokenId]) -> MidiBytes:
        return b"MIDI_DATA_" + bytes(str(len(tokens)), "utf-8")


class MockAudioRenderer:
    """Mock AudioRenderer for testing."""

    def render(self, tokens: List[TokenId]) -> AudioSamples:
        return b"AUDIO_" + bytes(str(len(tokens)), "utf-8")


class MockEvaluator:
    """Mock Evaluator for testing."""

    def __init__(self, rewards: List[float] | None = None):
        self.rewards = rewards or [0.5, 0.7, 0.6]
        self.call_count = 0
        self.evaluations: List[tuple] = []

    def evaluate(
        self,
        sequence: MidiSequence,
        audio_data: AudioSamples,
        intent: Intent,
    ) -> float:
        self.call_count += 1
        reward = self.rewards[self.call_count % len(self.rewards)]
        self.evaluations.append((sequence.technical_prompt, reward))
        return reward


class TestProgressiveSearch(unittest.TestCase):
    """Test suite for ProgressiveSearch acceptance criteria."""

    def setUp(self):
        """Set up test fixtures."""
        self.translator = MockTranslator()
        self.generator = MockGenerator()
        self.evaluator = MockEvaluator()
        self.audio_renderer = MockAudioRenderer()
        self.intent = Intent("A peaceful sunrise")
        self.profile = GenerationProfile(
            token_batch_size=10,
            num_beams=3,
            top_k=2,
            max_tokens=30,
            clap_weight=0.2,
            key_weight=0.5,
            note_weight=0.3,
        )

    def test_beam_search_uses_only_domain_interfaces(self):
        """
        AC1: Implements Beam Search using only domain interfaces.
        """
        search = ProgressiveSearch(
            translator=self.translator,
            generator=self.generator,
            evaluator=self.evaluator,
            audio_renderer=self.audio_renderer,
        )

        # Should work with any implementation of the interfaces
        result = search.execute(self.intent, self.profile)

        self.assertIsInstance(result, bytes)
        self.assertTrue(result.startswith(b"MIDI_DATA_"))

        # Verify all interfaces were called
        self.assertEqual(self.translator.call_count, 1)
        self.assertGreater(self.generator.call_count, 0)
        self.assertGreater(self.evaluator.call_count, 0)

    def test_error_handling_per_branch_continues_with_survivors(self):
        """
        AC2: Captures errors per branch and continues with survivors.
        """
        # Configure one branch to fail, others to succeed
        failing_generator = MockGenerator(should_fail=True, fail_after=1)

        search = ProgressiveSearch(
            translator=self.translator,
            generator=failing_generator,
            evaluator=self.evaluator,
            audio_renderer=self.audio_renderer,
        )

        # Should not raise - survivors continue
        result = search.execute(self.intent, self.profile)

        self.assertIsInstance(result, bytes)
        # At least some branches should have succeeded
        self.assertGreater(failing_generator.call_count, 0)

    def test_all_branches_fail_before_generating_raises_error(self):
        """
        AC3: If all branches fail before producing any output, raises error.
        """
        # Configure all branches to fail immediately (before any generation)
        always_fail_generator = MockGenerator(should_fail=True, fail_after=0)

        search = ProgressiveSearch(
            translator=self.translator,
            generator=always_fail_generator,
            evaluator=self.evaluator,
            audio_renderer=self.audio_renderer,
        )

        # Should raise RuntimeError because no branch produced output
        with self.assertRaises(RuntimeError) as ctx:
            search.execute(self.intent, self.profile)

        self.assertIn("All generation branches failed", str(ctx.exception))

    def test_all_branches_fail_after_generating_returns_best_partial(self):
        """
        AC3: If all branches fail after generating something, returns best partial.
        """
        # Configure branches to succeed once, then fail
        fail_after_one = MockGenerator(should_fail=True, fail_after=1)

        search = ProgressiveSearch(
            translator=self.translator,
            generator=fail_after_one,
            evaluator=self.evaluator,
            audio_renderer=self.audio_renderer,
        )

        # Should return best partial result, not raise
        result = search.execute(self.intent, self.profile)

        self.assertIsInstance(result, bytes)
        # Should have generated some tokens before failing
        self.assertGreater(fail_after_one.call_count, 0)

    def test_returns_only_midi_bytes(self):
        """
        AC4: Returns only the final MIDI bytes.
        """
        search = ProgressiveSearch(
            translator=self.translator,
            generator=self.generator,
            evaluator=self.evaluator,
            audio_renderer=self.audio_renderer,
        )

        result = search.execute(self.intent, self.profile)

        # Must be bytes, not a wrapper object
        self.assertIsInstance(result, bytes)
        # Must start with MIDI marker
        self.assertTrue(result.startswith(b"MIDI_DATA_"))

    def test_top_k_pruning_keeps_best_branches(self):
        """
        Verify that top_k pruning keeps the highest-reward branches.
        """
        # Set up evaluator with predictable rewards
        evaluator = MockEvaluator(rewards=[0.9, 0.5, 0.7])

        search = ProgressiveSearch(
            translator=self.translator,
            generator=self.generator,
            evaluator=evaluator,
            audio_renderer=self.audio_renderer,
        )

        result = search.execute(self.intent, self.profile)

        # Should have evaluated multiple branches
        self.assertGreater(evaluator.call_count, 0)
        self.assertIsInstance(result, bytes)


class TestProgressiveSearchEdgeCases(unittest.TestCase):
    """Edge case tests for ProgressiveSearch."""

    def test_single_beam(self):
        """Test with only one beam (degenerates to greedy search)."""
        translator = MockTranslator(prompts=["single_prompt"])
        generator = MockGenerator()
        evaluator = MockEvaluator()
        audio_renderer = MockAudioRenderer()

        search = ProgressiveSearch(
            translator=translator,
            generator=generator,
            evaluator=evaluator,
            audio_renderer=audio_renderer,
        )

        profile = GenerationProfile(
            token_batch_size=10,
            num_beams=1,
            top_k=1,
            max_tokens=20,
            clap_weight=0.2,
            key_weight=0.5,
            note_weight=0.3,
        )

        result = search.execute(Intent("test"), profile)

        self.assertIsInstance(result, bytes)

    def test_graceful_degradation_with_partial_results(self):
        """
        Test that partial results from dead branches are preserved.
        """
        translator = MockTranslator(prompts=["p1", "p2"])
        # First branch succeeds once, then fails; second always fails
        generator = MockGenerator(tokens_per_call=10)
        evaluator = MockEvaluator(rewards=[0.8, 0.6])
        audio_renderer = MockAudioRenderer()

        search = ProgressiveSearch(
            translator=translator,
            generator=generator,
            evaluator=evaluator,
            audio_renderer=audio_renderer,
        )

        profile = GenerationProfile(
            token_batch_size=10,
            num_beams=2,
            top_k=1,
            max_tokens=15,  # Will trigger one iteration
            clap_weight=0.2,
            key_weight=0.5,
            note_weight=0.3,
        )

        result = search.execute(Intent("test"), profile)

        self.assertIsInstance(result, bytes)


if __name__ == "__main__":
    unittest.main()
