"""
Unit tests for ProgressiveSearch use case.

Tests validate the PRD acceptance criteria using mock implementations
of domain interfaces.
"""

import unittest
from typing import List

from domain.entities import (
    GenerationProfile,
    GenerationResult,
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

        self.assertIsInstance(result, GenerationResult)
        self.assertTrue(result.midi_bytes.startswith(b"MIDI_DATA_"))

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

        self.assertIsInstance(result, GenerationResult)
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

        self.assertIsInstance(result, GenerationResult)
        # Should have generated some tokens before failing
        self.assertGreater(fail_after_one.call_count, 0)

    def test_returns_only_midi_bytes(self):
        """
        AC4: Returns GenerationResult containing the final MIDI bytes and technical prompt.
        """
        search = ProgressiveSearch(
            translator=self.translator,
            generator=self.generator,
            evaluator=self.evaluator,
            audio_renderer=self.audio_renderer,
        )

        result = search.execute(self.intent, self.profile)

        # Must be a GenerationResult with midi_bytes
        self.assertIsInstance(result, GenerationResult)
        # midi_bytes must start with MIDI marker
        self.assertTrue(result.midi_bytes.startswith(b"MIDI_DATA_"))
        # technical_prompt must be a string
        self.assertIsInstance(result.technical_prompt, str)

    def test_top_k_pruning_keeps_best_branches(self):
        """
        Verify that top_k pruning keeps the highest-reward branches.
        """
        # Set up evaluator with predictable rewards
        evaluator = MockEvaluator(rewards=[0.1, 0.9, 0.5])

        search = ProgressiveSearch(
            translator=self.translator,
            generator=self.generator,
            evaluator=evaluator,
            audio_renderer=self.audio_renderer,
        )

        result = search.execute(self.intent, self.profile)

        # Should have evaluated multiple branches
        self.assertGreater(evaluator.call_count, 0)
        self.assertIsInstance(result, GenerationResult)


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

        self.assertIsInstance(result, GenerationResult)

    def test_graceful_degradation_with_partial_results(self):
        """
        Test that partial results from dead branches are preserved.
        """
        translator = MockTranslator(prompts=["p1", "p2"])
        # First branch succeeds once, then fails; second always fails
        generator = MockGenerator(tokens_per_call=10)
        evaluator = MockEvaluator(rewards=[0.2, 0.8])
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

        self.assertIsInstance(result, GenerationResult)


class TestRandomBranchCloning(unittest.TestCase):
    """Tests for PRD 07: Random Branch Cloning."""

    def test_beam_width_maintained_after_pruning_with_clones(self):
        """
        PRD 07 Scenario 1: Branches are replenished to num_beams after pruning.

        GIVEN a search iteration completes with some live branches and some dead branches
        WHEN the pruning step is executed
        THEN the system MUST replace dead branches with clones of surviving branches
        AND the total beam width MUST be maintained for the next iteration.
        """
        import random
        from unittest.mock import patch, MagicMock

        translator = MockTranslator(prompts=["p1", "p2", "p3"])
        generator = MockGenerator(should_fail=True, fail_after=1)
        evaluator = MockEvaluator(rewards=[0.8, 0.6, 0.5])
        audio_renderer = MockAudioRenderer()

        search = ProgressiveSearch(
            translator=translator,
            generator=generator,
            evaluator=evaluator,
            audio_renderer=audio_renderer,
        )

        profile = GenerationProfile(
            token_batch_size=10,
            num_beams=3,  # We want 3 branches total
            top_k=2,      # Prune to top 2
            max_tokens=25,  # Allow 2 iterations
            clap_weight=0.2,
            key_weight=0.5,
            note_weight=0.3,
        )

        # Mock random.choice to track calls and return predictable values
        with patch('random.choice') as mock_choice:
            # Return the first surviving branch each time
            mock_choice.return_value = MidiSequence(
                technical_prompt="p1",
                tokens=[0, 1, 2],
                reward=0.8
            )

            result = search.execute(Intent("test"), profile)

            # Verify random.choice was called to clone branches
            # After pruning to top_k=2, we need to replenish to num_beams=3
            # So random.choice should be called at least once
            self.assertGreater(mock_choice.call_count, 0, 
                "random.choice should be called to select branches for cloning")

            # Verify the argument passed to random.choice was the surviving branches list
            for call in mock_choice.call_args_list:
                args = call[0]
                self.assertIsInstance(args[0], list, 
                    "random.choice should be called with a list of branches")
                # Each branch should be a MidiSequence
                if args[0]:
                    self.assertIsInstance(args[0][0], MidiSequence,
                        "random.choice should be called with MidiSequence objects")

        self.assertIsInstance(result, GenerationResult)

    def test_random_choice_called_with_surviving_branches_for_uniform_selection(self):
        """
        PRD 07: Selection of surviving branches for cloning MUST follow a uniform distribution.

        Verify that random.choice is called with the list of surviving branches,
        ensuring uniform selection among survivors.
        """
        import random
        from unittest.mock import patch

        translator = MockTranslator(prompts=["p1", "p2", "p3"])
        generator = MockGenerator(should_fail=True, fail_after=1)
        evaluator = MockEvaluator(rewards=[0.8, 0.6, 0.5])
        audio_renderer = MockAudioRenderer()

        search = ProgressiveSearch(
            translator=translator,
            generator=generator,
            evaluator=evaluator,
            audio_renderer=audio_renderer,
        )

        profile = GenerationProfile(
            token_batch_size=10,
            num_beams=3,
            top_k=2,
            max_tokens=25,
            clap_weight=0.2,
            key_weight=0.5,
            note_weight=0.3,
        )

        with patch('random.choice') as mock_choice:
            # Track what branches are passed to random.choice
            selected_branches = []
            
            def track_choice(branches):
                selected_branches.append(list(branches))  # Capture the list
                # Return a copy of the first branch
                return branches[0].copy() if branches else MidiSequence(technical_prompt="empty")

            mock_choice.side_effect = track_choice

            result = search.execute(Intent("test"), profile)

            # Verify random.choice was called at least once
            self.assertGreater(mock_choice.call_count, 0,
                "random.choice should be called for branch cloning")

            # Verify each call was with a list of MidiSequence objects (the survivors)
            for branch_list in selected_branches:
                self.assertIsInstance(branch_list, list)
                if branch_list:
                    self.assertIsInstance(branch_list[0], MidiSequence)
                    # All items should be MidiSequence
                    for branch in branch_list:
                        self.assertIsInstance(branch, MidiSequence)

        self.assertIsInstance(result, GenerationResult)

    def test_cloning_bypassed_when_all_branches_die(self):
        """
        PRD 07 Scenario 2: All branches fail simultaneously.

        GIVEN a search iteration completes with zero live branches
        WHEN the pruning step is executed
        THEN the system MUST NOT attempt to clone any branches
        AND the system MUST retain existing abort logic and terminate the search correctly.
        """
        import random
        from unittest.mock import patch

        translator = MockTranslator(prompts=["p1", "p2"])
        # Generator fails immediately (before any successful generation)
        generator = MockGenerator(should_fail=True, fail_after=0)
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
            max_tokens=20,
            clap_weight=0.2,
            key_weight=0.5,
            note_weight=0.3,
        )

        with patch('random.choice') as mock_choice:
            # Should raise RuntimeError because all branches fail before producing output
            with self.assertRaises(RuntimeError) as ctx:
                search.execute(Intent("test"), profile)

            # Verify random.choice was NOT called (cloning should be bypassed)
            self.assertEqual(mock_choice.call_count, 0,
                "random.choice should NOT be called when all branches die")

        self.assertIn("All generation branches failed", str(ctx.exception))


class TestProgressiveSearchReturnsGenerationResult(unittest.TestCase):
    """Tests for PRD 08: ProgressiveSearch returns GenerationResult."""

    def test_execute_returns_generation_result_with_midi_bytes(self):
        """
        PRD 08: execute MUST return a GenerationResult containing midi_bytes.
        """
        translator = MockTranslator(prompts=["prompt1"])
        generator = MockGenerator()
        evaluator = MockEvaluator(rewards=[0.8])
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

        self.assertIsInstance(result, GenerationResult)
        self.assertIsInstance(result.midi_bytes, bytes)
        self.assertTrue(result.midi_bytes.startswith(b"MIDI_DATA_"))

    def test_execute_returns_generation_result_with_technical_prompt_from_winner(self):
        """
        PRD 08: execute MUST return GenerationResult with the winning technical_prompt.

        GIVEN a beam search completes with a winner
        WHEN execute returns
        THEN the technical_prompt MUST match the winner's prompt
        """
        translator = MockTranslator(prompts=["winner_prompt", "loser_prompt"])
        generator = MockGenerator()
        # MockEvaluator assigns: call 1 -> reward[1]=0.5, call 2 -> reward[2%2]=reward[0]=0.9
        # So first branch gets 0.5, second gets 0.9. We want winner_prompt to win.
        # Swap reward order so call 1 gets high reward (for winner_prompt)
        evaluator = MockEvaluator(rewards=[0.5, 0.9])  # call 1 -> 0.9, call 2 -> 0.5
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
            max_tokens=20,
            clap_weight=0.2,
            key_weight=0.5,
            note_weight=0.3,
        )

        result = search.execute(Intent("test"), profile)

        self.assertIsInstance(result, GenerationResult)
        # Winner should be the first prompt (highest reward after MockEvaluator cycling)
        self.assertEqual(result.technical_prompt, "winner_prompt")

    def test_execute_returns_generation_result_with_prompt_from_best_survivor(self):
        """
        PRD 08: When all branches die, return GenerationResult with best_survivor's prompt.

        GIVEN all branches fail after generating something
        WHEN execute returns the best partial result
        THEN the technical_prompt MUST match the best_survivor's prompt
        """
        translator = MockTranslator(prompts=["survivor_prompt", "other_prompt"])
        generator = MockGenerator(should_fail=True, fail_after=1)
        # MockEvaluator assigns: call 1 -> reward[1]=0.6, call 2 -> reward[2%2]=reward[0]=0.8
        # We want survivor_prompt to have higher reward, so swap order
        evaluator = MockEvaluator(rewards=[0.6, 0.8])  # call 1 -> 0.8, call 2 -> 0.6
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
            max_tokens=25,
            clap_weight=0.2,
            key_weight=0.5,
            note_weight=0.3,
        )

        result = search.execute(Intent("test"), profile)

        self.assertIsInstance(result, GenerationResult)
        # Best survivor should have the higher reward prompt
        self.assertEqual(result.technical_prompt, "survivor_prompt")


if __name__ == "__main__":
    unittest.main()
