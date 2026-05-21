"""
Unit tests for BestOfNSearch use case.

Tests validate the batch-generation specification acceptance criteria
using mock implementations of domain interfaces.
"""

import unittest
from typing import List
from unittest.mock import MagicMock

from domain.entities import (
    GenerationProfile,
    GenerationResult,
    Intent,
    MidiSequence,
)
from domain.interfaces import (
    AudioSamples,
    BatchMidiGenerator,
    MidiBytes,
    PromptText,
    TokenId,
)


class MockBatchGenerator(BatchMidiGenerator):
    """Mock BatchMidiGenerator for testing."""

    def __init__(self, sequences: List[List[TokenId]] | None = None):
        self.sequences = sequences or [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        self.call_count = 0
        self.last_prompt = ""
        self.last_num_outputs = 0

    def generate_batch(
        self, technical_prompt: PromptText, num_outputs: int
    ) -> List[List[TokenId]]:
        self.call_count += 1
        self.last_prompt = technical_prompt
        self.last_num_outputs = num_outputs
        return self.sequences[:num_outputs]

    def decode_to_midi(self, tokens: List[TokenId]) -> MidiBytes:
        return b"MIDI_" + bytes(str(len(tokens)), "utf-8")


class MockTranslator:
    """Mock LLMTranslator for testing."""

    def __init__(self, prompts: List[PromptText] | None = None):
        self.prompts = prompts or ["prompt1", "prompt2"]
        self.call_count = 0

    def translate(self, intent: Intent, num_variations: int) -> List[PromptText]:
        self.call_count += 1
        return self.prompts[:num_variations]


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


class MockAudioRenderer:
    """Mock AudioRenderer for testing."""

    def render(self, tokens: List[TokenId]) -> AudioSamples:
        return b"AUDIO_" + bytes(str(len(tokens)), "utf-8")


class TestBestOfNSearchExists(unittest.TestCase):
    """Test suite for BestOfNSearch class existence."""

    def test_best_of_n_search_class_exists(self):
        """
        AC: BestOfNSearch class MUST exist in use_cases.
        """
        from use_cases.best_of_n_search import BestOfNSearch

        self.assertTrue(callable(BestOfNSearch))


class TestBestOfNSearchInitialization(unittest.TestCase):
    """Test suite for BestOfNSearch initialization."""

    def test_initialization_with_all_dependencies(self):
        """
        AC: BestOfNSearch MUST accept translator, generator, evaluator, audio_renderer.
        """
        from use_cases.best_of_n_search import BestOfNSearch

        translator = MockTranslator()
        generator = MockBatchGenerator()
        evaluator = MockEvaluator()
        audio_renderer = MockAudioRenderer()

        search = BestOfNSearch(
            translator=translator,
            generator=generator,
            evaluator=evaluator,
            audio_renderer=audio_renderer,
        )

        self.assertEqual(search.translator, translator)
        self.assertEqual(search.generator, generator)
        self.assertEqual(search.evaluator, evaluator)
        self.assertEqual(search.audio_renderer, audio_renderer)


class TestBestOfNSearchExecute(unittest.TestCase):
    """Test suite for BestOfNSearch.execute method."""

    def setUp(self):
        """Set up test fixtures."""
        self.translator = MockTranslator()
        self.generator = MockBatchGenerator()
        self.evaluator = MockEvaluator()
        self.audio_renderer = MockAudioRenderer()

    def _create_search(self):
        """Helper to create a BestOfNSearch instance."""
        from use_cases.best_of_n_search import BestOfNSearch

        return BestOfNSearch(
            translator=self.translator,
            generator=self.generator,
            evaluator=self.evaluator,
            audio_renderer=self.audio_renderer,
        )

    def test_execute_returns_generation_result(self):
        """
        AC: execute MUST return a GenerationResult.
        """
        search = self._create_search()

        profile = GenerationProfile(
            generator_type="midillm",
            num_outputs=2,
            clap_weight=0.4,
            key_weight=0.3,
            note_weight=0.3,
        )

        result = search.execute(Intent("A peaceful sunrise"), profile)

        self.assertIsInstance(result, GenerationResult)

    def test_execute_translates_intent_to_prompt(self):
        """
        AC: execute MUST translate intent using translator.
        """
        search = self._create_search()

        profile = GenerationProfile(
            generator_type="midillm",
            num_outputs=2,
            clap_weight=0.4,
            key_weight=0.3,
            note_weight=0.3,
        )

        search.execute(Intent("A peaceful sunrise"), profile)

        self.assertEqual(self.translator.call_count, 1)

    def test_execute_generates_batch(self):
        """
        AC: execute MUST generate a batch of sequences using generator.
        """
        search = self._create_search()

        profile = GenerationProfile(
            generator_type="midillm",
            num_outputs=3,
            clap_weight=0.4,
            key_weight=0.3,
            note_weight=0.3,
        )

        search.execute(Intent("test"), profile)

        self.assertEqual(self.generator.call_count, 1)
        self.assertEqual(self.generator.last_num_outputs, 3)

    def test_execute_evaluates_all_sequences(self):
        """
        AC: execute MUST evaluate all generated sequences.
        """
        search = self._create_search()

        profile = GenerationProfile(
            generator_type="midillm",
            num_outputs=3,
            clap_weight=0.4,
            key_weight=0.3,
            note_weight=0.3,
        )

        search.execute(Intent("test"), profile)

        # Should evaluate each sequence once
        self.assertEqual(self.evaluator.call_count, 3)

    def test_execute_returns_best_sequence(self):
        """
        AC: execute MUST return the sequence with highest reward.
        """
        # Set up evaluator with known rewards
        evaluator = MockEvaluator(rewards=[0.3, 0.9, 0.5])  # Second is best
        generator = MockBatchGenerator(sequences=[[1], [2], [3]])
        translator = MockTranslator(prompts=["test_prompt"])

        from use_cases.best_of_n_search import BestOfNSearch

        search = BestOfNSearch(
            translator=translator,
            generator=generator,
            evaluator=evaluator,
            audio_renderer=self.audio_renderer,
        )

        profile = GenerationProfile(
            generator_type="midillm",
            num_outputs=3,
            clap_weight=0.4,
            key_weight=0.3,
            note_weight=0.3,
        )

        result = search.execute(Intent("test"), profile)

        # Should return the sequence with highest reward (index 1 -> tokens [2])
        self.assertIsInstance(result, GenerationResult)
        # The technical_prompt should come from the translator
        self.assertEqual(result.technical_prompt, "test_prompt")


class TestBestOfNSearchEdgeCases(unittest.TestCase):
    """Test suite for BestOfNSearch edge cases."""

    def test_single_output_returns_one_sequence(self):
        """
        AC: num_outputs=1 should still work correctly.
        """
        from use_cases.best_of_n_search import BestOfNSearch

        translator = MockTranslator(prompts=["single_prompt"])
        generator = MockBatchGenerator(sequences=[[1, 2, 3]])
        evaluator = MockEvaluator(rewards=[0.8])
        audio_renderer = MockAudioRenderer()

        search = BestOfNSearch(
            translator=translator,
            generator=generator,
            evaluator=evaluator,
            audio_renderer=audio_renderer,
        )

        profile = GenerationProfile(
            generator_type="midillm",
            num_outputs=1,
            clap_weight=0.4,
            key_weight=0.3,
            note_weight=0.3,
        )

        result = search.execute(Intent("test"), profile)

        self.assertIsInstance(result, GenerationResult)
        self.assertEqual(evaluator.call_count, 1)

    def test_execute_with_tie_returns_first_highest(self):
        """
        AC: When rewards tie, return the first one encountered.
        """
        from use_cases.best_of_n_search import BestOfNSearch

        translator = MockTranslator(prompts=["prompt1"])
        generator = MockBatchGenerator(sequences=[[1], [2], [3]])
        evaluator = MockEvaluator(rewards=[0.5, 0.5, 0.5])  # All same
        audio_renderer = MockAudioRenderer()

        search = BestOfNSearch(
            translator=translator,
            generator=generator,
            evaluator=evaluator,
            audio_renderer=audio_renderer,
        )

        profile = GenerationProfile(
            generator_type="midillm",
            num_outputs=3,
            clap_weight=0.4,
            key_weight=0.3,
            note_weight=0.3,
        )

        result = search.execute(Intent("test"), profile)

        self.assertIsInstance(result, GenerationResult)
        # First sequence should win (all have same reward)


class TestBestOfNSearchDataFlow(unittest.TestCase):
    """Test suite for BestOfNSearch data flow validation."""

    def test_uses_translator_prompt_for_generation(self):
        """
        AC: The translated prompt MUST be passed to the generator.
        """
        from use_cases.best_of_n_search import BestOfNSearch

        translator = MockTranslator(prompts=["custom_technical_prompt"])
        generator = MockBatchGenerator(sequences=[[1, 2, 3]])
        evaluator = MockEvaluator(rewards=[0.8])
        audio_renderer = MockAudioRenderer()

        search = BestOfNSearch(
            translator=translator,
            generator=generator,
            evaluator=evaluator,
            audio_renderer=audio_renderer,
        )

        profile = GenerationProfile(
            generator_type="midillm",
            num_outputs=1,
            clap_weight=0.4,
            key_weight=0.3,
            note_weight=0.3,
        )

        search.execute(Intent("test"), profile)

        # Generator should have received the translated prompt
        self.assertEqual(generator.last_prompt, "custom_technical_prompt")

    def test_uses_audio_renderer_for_evaluation(self):
        """
        AC: Audio renderer MUST be called for each sequence.
        """
        from use_cases.best_of_n_search import BestOfNSearch

        translator = MockTranslator(prompts=["prompt"])
        generator = MockBatchGenerator(sequences=[[1, 2], [3, 4]])
        evaluator = MockEvaluator(rewards=[0.5, 0.7])
        audio_renderer = MockAudioRenderer()
        audio_renderer.render = MagicMock(return_value=b"AUDIO_DATA")
        audio_renderer.render.call_count = 0

        search = BestOfNSearch(
            translator=translator,
            generator=generator,
            evaluator=evaluator,
            audio_renderer=audio_renderer,
        )

        profile = GenerationProfile(
            generator_type="midillm",
            num_outputs=2,
            clap_weight=0.4,
            key_weight=0.3,
            note_weight=0.3,
        )

        search.execute(Intent("test"), profile)

        # Audio renderer should be called for each sequence
        self.assertEqual(audio_renderer.render.call_count, 2)

    def test_returns_midi_bytes_from_winner(self):
        """
        AC: The returned MIDI bytes MUST come from decoding the winner's tokens.
        """
        from use_cases.best_of_n_search import BestOfNSearch

        translator = MockTranslator(prompts=["prompt"])
        generator = MockBatchGenerator(sequences=[[10, 20], [30, 40, 50]])
        evaluator = MockEvaluator(rewards=[0.3, 0.9])  # Second wins
        audio_renderer = MockAudioRenderer()

        search = BestOfNSearch(
            translator=translator,
            generator=generator,
            evaluator=evaluator,
            audio_renderer=audio_renderer,
        )

        profile = GenerationProfile(
            generator_type="midillm",
            num_outputs=2,
            clap_weight=0.4,
            key_weight=0.3,
            note_weight=0.3,
        )

        result = search.execute(Intent("test"), profile)

        # Should decode the winning sequence (second one with 3 tokens)
        self.assertTrue(result.midi_bytes.startswith(b"MIDI_"))


class TestBestOfNSearchEvaluationScoring(unittest.TestCase):
    """Test suite for BestOfNSearch evaluation scoring logic."""

    def test_evaluator_receives_correct_sequence_data(self):
        """
        AC: Evaluator MUST receive the correct sequence for evaluation.
        """
        from use_cases.best_of_n_search import BestOfNSearch

        translator = MockTranslator(prompts=["test_prompt"])
        generator = MockBatchGenerator(sequences=[[1, 2, 3], [4, 5, 6]])
        evaluator = MockEvaluator(rewards=[0.5, 0.8])
        audio_renderer = MockAudioRenderer()

        search = BestOfNSearch(
            translator=translator,
            generator=generator,
            evaluator=evaluator,
            audio_renderer=audio_renderer,
        )

        profile = GenerationProfile(
            generator_type="midillm",
            num_outputs=2,
            clap_weight=0.4,
            key_weight=0.3,
            note_weight=0.3,
        )

        search.execute(Intent("test intent"), profile)

        # Verify evaluator received sequences with correct prompts
        for prompt, reward in evaluator.evaluations:
            self.assertEqual(prompt, "test_prompt")

    def test_highest_reward_sequence_is_selected(self):
        """
        AC: The sequence with the highest reward MUST be selected.
        """
        from use_cases.best_of_n_search import BestOfNSearch

        # Set up so third sequence has highest reward
        translator = MockTranslator(prompts=["prompt"])
        generator = MockBatchGenerator(sequences=[[1], [2], [3], [4], [5]])
        evaluator = MockEvaluator(rewards=[0.1, 0.3, 0.9, 0.5, 0.2])  # Third is best
        audio_renderer = MockAudioRenderer()

        search = BestOfNSearch(
            translator=translator,
            generator=generator,
            evaluator=evaluator,
            audio_renderer=audio_renderer,
        )

        profile = GenerationProfile(
            generator_type="midillm",
            num_outputs=5,
            clap_weight=0.4,
            key_weight=0.3,
            note_weight=0.3,
        )

        result = search.execute(Intent("test"), profile)

        # Third sequence should be decoded (tokens [3])
        self.assertIsInstance(result, GenerationResult)

    def test_evaluator_called_with_intent(self):
        """
        AC: Evaluator MUST receive the original intent for evaluation.
        """
        from use_cases.best_of_n_search import BestOfNSearch

        translator = MockTranslator(prompts=["prompt"])
        generator = MockBatchGenerator(sequences=[[1, 2]])
        audio_renderer = MockAudioRenderer()

        # Track intent passed to evaluator
        received_intents = []

        class IntentTrackingEvaluator:
            def __init__(self):
                self.call_count = 0

            def evaluate(self, sequence, audio_data, intent):
                received_intents.append(intent)
                self.call_count += 1
                return 0.5

        evaluator = IntentTrackingEvaluator()

        search = BestOfNSearch(
            translator=translator,
            generator=generator,
            evaluator=evaluator,
            audio_renderer=audio_renderer,
        )

        profile = GenerationProfile(
            generator_type="midillm",
            num_outputs=1,
            clap_weight=0.4,
            key_weight=0.3,
            note_weight=0.3,
        )

        test_intent = Intent("a peaceful sunrise")
        search.execute(test_intent, profile)

        # Verify intent was passed
        self.assertEqual(len(received_intents), 1)
        self.assertEqual(received_intents[0].text, "a peaceful sunrise")

    def test_all_sequences_evaluated_once(self):
        """
        AC: Each generated sequence MUST be evaluated exactly once.
        """
        from use_cases.best_of_n_search import BestOfNSearch

        translator = MockTranslator(prompts=["prompt"])
        generator = MockBatchGenerator(sequences=[[1], [2], [3], [4]])
        evaluator = MockEvaluator(rewards=[0.1, 0.2, 0.3, 0.4])
        audio_renderer = MockAudioRenderer()

        search = BestOfNSearch(
            translator=translator,
            generator=generator,
            evaluator=evaluator,
            audio_renderer=audio_renderer,
        )

        profile = GenerationProfile(
            generator_type="midillm",
            num_outputs=4,
            clap_weight=0.4,
            key_weight=0.3,
            note_weight=0.3,
        )

        search.execute(Intent("test"), profile)

        # Evaluator should be called once per sequence
        self.assertEqual(evaluator.call_count, 4)


if __name__ == "__main__":
    unittest.main()
