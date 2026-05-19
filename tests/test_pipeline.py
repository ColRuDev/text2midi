"""
Tests for Text2MidiPipeline.

Tests validate the DI container behavior and MIDI generation orchestration.
"""

import unittest
from typing import List
from unittest.mock import MagicMock, patch

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


class MockTranslator:
    """Mock LLMTranslator for testing."""

    def __init__(self, prompts: List[PromptText] | None = None):
        self.prompts = prompts or ["prompt1", "prompt2"]
        self.call_count = 0

    def translate(self, intent: Intent, num_variations: int) -> List[PromptText]:
        self.call_count += 1
        return self.prompts[:num_variations]


class MockGenerator:
    """Mock MidiGenerator for testing."""

    def __init__(self):
        self.call_count = 0

    def generate_step(
        self,
        technical_prompt: PromptText,
        current_tokens: List[TokenId],
        num_tokens: int,
    ) -> List[TokenId]:
        self.call_count += 1
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

    def __init__(self, reward: float = 0.7):
        self.reward = reward
        self.call_count = 0

    def evaluate(
        self,
        sequence: MidiSequence,
        audio_data: AudioSamples,
        intent: Intent,
    ) -> float:
        self.call_count += 1
        return self.reward


class TestText2MidiPipelineInit(unittest.TestCase):
    """Test suite for Text2MidiPipeline initialization."""

    def test_pipeline_class_exists(self):
        """
        AC2.1: Text2MidiPipeline class must exist.
        """
        from pipeline import Text2MidiPipeline

        self.assertTrue(callable(Text2MidiPipeline))

    def test_pipeline_initializes_all_adapters(self):
        """
        AC2.2: Pipeline __init__ must instantiate all heavy adapters.
        """
        from pipeline import Text2MidiPipeline

        # Mock the adapter constructors
        with (
            patch("pipeline.GoogleAITranslator") as mock_translator_cls,
            patch("pipeline.Text2MidiGenerator") as mock_generator_cls,
            patch("pipeline.CompositeEvaluator") as mock_evaluator_cls,
            patch("pipeline.InMemoryFluidSynthEngine") as mock_audio_cls,
        ):
            mock_translator_cls.return_value = MockTranslator()
            mock_generator_cls.return_value = MockGenerator()
            mock_evaluator_cls.return_value = MockEvaluator()
            mock_audio_cls.return_value = MockAudioRenderer()

            pipeline = Text2MidiPipeline()

            # Verify all adapters were instantiated
            mock_translator_cls.assert_called_once()
            mock_generator_cls.assert_called_once()
            mock_evaluator_cls.assert_called_once()
            mock_audio_cls.assert_called_once()

    def test_pipeline_stores_adapters_as_attributes(self):
        """
        AC2.2: Pipeline must store adapters for reuse.
        """
        from pipeline import Text2MidiPipeline

        mock_translator = MockTranslator()
        mock_generator = MockGenerator()
        mock_evaluator = MockEvaluator()
        mock_audio = MockAudioRenderer()

        with (
            patch("pipeline.GoogleAITranslator", return_value=mock_translator),
            patch("pipeline.Text2MidiGenerator", return_value=mock_generator),
            patch("pipeline.CompositeEvaluator", return_value=mock_evaluator),
            patch("pipeline.InMemoryFluidSynthEngine", return_value=mock_audio),
        ):
            pipeline = Text2MidiPipeline()

            self.assertIs(pipeline._translator, mock_translator)
            self.assertIs(pipeline._generator, mock_generator)
            self.assertIs(pipeline._evaluator, mock_evaluator)
            self.assertIs(pipeline._audio_renderer, mock_audio)


class TestText2MidiPipelineGenerate(unittest.TestCase):
    """Test suite for Text2MidiPipeline.generate method."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_translator = MockTranslator()
        self.mock_generator = MockGenerator()
        self.mock_evaluator = MockEvaluator()
        self.mock_audio = MockAudioRenderer()

    def _create_pipeline(self):
        """Helper to create a pipeline with mocked adapters."""
        from pipeline import Text2MidiPipeline

        with (
            patch("pipeline.GoogleAITranslator", return_value=self.mock_translator),
            patch("pipeline.Text2MidiGenerator", return_value=self.mock_generator),
            patch("pipeline.CompositeEvaluator", return_value=self.mock_evaluator),
            patch("pipeline.InMemoryFluidSynthEngine", return_value=self.mock_audio),
        ):
            return Text2MidiPipeline()

    def test_generate_method_exists(self):
        """
        AC2.3: Pipeline must have a generate method.
        """
        pipeline = self._create_pipeline()

        self.assertTrue(hasattr(pipeline, "generate"))
        self.assertTrue(callable(pipeline.generate))

    def test_generate_returns_generation_result(self):
        """
        PRD 08: generate must return a GenerationResult.
        """
        from config.profiles import BALANCED

        pipeline = self._create_pipeline()

        result = pipeline.generate("A peaceful sunrise", BALANCED)

        self.assertIsInstance(result, GenerationResult)
        self.assertTrue(result.midi_bytes.startswith(b"MIDI_DATA_"))
        self.assertIsInstance(result.technical_prompt, str)

    def test_generate_returns_midi_bytes_in_result(self):
        """
        PRD 08: generate must return GenerationResult with valid midi_bytes.
        """
        from config.profiles import BALANCED

        pipeline = self._create_pipeline()

        result = pipeline.generate("A peaceful sunrise", BALANCED)

        self.assertIsInstance(result.midi_bytes, bytes)
        self.assertTrue(result.midi_bytes.startswith(b"MIDI_DATA_"))

    def test_generate_uses_progressive_search(self):
        """
        AC2.3: generate must orchestrate ProgressiveSearch.
        """
        from config.profiles import BALANCED

        pipeline = self._create_pipeline()

        result = pipeline.generate("A peaceful sunrise", BALANCED)

        # Verify all adapters were used
        self.assertGreater(self.mock_translator.call_count, 0)
        self.assertGreater(self.mock_generator.call_count, 0)
        self.assertGreater(self.mock_evaluator.call_count, 0)

    def test_generate_with_one_shot_profile(self):
        """
        AC2.3: generate must work with ONE_SHOT profile.
        """
        from config.profiles import ONE_SHOT

        pipeline = self._create_pipeline()

        result = pipeline.generate("Quick melody", ONE_SHOT)

        self.assertIsInstance(result, GenerationResult)

    def test_generate_with_deep_search_profile(self):
        """
        AC2.3: generate must work with DEEP_SEARCH profile.
        """
        from config.profiles import DEEP_SEARCH

        pipeline = self._create_pipeline()

        # Reduce max_tokens for faster test
        small_deep = GenerationProfile(
            token_batch_size=DEEP_SEARCH.token_batch_size,
            num_beams=2,  # Fewer beams for speed
            top_k=DEEP_SEARCH.top_k,
            max_tokens=100,  # Small for testing
            clap_weight=DEEP_SEARCH.clap_weight,
            key_weight=DEEP_SEARCH.key_weight,
            note_weight=DEEP_SEARCH.note_weight,
        )

        result = pipeline.generate("Complex symphony", small_deep)

        self.assertIsInstance(result, GenerationResult)


class TestText2MidiPipelineReuse(unittest.TestCase):
    """Test suite for verifying pipeline reuses adapters."""

    def test_pipeline_reuses_same_adapters_across_calls(self):
        """
        AC2.2: Pipeline must reuse the same adapter instances.
        """
        from pipeline import Text2MidiPipeline

        mock_translator = MockTranslator()
        mock_generator = MockGenerator()
        mock_evaluator = MockEvaluator()
        mock_audio = MockAudioRenderer()

        with (
            patch("pipeline.GoogleAITranslator", return_value=mock_translator),
            patch("pipeline.Text2MidiGenerator", return_value=mock_generator),
            patch("pipeline.CompositeEvaluator", return_value=mock_evaluator),
            patch("pipeline.InMemoryFluidSynthEngine", return_value=mock_audio),
        ):
            pipeline = Text2MidiPipeline()

            # Use valid profile weights
            profile = GenerationProfile(
                token_batch_size=10,
                num_beams=2,
                top_k=1,
                max_tokens=30,
                clap_weight=0.4,
                key_weight=0.3,
                note_weight=0.3,
            )

            # First call
            result1 = pipeline.generate("First melody", profile)
            self.assertIsInstance(result1, GenerationResult)
            first_translator_count = mock_translator.call_count
            first_generator_count = mock_generator.call_count

            # Second call - should reuse same instances
            result2 = pipeline.generate("Second melody", profile)
            self.assertIsInstance(result2, GenerationResult)

            # Translator should have been called twice (once per generate)
            self.assertEqual(mock_translator.call_count, first_translator_count + 1)
            # Generator should have been called more
            self.assertGreater(mock_generator.call_count, first_generator_count)


if __name__ == "__main__":
    unittest.main()
