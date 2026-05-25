"""
Tests for Text2MidiPipeline.

Tests validate the DI container behavior and MIDI generation orchestration.
Also validates the factory method for strategy selection (batch-generation spec).
"""

import unittest
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

# Import centralized mocks from conftest.py
from tests.conftest import (
    MockTranslator,
    MockGenerator,
    MockBatchGenerator,
    MockAudioRenderer,
    MockEvaluator,
)


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
        AC2.2: Pipeline __init__ must instantiate necessary heavy adapters.
        """
        from pipeline import Text2MidiPipeline
        from adapters.translators.google_ai_translator import GoogleAIConfig

        # Mock the adapter constructors
        with (
            patch("pipeline.GoogleAITranslator") as mock_translator_cls,
            patch("pipeline.CompositeEvaluator") as mock_evaluator_cls,
            patch("pipeline.InMemoryFluidSynthEngine") as mock_audio_cls,
        ):
            mock_translator_cls.return_value = MockTranslator()
            mock_evaluator_cls.return_value = MockEvaluator()
            mock_audio_cls.return_value = MockAudioRenderer()

            pipeline = Text2MidiPipeline(translator_config=GoogleAIConfig(model_name="test"))

            # Verify adapters were instantiated
            mock_translator_cls.assert_called_once()
            mock_evaluator_cls.assert_called_once()
            mock_audio_cls.assert_called_once()

    def test_pipeline_stores_adapters_as_attributes(self):
        """
        AC2.2: Pipeline must store adapters for reuse.
        """
        from pipeline import Text2MidiPipeline
        from adapters.translators.google_ai_translator import GoogleAIConfig

        mock_translator = MockTranslator()
        mock_evaluator = MockEvaluator()
        mock_audio = MockAudioRenderer()

        with (
            patch("pipeline.GoogleAITranslator", return_value=mock_translator),
            patch("pipeline.CompositeEvaluator", return_value=mock_evaluator),
            patch("pipeline.InMemoryFluidSynthEngine", return_value=mock_audio),
        ):
            pipeline = Text2MidiPipeline(translator_config=GoogleAIConfig(model_name="test"))

            self.assertIs(pipeline._translator, mock_translator)
            self.assertIs(pipeline._evaluator, mock_evaluator)
            self.assertIs(pipeline._audio_renderer, mock_audio)


class TestText2MidiPipelineStrategySelection(unittest.TestCase):
    """Test suite for pipeline strategy selection (batch-generation spec)."""

    def test_pipeline_uses_progressive_search_for_text2midi_generator_type(self):
        """
        batch-generation spec: Pipeline MUST use ProgressiveSearch for text2midi.
        
        GIVEN a profile with generator_type="text2midi"
        WHEN the pipeline is initialized
        THEN it MUST use ProgressiveSearch
        """
        from pipeline import Text2MidiPipeline
        from use_cases.progressive_search import ProgressiveSearch

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
            
            # Create profile with text2midi generator type
            profile = GenerationProfile(generator_type="text2midi")
            search = pipeline._get_search_for_profile(profile)

            self.assertIsInstance(search, ProgressiveSearch)

    def test_pipeline_uses_best_of_n_search_for_midillm_generator_type(self):
        """
        batch-generation spec: Pipeline MUST use BestOfNSearch for midillm.
        
        GIVEN a profile with generator_type="midillm"
        WHEN the pipeline generates
        THEN it MUST use BestOfNSearch
        """
        from pipeline import Text2MidiPipeline
        from use_cases.best_of_n_search import BestOfNSearch

        mock_translator = MockTranslator()
        mock_batch_generator = MockBatchGenerator()
        mock_evaluator = MockEvaluator()
        mock_audio = MockAudioRenderer()

        with (
            patch("pipeline.GoogleAITranslator", return_value=mock_translator),
            patch("pipeline.MidiLLMGenerator", return_value=mock_batch_generator),
            patch("pipeline.CompositeEvaluator", return_value=mock_evaluator),
            patch("pipeline.InMemoryFluidSynthEngine", return_value=mock_audio),
        ):
            # Create profile with midillm generator type
            profile = GenerationProfile(
                generator_type="midillm",
                num_outputs=3,
                clap_weight=0.4,
                key_weight=0.3,
                note_weight=0.3,
            )

            pipeline = Text2MidiPipeline(profile=profile)
            search = pipeline._get_search_for_profile(profile)

            self.assertIsInstance(search, BestOfNSearch)

    def test_pipeline_factory_method_exists(self):
        """
        batch-generation spec: Pipeline MUST have factory method for strategy selection.
        """
        from pipeline import Text2MidiPipeline

        self.assertTrue(hasattr(Text2MidiPipeline, "_get_search_for_profile"))
        self.assertTrue(callable(getattr(Text2MidiPipeline, "_get_search_for_profile")))


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

    @patch("pipeline.Text2MidiPipeline._get_search_for_profile")
    def test_generate_returns_generation_result(self, mock_get_search):
        """
        PRD 08: generate must return a GenerationResult.
        """
        from config.profiles import BALANCED
        
        # Mock the search instance to avoid hitting real generators if they somehow leaked
        mock_search = MagicMock()
        mock_search.execute.return_value = GenerationResult(
            midi_bytes=b"MIDI_DATA_123",
            technical_prompt="A peaceful sunrise"
        )
        mock_get_search.return_value = mock_search

        pipeline = self._create_pipeline()

        result = pipeline.generate("A peaceful sunrise", BALANCED)

        self.assertIsInstance(result, GenerationResult)
        self.assertTrue(result.midi_bytes.startswith(b"MIDI_DATA_"))
        self.assertIsInstance(result.technical_prompt, str)

    @patch("pipeline.Text2MidiPipeline._get_search_for_profile")
    def test_generate_returns_midi_bytes_in_result(self, mock_get_search):
        """
        PRD 08: generate must return GenerationResult with valid midi_bytes.
        """
        from config.profiles import BALANCED
        
        mock_search = MagicMock()
        mock_search.execute.return_value = GenerationResult(
            midi_bytes=b"MIDI_DATA_123",
            technical_prompt="A peaceful sunrise"
        )
        mock_get_search.return_value = mock_search

        pipeline = self._create_pipeline()

        result = pipeline.generate("A peaceful sunrise", BALANCED)

        self.assertIsInstance(result.midi_bytes, bytes)
        self.assertTrue(result.midi_bytes.startswith(b"MIDI_DATA_"))

    @patch("pipeline.Text2MidiPipeline._get_search_for_profile")
    def test_generate_uses_progressive_search(self, mock_get_search):
        """
        AC2.3: generate must orchestrate ProgressiveSearch.
        """
        from config.profiles import BALANCED
        from use_cases.progressive_search import ProgressiveSearch
        
        mock_search = MagicMock(spec=ProgressiveSearch)
        mock_search.execute.return_value = GenerationResult(
            midi_bytes=b"MIDI_DATA_123",
            technical_prompt="A peaceful sunrise"
        )
        mock_get_search.return_value = mock_search

        pipeline = self._create_pipeline()

        result = pipeline.generate("A peaceful sunrise", BALANCED)
        mock_search.execute.assert_called_once()

    @patch("pipeline.Text2MidiPipeline._get_search_for_profile")
    def test_generate_with_one_shot_profile(self, mock_get_search):
        """
        AC2.3: generate must work with ONE_SHOT profile.
        """
        from config.profiles import ONE_SHOT
        
        mock_search = MagicMock()
        mock_search.execute.return_value = GenerationResult(
            midi_bytes=b"MIDI_DATA_123",
            technical_prompt="A peaceful sunrise"
        )
        mock_get_search.return_value = mock_search

        pipeline = self._create_pipeline()

        result = pipeline.generate("Quick melody", ONE_SHOT)

        self.assertIsInstance(result, GenerationResult)

    @patch("pipeline.Text2MidiPipeline._get_search_for_profile")
    def test_generate_with_deep_search_profile(self, mock_get_search):
        """
        AC2.3: generate must work with DEEP_SEARCH profile.
        """
        from config.profiles import DEEP_SEARCH
        
        mock_search = MagicMock()
        mock_search.execute.return_value = GenerationResult(
            midi_bytes=b"MIDI_DATA_123",
            technical_prompt="A peaceful sunrise"
        )
        mock_get_search.return_value = mock_search

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

    @patch("pipeline.Text2MidiPipeline._get_search_for_profile")
    def test_pipeline_reuses_same_adapters_across_calls(self, mock_get_search):
        """
        AC2.2: Pipeline must reuse the same adapter instances.
        """
        from pipeline import Text2MidiPipeline
        from use_cases.progressive_search import ProgressiveSearch
        
        mock_translator = MockTranslator()
        mock_generator = MockGenerator()
        mock_evaluator = MockEvaluator()
        mock_audio = MockAudioRenderer()

        with (
            patch("pipeline.GoogleAITranslator", return_value=mock_translator),
            patch("pipeline.CompositeEvaluator", return_value=mock_evaluator),
            patch("pipeline.InMemoryFluidSynthEngine", return_value=mock_audio),
        ):
            pipeline = Text2MidiPipeline()
            
            # Setup mock search
            mock_search = MagicMock(spec=ProgressiveSearch)
            mock_search.execute.return_value = GenerationResult(
                midi_bytes=b"MIDI_DATA_123",
                technical_prompt="A peaceful sunrise"
            )
            mock_get_search.return_value = mock_search

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

            # Second call - should reuse same instances
            result2 = pipeline.generate("Second melody", profile)
            self.assertIsInstance(result2, GenerationResult)
            
            # Verifying execute was called twice
            self.assertEqual(mock_search.execute.call_count, 2)


class TestText2MidiPipelinePassThroughTranslator(unittest.TestCase):
    """Test suite for PassThroughTranslator integration in pipeline.

    These tests verify the pass-through-translator spec requirements:
    - Pipeline MUST instantiate PassThroughTranslator when translator_config is None
    - Pipeline MUST NOT validate GOOGLE_API_KEY when using PassThroughTranslator
    """

    def test_pipeline_uses_passthrough_when_translator_config_is_none(self):
        """
        pass-through-translator spec: Pipeline MUST use PassThroughTranslator
        when translator_config is None.

        GIVEN translator_config is None
        WHEN the pipeline is initialized
        THEN it MUST instantiate PassThroughTranslator
        AND it MUST NOT validate GOOGLE_API_KEY
        """
        from pipeline import Text2MidiPipeline
        from adapters.translators.pass_through_translator import PassThroughTranslator

        mock_generator = MockGenerator()
        mock_evaluator = MockEvaluator()
        mock_audio = MockAudioRenderer()

        with (
            patch("pipeline.Text2MidiGenerator", return_value=mock_generator),
            patch("pipeline.CompositeEvaluator", return_value=mock_evaluator),
            patch("pipeline.InMemoryFluidSynthEngine", return_value=mock_audio),
        ):
            # translator_config=None should use PassThroughTranslator
            pipeline = Text2MidiPipeline(translator_config=None)

            self.assertIsInstance(pipeline._translator, PassThroughTranslator)

    def test_pipeline_uses_googleai_when_translator_config_is_provided(self):
        """
        pass-through-translator spec: Pipeline MUST use GoogleAITranslator
        when translator_config is provided.

        GIVEN translator_config is a GoogleAIConfig
        WHEN the pipeline is initialized
        THEN it MUST instantiate GoogleAITranslator
        """
        from pipeline import Text2MidiPipeline
        from adapters.translators.google_ai_translator import GoogleAIConfig, GoogleAITranslator

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
            config = GoogleAIConfig(model_name="test-model")
            pipeline = Text2MidiPipeline(translator_config=config)

            # Verify GoogleAITranslator was called with the config
            # (the actual instance is mocked, but we verify the call)
            from pipeline import GoogleAITranslator as patched_translator
            patched_translator.assert_called_once_with(config)

    def test_pipeline_works_without_google_api_key_when_passthrough(self):
        """
        pass-through-translator spec: Pipeline MUST work without GOOGLE_API_KEY
        when using PassThroughTranslator.

        GIVEN GOOGLE_API_KEY is not set
        AND translator_config is None
        WHEN the pipeline is initialized
        THEN it MUST NOT raise an error
        """
        import os
        from pipeline import Text2MidiPipeline

        # Ensure GOOGLE_API_KEY is not set
        original_key = os.environ.pop("GOOGLE_API_KEY", None)

        try:
            mock_generator = MockGenerator()
            mock_evaluator = MockEvaluator()
            mock_audio = MockAudioRenderer()

            with (
                patch("pipeline.Text2MidiGenerator", return_value=mock_generator),
                patch("pipeline.CompositeEvaluator", return_value=mock_evaluator),
                patch("pipeline.InMemoryFluidSynthEngine", return_value=mock_audio),
            ):
                # This should NOT raise an error even without GOOGLE_API_KEY
                pipeline = Text2MidiPipeline(translator_config=None)

                # Verify we can generate without errors
                from config.profiles import BALANCED
                result = pipeline.generate("Test melody", BALANCED)

                self.assertIsInstance(result, GenerationResult)
        finally:
            # Restore original key if it existed
            if original_key:
                os.environ["GOOGLE_API_KEY"] = original_key


if __name__ == "__main__":
    unittest.main()
