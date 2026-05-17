"""
Unit tests for Evaluator adapters.

Tests for ClapEvaluator, HeuristicsEvaluator, and CompositeEvaluator.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from adapters.evaluators.clap_evaluator import ClapEvaluator
from adapters.evaluators.heuristics import HeuristicsEvaluator
from adapters.evaluators.composite import CompositeEvaluator
from domain.entities import GenerationProfile, Intent, MidiSequence
from domain.interfaces import AudioSamples


class TestClapEvaluator:
    """Tests for ClapEvaluator."""

    @pytest.fixture
    def mock_clap_module(self):
        """Mock the CLAP module."""
        with patch("adapters.evaluators.clap_evaluator.clap") as mock:
            # Setup CLAP class mock
            mock_clap_instance = MagicMock()
            mock.CLAP.return_value = mock_clap_instance
            mock_clap_instance.get_audio_embeddings.return_value = MagicMock()
            mock_clap_instance.get_text_embeddings.return_value = MagicMock()
            mock_clap_instance.compute_similarity.return_value = MagicMock(
                cpu=lambda: MagicMock(item=lambda: 0.75)
            )
            yield mock

    @pytest.fixture
    def mock_clap_available(self):
        """Mock CLAP_AVAILABLE as True."""
        with patch("adapters.evaluators.clap_evaluator.CLAP_AVAILABLE", True):
            yield

    @pytest.fixture
    def sample_sequence(self):
        """Create a sample MidiSequence."""
        return MidiSequence(
            technical_prompt="tempo:80 key:C_major instruments:piano",
            tokens=[60, 64, 67, 72],
        )

    @pytest.fixture
    def sample_intent(self):
        """Create a sample Intent."""
        return Intent(text="A peaceful piano melody")

    @pytest.fixture
    def sample_audio(self):
        """Create sample audio bytes."""
        return b"\x00\x00\x80?" * 1000  # Mock float32 audio

    def test_clap_evaluator_returns_float_score(
        self, mock_clap_module, mock_clap_available, sample_sequence, sample_audio, sample_intent
    ):
        """ClapEvaluator returns a float score between 0 and 1."""
        evaluator = ClapEvaluator()
        score = evaluator.evaluate(sample_sequence, sample_audio, sample_intent)
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_clap_evaluator_uses_technical_prompt_by_default(
        self, mock_clap_module, mock_clap_available, sample_sequence, sample_audio, sample_intent
    ):
        """ClapEvaluator uses technical_prompt for CLAP evaluation by default."""
        evaluator = ClapEvaluator()
        evaluator.evaluate(sample_sequence, sample_audio, sample_intent)
        
        # Verify text embeddings were requested
        mock_clap_instance = mock_clap_module.CLAP.return_value
        mock_clap_instance.get_text_embeddings.assert_called()

    def test_clap_evaluator_can_use_original_intent(
        self, mock_clap_module, mock_clap_available, sample_sequence, sample_audio, sample_intent
    ):
        """ClapEvaluator can use original intent for evaluation."""
        evaluator = ClapEvaluator()
        evaluator.set_clap_prompt_source("original")
        evaluator.evaluate(sample_sequence, sample_audio, sample_intent)
        
        # Verify text embeddings were requested
        mock_clap_instance = mock_clap_module.CLAP.return_value
        mock_clap_instance.get_text_embeddings.assert_called()

    def test_clap_evaluator_handles_missing_model_gracefully(self, sample_sequence, sample_audio, sample_intent):
        """ClapEvaluator handles missing CLAP model gracefully."""
        with patch("adapters.evaluators.clap_evaluator.clap", None):
            evaluator = ClapEvaluator()
            score = evaluator.evaluate(sample_sequence, sample_audio, sample_intent)
            
            # Should return a fallback score
            assert isinstance(score, float)
            assert score == 0.5  # Default fallback


class TestHeuristicsEvaluator:
    """Tests for HeuristicsEvaluator."""

    @pytest.fixture
    def sample_sequence(self):
        """Create a sample MidiSequence."""
        return MidiSequence(
            technical_prompt="tempo:80 key:C_major instruments:piano",
            tokens=[60, 64, 67, 72, 76],
        )

    @pytest.fixture
    def sample_audio(self):
        """Create sample audio bytes."""
        return b"\x00\x00\x80?" * 1000

    @pytest.fixture
    def sample_intent(self):
        """Create a sample Intent."""
        return Intent(text="A peaceful piano melody")

    def test_heuristics_evaluator_returns_float_score(
        self, sample_sequence, sample_audio, sample_intent
    ):
        """HeuristicsEvaluator returns a float score."""
        evaluator = HeuristicsEvaluator()
        score = evaluator.evaluate(sample_sequence, sample_audio, sample_intent)
        
        assert isinstance(score, float)

    def test_heuristics_evaluator_analyzes_key_consistency(
        self, sample_sequence, sample_audio, sample_intent
    ):
        """HeuristicsEvaluator analyzes key consistency."""
        evaluator = HeuristicsEvaluator()
        
        # Sequence with C-major notes (60=C, 64=E, 67=G, 72=C)
        score = evaluator.evaluate(sample_sequence, sample_audio, sample_intent)
        
        assert isinstance(score, float)

    def test_heuristics_evaluator_handles_empty_tokens(
        self, sample_audio, sample_intent
    ):
        """HeuristicsEvaluator handles empty token list."""
        sequence = MidiSequence(
            technical_prompt="tempo:80 key:C_major",
            tokens=[],
        )
        
        evaluator = HeuristicsEvaluator()
        score = evaluator.evaluate(sequence, sample_audio, sample_intent)
        
        # Should return low score for empty sequence
        assert isinstance(score, float)

    def test_heuristics_evaluator_scores_key_match(
        self, sample_audio, sample_intent
    ):
        """HeuristicsEvaluator scores based on key match."""
        # C-major sequence (C, E, G notes)
        c_major_sequence = MidiSequence(
            technical_prompt="tempo:80 key:C_major",
            tokens=[60, 64, 67],  # C, E, G - C major triad
        )
        
        # Non-C-major sequence (C#, D#, F# - not in C major)
        non_c_major_sequence = MidiSequence(
            technical_prompt="tempo:80 key:C_major",
            tokens=[61, 63, 66],  # C#, D#, F# - not in C major
        )
        
        evaluator = HeuristicsEvaluator()
        c_major_score = evaluator.evaluate(c_major_sequence, sample_audio, sample_intent)
        non_c_major_score = evaluator.evaluate(non_c_major_sequence, sample_audio, sample_intent)
        
        # C-major sequence should score higher for key consistency
        assert c_major_score > non_c_major_score


class TestCompositeEvaluator:
    """Tests for CompositeEvaluator."""

    @pytest.fixture
    def mock_clap(self):
        """Create a mocked ClapEvaluator."""
        mock = Mock()
        mock.evaluate.return_value = 0.8
        return mock

    @pytest.fixture
    def mock_heuristics(self):
        """Create a mocked HeuristicsEvaluator."""
        mock = Mock()
        mock.evaluate.return_value = 0.6
        return mock

    @pytest.fixture
    def sample_sequence(self):
        """Create a sample MidiSequence."""
        return MidiSequence(
            technical_prompt="tempo:80 key:C_major instruments:piano",
            tokens=[60, 64, 67],
        )

    @pytest.fixture
    def sample_audio(self):
        """Create sample audio bytes."""
        return b"\x00\x00\x80?" * 1000

    @pytest.fixture
    def sample_intent(self):
        """Create a sample Intent."""
        return Intent(text="A peaceful piano melody")

    def test_composite_evaluator_combines_scores(
        self, mock_clap, mock_heuristics, sample_sequence, sample_audio, sample_intent
    ):
        """CompositeEvaluator combines CLAP and heuristics scores."""
        profile = GenerationProfile(
            clap_weight=0.6,
            key_weight=0.2,
            note_weight=0.2,
        )
        
        evaluator = CompositeEvaluator(
            clap_evaluator=mock_clap,
            heuristics_evaluator=mock_heuristics,
            profile=profile,
        )
        
        score = evaluator.evaluate(sample_sequence, sample_audio, sample_intent)
        
        # Score should be weighted combination
        # clap: 0.8 * 0.6 = 0.48
        # heuristics: 0.6 * (0.2 + 0.2) = 0.24
        # Total: 0.48 + 0.24 = 0.72
        expected = 0.8 * 0.6 + 0.6 * 0.4
        assert abs(score - expected) < 0.01

    def test_composite_evaluator_uses_generation_profile_weights(
        self, mock_clap, mock_heuristics, sample_sequence, sample_audio, sample_intent
    ):
        """CompositeEvaluator uses weights from GenerationProfile."""
        profile = GenerationProfile(
            clap_weight=1.0,
            key_weight=0.0,
            note_weight=0.0,
        )
        
        evaluator = CompositeEvaluator(
            clap_evaluator=mock_clap,
            heuristics_evaluator=mock_heuristics,
            profile=profile,
        )
        
        score = evaluator.evaluate(sample_sequence, sample_audio, sample_intent)
        
        # With clap_weight=1.0, score should equal clap score
        assert score == 0.8

    def test_composite_evaluator_calls_both_evaluators(
        self, mock_clap, mock_heuristics, sample_sequence, sample_audio, sample_intent
    ):
        """CompositeEvaluator calls both evaluators."""
        profile = GenerationProfile(
            clap_weight=0.5,
            key_weight=0.25,
            note_weight=0.25,
        )
        
        evaluator = CompositeEvaluator(
            clap_evaluator=mock_clap,
            heuristics_evaluator=mock_heuristics,
            profile=profile,
        )
        
        evaluator.evaluate(sample_sequence, sample_audio, sample_intent)
        
        mock_clap.evaluate.assert_called_once_with(
            sample_sequence, sample_audio, sample_intent
        )
        mock_heuristics.evaluate.assert_called_once_with(
            sample_sequence, sample_audio, sample_intent
        )

    def test_composite_evaluator_returns_zero_on_empty_evaluators(
        self, sample_sequence, sample_audio, sample_intent
    ):
        """CompositeEvaluator handles missing evaluators gracefully."""
        profile = GenerationProfile(
            clap_weight=0.5,
            key_weight=0.25,
            note_weight=0.25,
        )
        
        evaluator = CompositeEvaluator(
            clap_evaluator=None,
            heuristics_evaluator=None,
            profile=profile,
        )
        
        score = evaluator.evaluate(sample_sequence, sample_audio, sample_intent)
        
        # Should return 0.0 when no evaluators available
        assert score == 0.0

    def test_composite_evaluator_respects_clap_prompt_source(
        self, mock_clap, mock_heuristics, sample_sequence, sample_audio, sample_intent
    ):
        """CompositeEvaluator propagates clap_prompt_source to ClapEvaluator."""
        profile = GenerationProfile(
            clap_weight=0.5,
            key_weight=0.25,
            note_weight=0.25,
        )
        
        evaluator = CompositeEvaluator(
            clap_evaluator=mock_clap,
            heuristics_evaluator=mock_heuristics,
            profile=profile,
        )
        
        evaluator.set_clap_prompt_source("original")
        
        assert evaluator.clap_prompt_source == "original"
