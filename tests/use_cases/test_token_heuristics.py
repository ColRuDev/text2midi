"""
Tests for TokenHeuristics use case.

Tests validate memory-based token evaluation for MIDI generation heuristics.
"""

import pytest

from domain.entities import Intent, MidiSequence
from domain.interfaces import AudioSamples


@pytest.fixture
def sample_vocab():
    """Create a sample inverted vocabulary for testing."""
    return {
        0: "PAD_None",
        1: "BOS_None",
        2: "EOS_None",
        3: "Bar_None",
        10: "Pitch_60",  # C4
        11: "Pitch_62",  # D4
        12: "Pitch_64",  # E4
        13: "Pitch_61",  # C#4 (not in C major)
        20: "Program_0",  # Piano
        21: "Program_24",  # Guitar
        22: "Program_-1",  # Drums
        30: "TimeSig_4/4",
        31: "TimeSig_3/4",
        40: "Velocity_64",
        50: "Duration_1.0",
    }


@pytest.fixture
def sample_intent():
    """Create a sample Intent."""
    return Intent(text="A peaceful piano melody")


@pytest.fixture
def sample_audio():
    """Create sample audio bytes."""
    return b"\x00\x00\x80?" * 100


class TestTokenHeuristicsInstantiation:
    """Tests for TokenHeuristics instantiation."""

    def test_token_heuristics_accepts_vocab_mapping(self, sample_vocab):
        """
        PRD 09 Scenario: Vocabulary injection
        GIVEN an inverted vocabulary mapping
        WHEN TokenHeuristics is instantiated
        THEN it MUST accept the vocabulary via constructor dependency injection
        """
        from use_cases.token_heuristics import TokenHeuristics

        evaluator = TokenHeuristics(vocab_mapping=sample_vocab)
        assert evaluator is not None

    def test_token_heuristics_implements_evaluator_interface(self, sample_vocab):
        """
        TokenHeuristics must implement the Evaluator interface.
        """
        from domain.interfaces import Evaluator
        from use_cases.token_heuristics import TokenHeuristics

        evaluator = TokenHeuristics(vocab_mapping=sample_vocab)
        assert isinstance(evaluator, Evaluator)


class TestTokenHeuristicsEvaluation:
    """Tests for TokenHeuristics.evaluate() method."""

    def test_evaluate_returns_float_score(
        self, sample_vocab, sample_intent, sample_audio
    ):
        """
        TokenHeuristics.evaluate() must return a float score.
        """
        from use_cases.token_heuristics import TokenHeuristics

        evaluator = TokenHeuristics(vocab_mapping=sample_vocab)
        sequence = MidiSequence(
            technical_prompt="tempo:80 key:C_major instruments:piano",
            tokens=[10, 11, 12],  # C, D, E (in C major)
        )

        score = evaluator.evaluate(sequence, sample_audio, sample_intent)
        assert isinstance(score, float)

    def test_evaluate_returns_score_in_valid_range(
        self, sample_vocab, sample_intent, sample_audio
    ):
        """
        Score must be in valid range [0, 1].
        """
        from use_cases.token_heuristics import TokenHeuristics

        evaluator = TokenHeuristics(vocab_mapping=sample_vocab)
        sequence = MidiSequence(
            technical_prompt="tempo:80 key:C_major instruments:piano",
            tokens=[10, 11, 12],
        )

        score = evaluator.evaluate(sequence, sample_audio, sample_intent)
        assert 0.0 <= score <= 1.0

    def test_evaluate_handles_empty_tokens(
        self, sample_vocab, sample_intent, sample_audio
    ):
        """
        TokenHeuristics must handle empty token sequences.
        """
        from use_cases.token_heuristics import TokenHeuristics

        evaluator = TokenHeuristics(vocab_mapping=sample_vocab)
        sequence = MidiSequence(
            technical_prompt="tempo:80 key:C_major",
            tokens=[],
        )

        score = evaluator.evaluate(sequence, sample_audio, sample_intent)
        assert isinstance(score, float)


class TestTokenHeuristicsKeyScalePenalty:
    """Tests for key/scale violation penalties."""

    def test_in_key_tokens_score_higher_than_out_of_key(
        self, sample_vocab, sample_intent, sample_audio
    ):
        """
        PRD 09 Scenario: Branch violates key/scale constraints
        GIVEN a prompt specifying key:C_major
        WHEN a branch produces tokens in the key vs out of the key
        THEN the in-key branch MUST score higher
        """
        from use_cases.token_heuristics import TokenHeuristics

        evaluator = TokenHeuristics(vocab_mapping=sample_vocab)

        # In C major (C, D, E)
        in_key_sequence = MidiSequence(
            technical_prompt="tempo:80 key:C_major instruments:piano",
            tokens=[10, 11, 12, 10, 11, 12],  # C, D, E repeated
        )

        # Out of C major (contains C#)
        out_of_key_sequence = MidiSequence(
            technical_prompt="tempo:80 key:C_major instruments:piano",
            tokens=[10, 13, 12, 10, 13, 12],  # C, C#, E (C# not in C major)
        )

        in_key_score = evaluator.evaluate(in_key_sequence, sample_audio, sample_intent)
        out_of_key_score = evaluator.evaluate(
            out_of_key_sequence, sample_audio, sample_intent
        )

        assert in_key_score > out_of_key_score


class TestTokenHeuristicsTimeSigPenalty:
    """Tests for time signature violation penalties."""

    def test_timesig_violation_applies_penalty(
        self, sample_vocab, sample_intent, sample_audio
    ):
        """
        PRD 09 Scenario: Branch violates time signature constraints
        GIVEN a prompt specifying initial TimeSig_4/4
        WHEN a branch produces tokens with inconsistent time signature
        THEN the system MUST apply an exponential penalty
        """
        from use_cases.token_heuristics import TokenHeuristics

        evaluator = TokenHeuristics(vocab_mapping=sample_vocab)

        # With matching time signature
        matching_sequence = MidiSequence(
            technical_prompt="tempo:80 key:C_major timesig:4/4 instruments:piano",
            tokens=[30, 10, 11, 12],  # TimeSig_4/4, C, D, E
        )

        # With different time signature
        different_sequence = MidiSequence(
            technical_prompt="tempo:80 key:C_major timesig:4/4 instruments:piano",
            tokens=[31, 10, 11, 12],  # TimeSig_3/4 instead of 4/4
        )

        matching_score = evaluator.evaluate(matching_sequence, sample_audio, sample_intent)
        different_score = evaluator.evaluate(different_sequence, sample_audio, sample_intent)

        # Matching time signature should score higher
        assert matching_score >= different_score


class TestTokenHeuristicsStrictInstruments:
    """Tests for strict instrument control."""

    def test_strict_instruments_true_penalizes_unrequested(
        self, sample_vocab, sample_intent, sample_audio
    ):
        """
        PRD 09 Scenario: Strict instrument control is enabled
        GIVEN a GenerationProfile with strict_instruments=True
        WHEN generating tokens with unrequested instruments
        THEN the system MUST severely penalize that branch
        """
        from use_cases.token_heuristics import TokenHeuristics

        evaluator = TokenHeuristics(
            vocab_mapping=sample_vocab,
            strict_instruments=True,
            requested_programs={0},  # Only piano requested
        )

        # Has only requested instrument (piano)
        correct_sequence = MidiSequence(
            technical_prompt="tempo:80 key:C_major instruments:piano",
            tokens=[20, 10, 11, 12],  # Program_0 (piano), C, D, E
        )

        # Has unrequested instrument (guitar)
        wrong_sequence = MidiSequence(
            technical_prompt="tempo:80 key:C_major instruments:piano",
            tokens=[21, 10, 11, 12],  # Program_24 (guitar) - not requested!
        )

        correct_score = evaluator.evaluate(correct_sequence, sample_audio, sample_intent)
        wrong_score = evaluator.evaluate(wrong_sequence, sample_audio, sample_intent)

        # Correct instrument should score much higher
        assert correct_score > wrong_score

    def test_strict_instruments_false_ignores_unrequested(
        self, sample_vocab, sample_intent, sample_audio
    ):
        """
        PRD 09 Scenario: Strict instrument control is disabled
        GIVEN a GenerationProfile with strict_instruments=False
        WHEN generating tokens with unrequested instruments
        THEN the system MUST NOT penalize branches for instrument mismatch
        """
        from use_cases.token_heuristics import TokenHeuristics

        evaluator = TokenHeuristics(
            vocab_mapping=sample_vocab,
            strict_instruments=False,
            requested_programs={0},  # Piano requested
        )

        # Has only requested instrument
        correct_sequence = MidiSequence(
            technical_prompt="tempo:80 key:C_major instruments:piano",
            tokens=[20, 10, 11, 12],
        )

        # Has unrequested instrument
        wrong_sequence = MidiSequence(
            technical_prompt="tempo:80 key:C_major instruments:piano",
            tokens=[21, 10, 11, 12],  # Guitar - not requested
        )

        correct_score = evaluator.evaluate(correct_sequence, sample_audio, sample_intent)
        wrong_score = evaluator.evaluate(wrong_sequence, sample_audio, sample_intent)

        # Scores should be similar (no penalty for wrong instrument)
        # Allow some tolerance for other factors
        assert abs(correct_score - wrong_score) < 0.3


class TestTokenHeuristicsParsing:
    """Tests for parsing technical prompt constraints."""

    def test_parse_key_from_prompt(self, sample_vocab):
        """
        TokenHeuristics must parse key information from technical prompt.
        """
        from use_cases.token_heuristics import TokenHeuristics

        evaluator = TokenHeuristics(vocab_mapping=sample_vocab)

        # Parse C major
        key_info = evaluator._parse_key_from_prompt("tempo:80 key:C_major instruments:piano")
        assert key_info["root"] == 0  # C = 0
        assert key_info["scale"] == "major"

    def test_parse_minor_key_from_prompt(self, sample_vocab):
        """
        TokenHeuristics must parse minor key from technical prompt.
        """
        from use_cases.token_heuristics import TokenHeuristics

        evaluator = TokenHeuristics(vocab_mapping=sample_vocab)

        key_info = evaluator._parse_key_from_prompt("tempo:80 key:A_minor instruments:piano")
        assert key_info["root"] == 9  # A = 9
        assert key_info["scale"] == "minor"

    def test_parse_key_from_natural_language_prompt(self, sample_vocab):
        """
        TokenHeuristics must parse key from natural language prompt.
        """
        from use_cases.token_heuristics import TokenHeuristics

        evaluator = TokenHeuristics(vocab_mapping=sample_vocab)

        key_info = evaluator._parse_key_from_prompt("The song is in the key of C Major with a 4/4 time signature")
        assert key_info["root"] == 0
        assert key_info["scale"] == "major"

    def test_parse_instruments_from_prompt(self, sample_vocab):
        """
        TokenHeuristics must parse instruments from technical prompt.
        """
        from use_cases.token_heuristics import TokenHeuristics

        evaluator = TokenHeuristics(vocab_mapping=sample_vocab)

        programs = evaluator._parse_programs_from_prompt("instruments:piano,guitar")
        # Piano = 0, Guitar = 24 in General MIDI
        assert 0 in programs or "piano" in programs

    def test_parse_instruments_from_natural_language_prompt(self, sample_vocab):
        """
        TokenHeuristics must parse instruments from natural language prompt.
        """
        from use_cases.token_heuristics import TokenHeuristics

        evaluator = TokenHeuristics(vocab_mapping=sample_vocab)

        programs = evaluator._parse_programs_from_prompt("featuring a soft acoustic grand piano and strings")
        assert 0 in programs  # acoustic piano
        assert 48 in programs  # strings

    def test_parse_timesig_from_prompt(self, sample_vocab):
        """
        TokenHeuristics must parse time signature from technical prompt.
        """
        from use_cases.token_heuristics import TokenHeuristics

        evaluator = TokenHeuristics(vocab_mapping=sample_vocab)

        timesig = evaluator._parse_timesig_from_prompt("tempo:80 timesig:4/4")
        assert timesig == "4/4"

    def test_parse_timesig_from_natural_language_prompt(self, sample_vocab):
        """
        TokenHeuristics must parse time signature from natural language prompt.
        """
        from use_cases.token_heuristics import TokenHeuristics

        evaluator = TokenHeuristics(vocab_mapping=sample_vocab)

        timesig = evaluator._parse_timesig_from_prompt("with a 3/4 time signature and a slow tempo")
        assert timesig == "3/4"
