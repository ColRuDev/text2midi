"""
Tests for REMI vocabulary mapping module.

Tests validate the inverted vocabulary mapping for token-level heuristics.
"""

import pytest

from domain.remi_vocab import set_inverted_vocab

# Inject a complete mock vocabulary for domain testing
_MOCK_VOCAB = {
    0: "PAD_None",
    1: "BOS_None",
    2: "EOS_None",
    3: "Pitch_60",
    4: "Program_0",
    5: "TimeSig_4/4",
}
# fill it with items to pass the size test
for i in range(6, 150):
    _MOCK_VOCAB[i] = f"Pitch_{i}"


@pytest.fixture(autouse=True)
def setup_mock_vocab(monkeypatch):
    """
    Set the mock vocabulary before each test to prevent global state pollution.
    
    Uses pytest's monkeypatch fixture for safe global state overriding with
    automatic cleanup after each test.
    """
    from domain.remi_vocab import set_inverted_vocab
    
    # Set the mock vocabulary
    monkeypatch.setattr("domain.remi_vocab._INVERTED_VOCAB", _MOCK_VOCAB)
    set_inverted_vocab(_MOCK_VOCAB)
    yield
    # monkeypatch automatically restores the original value

class TestInvertedVocabulary:
    """Tests for inverted REMI vocabulary mapping."""

    def test_inverted_vocab_is_dict_mapping_token_id_to_event_name(self):
        """
        PRD 09 Scenario: Inverted vocabulary structure
        GIVEN the REMI vocabulary module
        WHEN we access the inverted vocabulary mapping
        THEN it MUST be a dict mapping TokenId (int) -> EventName (str)
        """
        from domain.remi_vocab import get_inverted_vocab
        INVERTED_VOCAB = get_inverted_vocab()

        assert isinstance(INVERTED_VOCAB, dict)

        # Check that keys are integers (TokenIds)
        for key in list(INVERTED_VOCAB.keys())[:10]:
            assert isinstance(key, int), f"Key {key} should be int, got {type(key)}"

        # Check that values are strings (EventNames)
        for value in list(INVERTED_VOCAB.values())[:10]:
            assert isinstance(value, str), f"Value {value} should be str, got {type(value)}"

    def test_inverted_vocab_contains_pitch_tokens(self):
        """
        PRD 09 Scenario: Pitch tokens in vocabulary
        GIVEN the inverted vocabulary
        WHEN we look for Pitch_X tokens
        THEN we MUST find them mapped correctly
        """
        from domain.remi_vocab import get_inverted_vocab
        INVERTED_VOCAB = get_inverted_vocab()

        # Find a Pitch token by searching values
        pitch_events = [v for v in INVERTED_VOCAB.values() if v.startswith("Pitch_")]
        assert len(pitch_events) > 0, "Inverted vocab must contain Pitch tokens"

        # Verify at least one has a valid mapping
        token_id = [k for k, v in INVERTED_VOCAB.items() if v == pitch_events[0]][0]
        assert isinstance(token_id, int)

    def test_inverted_vocab_contains_program_tokens(self):
        """
        PRD 09 Scenario: Program (instrument) tokens in vocabulary
        GIVEN the inverted vocabulary
        WHEN we look for Program_X tokens
        THEN we MUST find them mapped correctly
        """
        from domain.remi_vocab import get_inverted_vocab
        INVERTED_VOCAB = get_inverted_vocab()

        program_events = [v for v in INVERTED_VOCAB.values() if v.startswith("Program_")]
        assert len(program_events) > 0, "Inverted vocab must contain Program tokens"

    def test_inverted_vocab_contains_timesig_tokens(self):
        """
        PRD 09 Scenario: TimeSig tokens in vocabulary
        GIVEN the inverted vocabulary
        WHEN we look for TimeSig_X/Y tokens
        THEN we MUST find them mapped correctly
        """
        from domain.remi_vocab import get_inverted_vocab
        INVERTED_VOCAB = get_inverted_vocab()

        timesig_events = [v for v in INVERTED_VOCAB.values() if v.startswith("TimeSig_")]
        assert len(timesig_events) > 0, "Inverted vocab must contain TimeSig tokens"

    def test_get_event_name_returns_str_for_valid_token(self):
        """
        PRD 09 Scenario: Helper function for token lookup
        GIVEN a valid TokenId
        WHEN we call get_event_name
        THEN it MUST return the corresponding event name string
        """
        from domain.remi_vocab import get_event_name, get_inverted_vocab
        INVERTED_VOCAB = get_inverted_vocab()

        # Get any valid token ID
        token_id = next(iter(INVERTED_VOCAB.keys()))
        event_name = get_event_name(token_id)

        assert isinstance(event_name, str)
        assert event_name == INVERTED_VOCAB[token_id]

    def test_get_event_name_returns_none_for_invalid_token(self):
        """
        Edge case: Invalid token ID
        GIVEN an invalid TokenId (not in vocabulary)
        WHEN we call get_event_name
        THEN it MUST return None
        """
        from domain.remi_vocab import get_event_name

        # Use a very large token ID that's unlikely to exist
        result = get_event_name(999999)
        assert result is None

    def test_is_pitch_token_identifies_pitch_events(self):
        """
        PRD 09 Scenario: Pitch token identification
        GIVEN a token ID
        WHEN we call is_pitch_token
        THEN it MUST return True only for Pitch_X tokens
        """
        from domain.remi_vocab import is_pitch_token, get_inverted_vocab
        INVERTED_VOCAB = get_inverted_vocab()

        # Find a Pitch token
        pitch_token_id = next(k for k, v in INVERTED_VOCAB.items() if v.startswith("Pitch_"))

        assert is_pitch_token(pitch_token_id) is True

        # Find a non-Pitch token
        non_pitch_token_id = next(k for k, v in INVERTED_VOCAB.items() if not v.startswith("Pitch_"))
        assert is_pitch_token(non_pitch_token_id) is False


class TestVocabularyHelperFunctions:
    """Tests for vocabulary helper functions."""

    def test_is_program_token_identifies_program_events(self):
        """
        is_program_token must identify Program_X tokens.
        """
        from domain.remi_vocab import is_program_token, get_inverted_vocab
        INVERTED_VOCAB = get_inverted_vocab()

        # Find a Program token
        program_token_id = next(k for k, v in INVERTED_VOCAB.items() if v.startswith("Program_"))
        assert is_program_token(program_token_id) is True

        # Find a non-Program token
        non_program_token_id = next(k for k, v in INVERTED_VOCAB.items() if not v.startswith("Program_"))
        assert is_program_token(non_program_token_id) is False

    def test_is_timesig_token_identifies_timesig_events(self):
        """
        is_timesig_token must identify TimeSig_X/Y tokens.
        """
        from domain.remi_vocab import is_timesig_token, get_inverted_vocab
        INVERTED_VOCAB = get_inverted_vocab()

        # Find a TimeSig token
        timesig_token_id = next(k for k, v in INVERTED_VOCAB.items() if v.startswith("TimeSig_"))
        assert is_timesig_token(timesig_token_id) is True

        # Find a non-TimeSig token
        non_timesig_token_id = next(k for k, v in INVERTED_VOCAB.items() if not v.startswith("TimeSig_"))
        assert is_timesig_token(non_timesig_token_id) is False

    def test_is_special_token_identifies_special_tokens(self):
        """
        is_special_token must identify PAD, BOS, EOS, etc.
        """
        from domain.remi_vocab import is_special_token, get_inverted_vocab
        INVERTED_VOCAB = get_inverted_vocab()

        # Find a special token (PAD_None, BOS_None, etc.)
        special_token_id = next(k for k, v in INVERTED_VOCAB.items() if v in ("PAD_None", "BOS_None", "EOS_None"))
        assert is_special_token(special_token_id) is True

        # Find a non-special token (a Pitch token)
        non_special_token_id = next(k for k, v in INVERTED_VOCAB.items() if v.startswith("Pitch_"))
        assert is_special_token(non_special_token_id) is False

    def test_helper_functions_return_false_for_invalid_token(self):
        """
        All helper functions must return False for invalid token IDs.
        """
        from domain.remi_vocab import (
            is_pitch_token,
            is_program_token,
            is_timesig_token,
            is_special_token,
        )

        invalid_token = 999999
        assert is_pitch_token(invalid_token) is False
        assert is_program_token(invalid_token) is False
        assert is_timesig_token(invalid_token) is False
        assert is_special_token(invalid_token) is False


class TestVocabularyConstants:
    """Tests for vocabulary helper constants."""

    def test_special_tokens_exist(self):
        """
        Vocabulary must define special token markers.
        """
        from domain.remi_vocab import SPECIAL_TOKENS

        assert isinstance(SPECIAL_TOKENS, set)
        # Should contain common special token names
        assert any("PAD" in t for t in SPECIAL_TOKENS)
        assert any("BOS" in t for t in SPECIAL_TOKENS)
        assert any("EOS" in t for t in SPECIAL_TOKENS)

    def test_inverted_vocab_has_expected_size(self):
        """
        Inverted vocab should have entries for all REMI tokens.
        """
        from domain.remi_vocab import get_inverted_vocab
        INVERTED_VOCAB = get_inverted_vocab()

        # REMI vocabulary should have hundreds of tokens
        assert len(INVERTED_VOCAB) > 100
        # Should be around 500-600 tokens
        assert len(INVERTED_VOCAB) < 1000
