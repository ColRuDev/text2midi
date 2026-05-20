"""
Tests for domain entities - GenerationResult dataclass and GenerationProfile.

Tests validate the PRD 08 acceptance criteria for the GenerationResult entity
and PRD 09 acceptance criteria for GenerationProfile strict_instruments toggle.
"""

import unittest

from domain.entities import GenerationProfile, GenerationResult, MidiBytes, PromptText


class TestGenerationResult(unittest.TestCase):
    """Test suite for GenerationResult dataclass."""

    def test_generation_result_stores_midi_bytes_and_technical_prompt(self):
        """
        PRD 08 Scenario: Initialization
        GIVEN valid midi_bytes (bytes) and a technical_prompt (str)
        WHEN a GenerationResult is instantiated
        THEN it MUST store both values correctly
        """
        midi_bytes: MidiBytes = b"MIDI_BINARY_DATA_12345"
        technical_prompt: PromptText = "C major scale, piano, tempo 120 BPM"

        result = GenerationResult(
            midi_bytes=midi_bytes,
            technical_prompt=technical_prompt,
        )

        self.assertEqual(result.midi_bytes, midi_bytes)
        self.assertEqual(result.technical_prompt, technical_prompt)

    def test_generation_result_is_frozen_immutable(self):
        """
        GenerationResult should be immutable (frozen dataclass)
        to match the pattern of other domain entities like Intent.
        """
        result = GenerationResult(
            midi_bytes=b"test_midi",
            technical_prompt="test prompt",
        )

        with self.assertRaises(AttributeError):
            result.midi_bytes = b"new_midi"

        with self.assertRaises(AttributeError):
            result.technical_prompt = "new prompt"

    def test_generation_result_with_empty_prompt(self):
        """
        Edge case: GenerationResult should accept empty technical_prompt.
        """
        result = GenerationResult(
            midi_bytes=b"test_midi",
            technical_prompt="",
        )

        self.assertEqual(result.midi_bytes, b"test_midi")
        self.assertEqual(result.technical_prompt, "")

    def test_generation_result_with_empty_midi_bytes(self):
        """
        Edge case: GenerationResult should accept empty midi_bytes.
        """
        result = GenerationResult(
            midi_bytes=b"",
            technical_prompt="test prompt",
        )

        self.assertEqual(result.midi_bytes, b"")
        self.assertEqual(result.technical_prompt, "test prompt")


class TestGenerationProfileStrictInstruments(unittest.TestCase):
    """Test suite for GenerationProfile strict_instruments toggle (PRD 09)."""

    def test_generation_profile_has_strict_instruments_field(self):
        """
        PRD 09 Scenario: Instantiating profile with strict instruments
        GIVEN a request for a generation profile
        WHEN the profile is created
        THEN it MUST allow setting `strict_instruments` to `True` or `False`
        """
        profile = GenerationProfile(strict_instruments=True)
        self.assertTrue(profile.strict_instruments)

        profile_false = GenerationProfile(strict_instruments=False)
        self.assertFalse(profile_false.strict_instruments)

    def test_generation_profile_strict_instruments_defaults_to_false(self):
        """
        PRD 09 Scenario: Default value for strict_instruments
        GIVEN a GenerationProfile created without explicit strict_instruments
        WHEN the profile is instantiated
        THEN strict_instruments MUST default to `False`
        """
        profile = GenerationProfile()
        self.assertFalse(profile.strict_instruments)

    def test_generation_profile_with_strict_instruments_true(self):
        """
        PRD 09 Scenario: Strict instrument control is enabled
        GIVEN a GenerationProfile with strict_instruments=True
        WHEN generating tokens
        THEN the system MUST severely penalize branches with unrequested instruments
        """
        profile = GenerationProfile(strict_instruments=True)
        self.assertTrue(profile.strict_instruments)

    def test_generation_profile_strict_instruments_is_bool(self):
        """
        strict_instruments must be a boolean type.
        """
        profile = GenerationProfile(strict_instruments=False)
        self.assertIsInstance(profile.strict_instruments, bool)


if __name__ == "__main__":
    unittest.main()
