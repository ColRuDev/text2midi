"""
Tests for domain entities - GenerationResult dataclass.

Tests validate the PRD 08 acceptance criteria for the GenerationResult entity.
"""

import unittest

from domain.entities import GenerationResult, MidiBytes, PromptText


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


if __name__ == "__main__":
    unittest.main()
