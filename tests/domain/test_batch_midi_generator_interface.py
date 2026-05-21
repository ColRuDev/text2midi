"""
Tests for BatchMidiGenerator interface - PRD batch-generation.

Tests validate the interface contract for batch-oriented MIDI generation
as specified in the SDD delta specs.
"""

import unittest
from abc import ABC
from typing import List

from domain.entities import MidiBytes, PromptText, TokenId


class TestBatchMidiGeneratorInterface(unittest.TestCase):
    """Test suite for BatchMidiGenerator interface contract."""

    def test_batch_midi_generator_interface_exists(self):
        """
        AC: BatchMidiGenerator interface MUST exist in domain.interfaces.
        """
        from domain.interfaces import BatchMidiGenerator

        self.assertTrue(issubclass(BatchMidiGenerator, ABC))

    def test_batch_midi_generator_has_generate_batch_method(self):
        """
        AC: BatchMidiGenerator MUST have generate_batch method.
        
        Signature: generate_batch(technical_prompt: PromptText, num_outputs: int) -> List[List[TokenId]]
        """
        from domain.interfaces import BatchMidiGenerator

        self.assertTrue(hasattr(BatchMidiGenerator, "generate_batch"))
        
        # Verify it's an abstract method
        import inspect
        method = getattr(BatchMidiGenerator, "generate_batch")
        self.assertTrue(getattr(method, "__isabstractmethod__", False))

    def test_batch_midi_generator_has_decode_to_midi_method(self):
        """
        AC: BatchMidiGenerator MUST have decode_to_midi method.
        
        Signature: decode_to_midi(tokens: List[TokenId]) -> MidiBytes
        """
        from domain.interfaces import BatchMidiGenerator

        self.assertTrue(hasattr(BatchMidiGenerator, "decode_to_midi"))
        
        # Verify it's an abstract method
        import inspect
        method = getattr(BatchMidiGenerator, "decode_to_midi")
        self.assertTrue(getattr(method, "__isabstractmethod__", False))

    def test_batch_midi_generator_cannot_be_instantiated_directly(self):
        """
        AC: BatchMidiGenerator is abstract and cannot be instantiated directly.
        """
        from domain.interfaces import BatchMidiGenerator

        with self.assertRaises(TypeError):
            BatchMidiGenerator()


class TestBatchMidiGeneratorImplementation(unittest.TestCase):
    """Tests for implementing BatchMidiGenerator interface."""

    def test_concrete_implementation_must_implement_all_methods(self):
        """
        AC: Concrete implementations MUST implement all abstract methods.
        """
        from domain.interfaces import BatchMidiGenerator

        # Incomplete implementation - missing decode_to_midi
        class IncompleteGenerator(BatchMidiGenerator):
            def generate_batch(
                self, technical_prompt: PromptText, num_outputs: int
            ) -> List[List[TokenId]]:
                return [[1, 2, 3] for _ in range(num_outputs)]

        with self.assertRaises(TypeError):
            IncompleteGenerator()

    def test_complete_implementation_can_be_instantiated(self):
        """
        AC: Complete implementations CAN be instantiated.
        """
        from domain.interfaces import BatchMidiGenerator

        class CompleteGenerator(BatchMidiGenerator):
            def generate_batch(
                self, technical_prompt: PromptText, num_outputs: int
            ) -> List[List[TokenId]]:
                return [[1, 2, 3] for _ in range(num_outputs)]

            def decode_to_midi(self, tokens: List[TokenId]) -> MidiBytes:
                return b"MIDI_DATA"

        generator = CompleteGenerator()
        self.assertIsInstance(generator, BatchMidiGenerator)

    def test_generate_batch_returns_correct_number_of_sequences(self):
        """
        AC: generate_batch MUST return exactly num_outputs sequences.
        """
        from domain.interfaces import BatchMidiGenerator

        class MockGenerator(BatchMidiGenerator):
            def generate_batch(
                self, technical_prompt: PromptText, num_outputs: int
            ) -> List[List[TokenId]]:
                return [[i * 10 + j for j in range(5)] for i in range(num_outputs)]

            def decode_to_midi(self, tokens: List[TokenId]) -> MidiBytes:
                return b"MIDI_DATA"

        generator = MockGenerator()
        
        # Test with different num_outputs values
        for num in [1, 3, 5, 10]:
            with self.subTest(num_outputs=num):
                result = generator.generate_batch("test prompt", num)
                self.assertEqual(len(result), num)

    def test_generate_batch_returns_list_of_token_lists(self):
        """
        AC: generate_batch MUST return List[List[TokenId]].
        """
        from domain.interfaces import BatchMidiGenerator

        class MockGenerator(BatchMidiGenerator):
            def generate_batch(
                self, technical_prompt: PromptText, num_outputs: int
            ) -> List[List[TokenId]]:
                return [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

            def decode_to_midi(self, tokens: List[TokenId]) -> MidiBytes:
                return b"MIDI_DATA"

        generator = MockGenerator()
        result = generator.generate_batch("test prompt", 3)

        self.assertIsInstance(result, list)
        for seq in result:
            self.assertIsInstance(seq, list)
            for token in seq:
                self.assertIsInstance(token, int)

    def test_decode_to_midi_returns_bytes(self):
        """
        AC: decode_to_midi MUST return MidiBytes (bytes).
        """
        from domain.interfaces import BatchMidiGenerator

        class MockGenerator(BatchMidiGenerator):
            def generate_batch(
                self, technical_prompt: PromptText, num_outputs: int
            ) -> List[List[TokenId]]:
                return [[1, 2, 3] for _ in range(num_outputs)]

            def decode_to_midi(self, tokens: List[TokenId]) -> MidiBytes:
                return b"MIDI_" + bytes(tokens)

        generator = MockGenerator()
        result = generator.decode_to_midi([60, 64, 67])

        self.assertIsInstance(result, bytes)


class TestBatchMidiGeneratorMockIntegration(unittest.TestCase):
    """Tests for BatchMidiGenerator mock integration and token handling."""

    def test_mock_generator_handles_empty_prompt(self):
        """
        AC: Mock generator should handle empty prompts gracefully.
        """
        from domain.interfaces import BatchMidiGenerator

        class MockGenerator(BatchMidiGenerator):
            def generate_batch(
                self, technical_prompt: PromptText, num_outputs: int
            ) -> List[List[TokenId]]:
                return [[1, 2] for _ in range(num_outputs)]

            def decode_to_midi(self, tokens: List[TokenId]) -> MidiBytes:
                return b"MIDI"

        generator = MockGenerator()
        result = generator.generate_batch("", 1)

        self.assertEqual(len(result), 1)

    def test_mock_generator_token_ids_are_integers(self):
        """
        AC: Token IDs MUST be integers.
        """
        from domain.interfaces import BatchMidiGenerator

        class MockGenerator(BatchMidiGenerator):
            def generate_batch(
                self, technical_prompt: PromptText, num_outputs: int
            ) -> List[List[TokenId]]:
                return [[1, 2, 3, 4, 5]]

            def decode_to_midi(self, tokens: List[TokenId]) -> MidiBytes:
                return b"MIDI"

        generator = MockGenerator()
        result = generator.generate_batch("test", 1)

        for token in result[0]:
            self.assertIsInstance(token, int)

    def test_mock_generator_with_large_batch(self):
        """
        AC: Generator should handle large num_outputs values.
        """
        from domain.interfaces import BatchMidiGenerator

        class MockGenerator(BatchMidiGenerator):
            def generate_batch(
                self, technical_prompt: PromptText, num_outputs: int
            ) -> List[List[TokenId]]:
                return [[i] for i in range(num_outputs)]

            def decode_to_midi(self, tokens: List[TokenId]) -> MidiBytes:
                return b"MIDI"

        generator = MockGenerator()
        result = generator.generate_batch("test", 100)

        self.assertEqual(len(result), 100)

    def test_mock_decode_produces_non_empty_bytes(self):
        """
        AC: decode_to_midi MUST produce non-empty bytes for valid tokens.
        """
        from domain.interfaces import BatchMidiGenerator

        class MockGenerator(BatchMidiGenerator):
            def generate_batch(
                self, technical_prompt: PromptText, num_outputs: int
            ) -> List[List[TokenId]]:
                return [[1, 2, 3]]

            def decode_to_midi(self, tokens: List[TokenId]) -> MidiBytes:
                # Simulate realistic MIDI output
                return b"MThd" + bytes(tokens)

        generator = MockGenerator()
        result = generator.decode_to_midi([60, 64, 67])

        self.assertGreater(len(result), 0)

    def test_mock_generator_with_special_token_ids(self):
        """
        AC: Generator should handle special token IDs (BOS, EOS, PAD).
        """
        from domain.interfaces import BatchMidiGenerator

        # Standard special tokens
        BOS, EOS, PAD = 1, 2, 0

        class MockGenerator(BatchMidiGenerator):
            def generate_batch(
                self, technical_prompt: PromptText, num_outputs: int
            ) -> List[List[TokenId]]:
                # Simulate sequences with BOS/EOS wrapping
                return [[BOS, 10, 20, 30, EOS] for _ in range(num_outputs)]

            def decode_to_midi(self, tokens: List[TokenId]) -> MidiBytes:
                return b"MIDI"

        generator = MockGenerator()
        result = generator.generate_batch("test", 3)

        for seq in result:
            self.assertEqual(seq[0], BOS)
            self.assertEqual(seq[-1], EOS)


if __name__ == "__main__":
    unittest.main()
