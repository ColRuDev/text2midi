"""
Unit tests for MidiLLMGenerator adapter.

Tests validate the batch-generation specification for the MidiLLMGenerator
which implements BatchMidiGenerator interface.
"""

import unittest
from typing import List
from unittest.mock import MagicMock, patch

from domain.entities import MidiBytes, PromptText, TokenId


class TestMidiLLMGeneratorExists(unittest.TestCase):
    """Test suite for MidiLLMGenerator class existence."""

    def test_midillm_generator_class_exists(self):
        """
        AC: MidiLLMGenerator class MUST exist in adapters.generators.
        """
        from adapters.generators.midillm_generator import MidiLLMGenerator

        self.assertTrue(callable(MidiLLMGenerator))

    def test_midillm_generator_implements_batch_interface(self):
        """
        AC: MidiLLMGenerator MUST implement BatchMidiGenerator interface.
        """
        from adapters.generators.midillm_generator import MidiLLMGenerator
        from domain.interfaces import BatchMidiGenerator

        self.assertTrue(issubclass(MidiLLMGenerator, BatchMidiGenerator))


class TestMidiLLMGeneratorConfig(unittest.TestCase):
    """Test suite for MidiLLMGeneratorConfig."""

    def test_config_class_exists(self):
        """
        AC: MidiLLMGeneratorConfig MUST exist for configuration.
        """
        from adapters.generators.midillm_generator import MidiLLMGeneratorConfig

        self.assertTrue(callable(MidiLLMGeneratorConfig))

    def test_config_has_model_name_field(self):
        """
        AC: Config MUST have model_name field for model selection.
        """
        from adapters.generators.midillm_generator import MidiLLMGeneratorConfig

        config = MidiLLMGeneratorConfig(model_name="test-model")
        self.assertEqual(config.model_name, "test-model")

    def test_config_has_backend_field(self):
        """
        AC: Config MUST have backend field for vLLM/Transformers selection.
        """
        from adapters.generators.midillm_generator import MidiLLMGeneratorConfig

        config = MidiLLMGeneratorConfig(backend="vllm")
        self.assertEqual(config.backend, "vllm")

    def test_config_defaults(self):
        """
        AC: Config MUST have sensible defaults.
        """
        from adapters.generators.midillm_generator import MidiLLMGeneratorConfig

        config = MidiLLMGeneratorConfig()
        self.assertIsNotNone(config.model_name)
        self.assertIsNotNone(config.backend)


class TestMidiLLMGeneratorMethods(unittest.TestCase):
    """Test suite for MidiLLMGenerator methods."""

    def test_generate_batch_method_exists(self):
        """
        AC: MidiLLMGenerator MUST have generate_batch method.
        """
        from adapters.generators.midillm_generator import MidiLLMGenerator

        self.assertTrue(hasattr(MidiLLMGenerator, "generate_batch"))

    def test_decode_to_midi_method_exists(self):
        """
        AC: MidiLLMGenerator MUST have decode_to_midi method.
        """
        from adapters.generators.midillm_generator import MidiLLMGenerator

        self.assertTrue(hasattr(MidiLLMGenerator, "decode_to_midi"))


class TestMidiLLMGeneratorMocked(unittest.TestCase):
    """Test suite for MidiLLMGenerator with mocked backend."""

    def test_generate_batch_returns_list_of_token_lists(self):
        """
        AC: generate_batch MUST return List[List[TokenId]].
        """
        from adapters.generators.midillm_generator import (
            MidiLLMGenerator,
            MidiLLMGeneratorConfig,
        )

        config = MidiLLMGeneratorConfig(
            model_name="test-model",
            backend="mock",  # Use mock backend for testing
        )

        with patch.object(MidiLLMGenerator, "_load_model"):
            with patch.object(MidiLLMGenerator, "_load_tokenizer"):
                generator = MidiLLMGenerator(config)
                generator._model = MagicMock()
                generator._tokenizer = MagicMock()
                generator._midi_vocab = {"PAD": 0, "BOS": 1, "EOS": 2}
                generator._inv_midi_vocab = {0: "PAD", 1: "BOS", 2: "EOS"}

                # Mock the internal generation
                generator._generate_tokens = MagicMock(
                    return_value=[[1, 10, 20, 2], [1, 30, 40, 2], [1, 50, 60, 2]]
                )

                result = generator.generate_batch("test prompt", num_outputs=3)

                self.assertIsInstance(result, list)
                self.assertEqual(len(result), 3)
                for seq in result:
                    self.assertIsInstance(seq, list)

    def test_decode_to_midi_returns_bytes(self):
        """
        AC: decode_to_midi MUST return MidiBytes.
        """
        from adapters.generators.midillm_generator import (
            MidiLLMGenerator,
            MidiLLMGeneratorConfig,
        )

        config = MidiLLMGeneratorConfig(
            model_name="test-model",
            backend="mock",
        )

        with patch.object(MidiLLMGenerator, "_load_model"):
            with patch.object(MidiLLMGenerator, "_load_tokenizer"):
                generator = MidiLLMGenerator(config)
                generator._model = MagicMock()
                generator._tokenizer = MagicMock()
                generator._midi_vocab = {"PAD": 0, "BOS": 1, "EOS": 2}
                generator._inv_midi_vocab = {0: "PAD", 1: "BOS", 2: "EOS"}
                generator._midi_tokenizer = None

                # Mock decode method
                generator._decode_tokens = MagicMock(return_value=b"MIDI_DATA")

                result = generator.decode_to_midi([1, 10, 20, 2])

                self.assertIsInstance(result, bytes)


class TestMidiLLMGeneratorBatchBehavior(unittest.TestCase):
    """Test suite for MidiLLMGenerator batch generation behavior."""

    def test_generate_batch_returns_correct_number_of_sequences(self):
        """
        AC: generate_batch MUST return exactly num_outputs sequences.
        """
        from adapters.generators.midillm_generator import (
            MidiLLMGenerator,
            MidiLLMGeneratorConfig,
        )

        config = MidiLLMGeneratorConfig(backend="mock")

        with patch.object(MidiLLMGenerator, "_load_model"):
            with patch.object(MidiLLMGenerator, "_load_tokenizer"):
                generator = MidiLLMGenerator(config)
                generator._model = MagicMock()
                generator._tokenizer = MagicMock()
                generator._midi_vocab = {"PAD": 0, "BOS": 1, "EOS": 2}
                generator._inv_midi_vocab = {0: "PAD", 1: "BOS", 2: "EOS"}

                for num in [1, 3, 5]:
                    with self.subTest(num_outputs=num):
                        generator._generate_tokens = MagicMock(
                            return_value=[[1, i, 2] for i in range(num)]
                        )
                        result = generator.generate_batch("test", num_outputs=num)
                        self.assertEqual(len(result), num)

    def test_generate_batch_with_single_output(self):
        """
        AC: generate_batch with num_outputs=1 MUST work correctly.
        """
        from adapters.generators.midillm_generator import (
            MidiLLMGenerator,
            MidiLLMGeneratorConfig,
        )

        config = MidiLLMGeneratorConfig(backend="mock")

        with patch.object(MidiLLMGenerator, "_load_model"):
            with patch.object(MidiLLMGenerator, "_load_tokenizer"):
                generator = MidiLLMGenerator(config)
                generator._model = MagicMock()
                generator._tokenizer = MagicMock()
                generator._midi_vocab = {"PAD": 0, "BOS": 1, "EOS": 2}
                generator._inv_midi_vocab = {0: "PAD", 1: "BOS", 2: "EOS"}
                generator._generate_tokens = MagicMock(return_value=[[1, 100, 2]])

                result = generator.generate_batch("test prompt", num_outputs=1)

                self.assertEqual(len(result), 1)
                self.assertIsInstance(result[0], list)


class TestMidiLLMGeneratorConfigValidation(unittest.TestCase):
    """Test suite for MidiLLMGeneratorConfig validation."""

    def test_config_rejects_invalid_device(self):
        """
        AC: Config MUST reject invalid device values.
        """
        from adapters.generators.midillm_generator import MidiLLMGeneratorConfig

        with self.assertRaises(ValueError):
            MidiLLMGeneratorConfig(device="invalid_device")

    def test_config_rejects_invalid_backend(self):
        """
        AC: Config MUST reject invalid backend values.
        """
        from adapters.generators.midillm_generator import MidiLLMGeneratorConfig

        with self.assertRaises(ValueError):
            MidiLLMGeneratorConfig(backend="invalid_backend")

    def test_config_accepts_valid_backends(self):
        """
        AC: Config MUST accept valid backend values.
        """
        from adapters.generators.midillm_generator import MidiLLMGeneratorConfig

        for backend in ["vllm", "transformers", "mock"]:
            with self.subTest(backend=backend):
                config = MidiLLMGeneratorConfig(backend=backend)
                self.assertEqual(config.backend, backend)


if __name__ == "__main__":
    unittest.main()
