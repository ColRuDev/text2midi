"""
Unit tests for Text2MidiGenerator adapter.

Tests cover:
- Configuration dataclass validation
- Model/tokenizer/vocabulary loading
- Device selection (CPU/CUDA/MPS auto-detection)
- Token generation with custom Transformer
- MIDI decoding with custom vocabulary
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, mock_open

import pytest

from adapters.exceptions import GeneratorError
from adapters.generators.text2midi_generator import (
    Text2MidiGenerator,
    Text2MidiGeneratorConfig,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_torch(mocker: pytest.MockFixture) -> pytest.Mock:
    """Fixture for mocking torch module with configurable CUDA/MPS availability."""
    mock = mocker.patch("adapters.generators.text2midi_generator.torch")
    mock.cuda.is_available.return_value = False
    mock.backends.mps.is_available.return_value = False
    mock.device.return_value = mocker.MagicMock()
    return mock


@pytest.fixture
def mock_t5_tokenizer(mocker: pytest.MockFixture) -> pytest.Mock:
    """Fixture for mocking T5Tokenizer class."""
    return mocker.patch("adapters.generators.text2midi_generator.T5Tokenizer")


@pytest.fixture
def mock_transformer(mocker: pytest.MockFixture) -> pytest.Mock:
    """Fixture for mocking Transformer class."""
    return mocker.patch("adapters.generators.text2midi_generator.Transformer")


@pytest.fixture
def mock_text_tokenizer(mocker: pytest.MockFixture) -> pytest.Mock:
    """Fixture for a mock text tokenizer instance."""
    mock = mocker.MagicMock()
    mock.return_value = {"input_ids": mocker.MagicMock(), "attention_mask": mocker.MagicMock()}
    mock.return_value["input_ids"].to.return_value = mocker.MagicMock()
    mock.return_value["attention_mask"].to.return_value = mocker.MagicMock()
    return mock


@pytest.fixture
def mock_model(mocker: pytest.MockFixture) -> pytest.Mock:
    """Fixture for a mock model instance."""
    mock = mocker.MagicMock()
    mock.eval.return_value = None
    mock.parameters.return_value = iter([])
    mock.to.return_value = mock
    return mock


@pytest.fixture
def mock_model_with_logits(mocker: pytest.MockFixture) -> pytest.Mock:
    """Fixture for a mock model with logits for generate_step tests."""
    mock = mocker.MagicMock()
    mock.eval.return_value = None
    mock.parameters.return_value = iter([])
    mock.to.return_value = mock

    # Create mock logits that return different values on each call
    call_count = [0]

    def mock_forward(*args, **kwargs):
        logits_mock = mocker.MagicMock()
        # Simulate argmax returning different tokens
        logits_mock.argmax.return_value.item = lambda: 100 + call_count[0]
        call_count[0] += 1
        return logits_mock

    # Setup the model to return logits when called
    mock.return_value = mocker.MagicMock()
    mock.return_value.__getitem__ = lambda self, key: mock_forward()
    mock.return_value[:, -1, :].argmax.return_value.item = lambda: 42

    return mock


@pytest.fixture
def sample_midi_vocab() -> Dict[str, int]:
    """Sample MIDI vocabulary for testing."""
    return {
        "Bar_None": 0,
        "Position_0": 1,
        "Pitch_60": 2,
        "Velocity_100": 3,
        "Duration_1.0": 4,
        "EOS": 5,
    }


@pytest.fixture
def mock_vocab_file(sample_midi_vocab: Dict[str, int], tmp_path: Path) -> Path:
    """Create a temporary pickle file with sample vocabulary."""
    vocab_file = tmp_path / "vocab.pkl"
    with open(vocab_file, "wb") as f:
        pickle.dump(sample_midi_vocab, f)
    return vocab_file


# =============================================================================
# TEST CLASSES
# =============================================================================


class TestText2MidiGeneratorConfig:
    """Tests for Text2MidiGeneratorConfig dataclass."""

    def test_default_values(self) -> None:
        """Config should have sensible defaults."""
        config = Text2MidiGeneratorConfig()

        assert config.model_path == ""
        assert config.text_tokenizer_path == "google/flan-t5-base"
        assert config.midi_vocab_path == "models/text2midi/vocab_remi.pkl"
        assert config.device == "auto"

    def test_custom_values(self) -> None:
        """Config should accept custom values."""
        config = Text2MidiGeneratorConfig(
            model_path="/path/to/model.bin",
            text_tokenizer_path="custom/t5",
            midi_vocab_path="/path/to/vocab.pkl",
            device="cuda",
        )

        assert config.model_path == "/path/to/model.bin"
        assert config.text_tokenizer_path == "custom/t5"
        assert config.midi_vocab_path == "/path/to/vocab.pkl"
        assert config.device == "cuda"

    @pytest.mark.parametrize("device", ["cpu", "cuda", "mps", "auto"])
    def test_valid_devices_accepted(self, device: str) -> None:
        """Config should accept all valid device values."""
        config = Text2MidiGeneratorConfig(device=device)
        assert config.device == device

    def test_invalid_device_raises_error(self) -> None:
        """Config should reject invalid device values."""
        with pytest.raises(ValueError, match="device must be"):
            Text2MidiGeneratorConfig(device="invalid")

    def test_empty_paths_are_accepted(self) -> None:
        """Config should accept empty paths (validation happens at load time)."""
        config = Text2MidiGeneratorConfig(model_path="", midi_vocab_path="")
        assert config.model_path == ""
        assert config.midi_vocab_path == ""


class TestText2MidiGeneratorInit:
    """Tests for Text2MidiGenerator initialization."""

    def test_init_loads_all_components(
        self,
        mock_torch: pytest.Mock,
        mock_t5_tokenizer: pytest.Mock,
        mock_transformer: pytest.Mock,
        mock_vocab_file: Path,
        mock_text_tokenizer: pytest.Mock,
        mock_model: pytest.Mock,
    ) -> None:
        """Initialization should load text tokenizer, MIDI vocab, and model."""
        mock_t5_tokenizer.from_pretrained.return_value = mock_text_tokenizer
        mock_transformer.return_value = mock_model

        config = Text2MidiGeneratorConfig(
            model_path="",
            midi_vocab_path=str(mock_vocab_file),
        )
        generator = Text2MidiGenerator(config)

        mock_t5_tokenizer.from_pretrained.assert_called_once_with(
            "google/flan-t5-base"
        )
        mock_transformer.assert_called_once()
        assert generator._text_tokenizer is mock_text_tokenizer
        assert generator._model is mock_model

    @pytest.mark.parametrize(
        "cuda_available,mps_available,expected_device",
        [
            (False, False, "cpu"),
            (True, False, "cuda"),
            (False, True, "mps"),
            (True, True, "cuda"),  # CUDA takes precedence over MPS
        ],
    )
    def test_init_auto_device_selection(
        self,
        mock_torch: pytest.Mock,
        mock_t5_tokenizer: pytest.Mock,
        mock_transformer: pytest.Mock,
        mock_vocab_file: Path,
        mock_text_tokenizer: pytest.Mock,
        mock_model: pytest.Mock,
        cuda_available: bool,
        mps_available: bool,
        expected_device: str,
    ) -> None:
        """Model should be placed on correct device based on CUDA/MPS availability."""
        mock_torch.cuda.is_available.return_value = cuda_available
        mock_torch.backends.mps.is_available.return_value = mps_available
        mock_t5_tokenizer.from_pretrained.return_value = mock_text_tokenizer
        mock_transformer.return_value = mock_model

        config = Text2MidiGeneratorConfig(
            model_path="",
            midi_vocab_path=str(mock_vocab_file),
        )
        Text2MidiGenerator(config)

        mock_torch.device.assert_called_once_with(expected_device)

    @pytest.mark.parametrize(
        "device,cuda_available,mps_available,expected_device",
        [
            ("cpu", True, True, "cpu"),   # Explicit CPU overrides everything
            ("cuda", False, False, "cuda"),  # Explicit CUDA overrides auto-detect
            ("mps", False, False, "mps"),  # Explicit MPS overrides auto-detect
        ],
    )
    def test_init_explicit_device_overrides_auto(
        self,
        mock_torch: pytest.Mock,
        mock_t5_tokenizer: pytest.Mock,
        mock_transformer: pytest.Mock,
        mock_vocab_file: Path,
        mock_text_tokenizer: pytest.Mock,
        mock_model: pytest.Mock,
        device: str,
        cuda_available: bool,
        mps_available: bool,
        expected_device: str,
    ) -> None:
        """Explicit device setting should override auto-detection."""
        mock_torch.cuda.is_available.return_value = cuda_available
        mock_torch.backends.mps.is_available.return_value = mps_available
        mock_t5_tokenizer.from_pretrained.return_value = mock_text_tokenizer
        mock_transformer.return_value = mock_model

        config = Text2MidiGeneratorConfig(
            model_path="",
            midi_vocab_path=str(mock_vocab_file),
            device=device,
        )
        Text2MidiGenerator(config)

        mock_torch.device.assert_called_once_with(expected_device)

    def test_init_raises_error_on_missing_vocab(
        self,
        mock_torch: pytest.Mock,
        mock_t5_tokenizer: pytest.Mock,
        mock_text_tokenizer: pytest.Mock,
    ) -> None:
        """Init should raise GeneratorError if vocab file is missing."""
        config = Text2MidiGeneratorConfig(
            model_path="",
            midi_vocab_path="/nonexistent/vocab.pkl",
        )

        with pytest.raises(GeneratorError, match="Failed to load MIDI tokenizer"):
            Text2MidiGenerator(config)

    def test_init_raises_error_on_tokenizer_failure(
        self,
        mock_torch: pytest.Mock,
        mock_t5_tokenizer: pytest.Mock,
        mock_vocab_file: Path,
    ) -> None:
        """Init should raise GeneratorError if tokenizer loading fails."""
        mock_t5_tokenizer.from_pretrained.side_effect = OSError("Not found")

        config = Text2MidiGeneratorConfig(
            model_path="",
            midi_vocab_path=str(mock_vocab_file),
        )

        with pytest.raises(GeneratorError, match="Failed to load text tokenizer"):
            Text2MidiGenerator(config)


class TestGenerateStep:
    """Tests for generate_step method."""

    def test_generate_step_returns_new_tokens(
        self,
        mock_torch: pytest.Mock,
        mock_t5_tokenizer: pytest.Mock,
        mock_transformer: pytest.Mock,
        mock_vocab_file: Path,
        mock_text_tokenizer: pytest.Mock,
        mock_model: pytest.Mock,
        mocker: pytest.MockFixture,
    ) -> None:
        """generate_step should return newly generated tokens."""
        mock_t5_tokenizer.from_pretrained.return_value = mock_text_tokenizer
        mock_transformer.return_value = mock_model

        # Setup the model to return generated tokens
        mock_model.generate.return_value[0].tolist.return_value = [42, 42, 42]

        config = Text2MidiGeneratorConfig(
            model_path="",
            midi_vocab_path=str(mock_vocab_file),
        )
        generator = Text2MidiGenerator(config)

        result = generator.generate_step(
            technical_prompt="tempo:80 key:C_major",
            current_tokens=[],
            num_tokens=3,
        )

        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(t, int) for t in result)

    def test_generate_step_with_existing_tokens(
        self,
        mock_torch: pytest.Mock,
        mock_t5_tokenizer: pytest.Mock,
        mock_transformer: pytest.Mock,
        mock_vocab_file: Path,
        mock_text_tokenizer: pytest.Mock,
        mock_model: pytest.Mock,
        mocker: pytest.MockFixture,
    ) -> None:
        """generate_step should handle existing tokens correctly."""
        mock_t5_tokenizer.from_pretrained.return_value = mock_text_tokenizer
        mock_transformer.return_value = mock_model

        # Setup the model to return generated tokens
        mock_model.generate.return_value[0].tolist.return_value = [100, 100]

        config = Text2MidiGeneratorConfig(
            model_path="",
            midi_vocab_path=str(mock_vocab_file),
        )
        generator = Text2MidiGenerator(config)

        result = generator.generate_step(
            technical_prompt="tempo:80 key:C_major",
            current_tokens=[10, 20, 30],
            num_tokens=2,
        )

        assert isinstance(result, list)
        assert len(result) == 2

    def test_generate_step_raises_error_on_failure(
        self,
        mock_torch: pytest.Mock,
        mock_t5_tokenizer: pytest.Mock,
        mock_transformer: pytest.Mock,
        mock_vocab_file: Path,
        mock_text_tokenizer: pytest.Mock,
        mock_model: pytest.Mock,
    ) -> None:
        """generate_step should raise GeneratorError on failure."""
        mock_t5_tokenizer.from_pretrained.return_value = mock_text_tokenizer
        mock_transformer.return_value = mock_model
        mock_model.generate.side_effect = RuntimeError("Forward pass failed")

        config = Text2MidiGeneratorConfig(
            model_path="",
            midi_vocab_path=str(mock_vocab_file),
        )
        generator = Text2MidiGenerator(config)

        with pytest.raises(GeneratorError, match="Token generation failed"):
            generator.generate_step(
                technical_prompt="tempo:80 key:C_major",
                current_tokens=[],
                num_tokens=2,
            )


class TestDecodeToMidi:
    """Tests for decode_to_midi method."""

    def test_decode_to_midi_returns_bytes(
        self,
        mock_torch: pytest.Mock,
        mock_t5_tokenizer: pytest.Mock,
        mock_transformer: pytest.Mock,
        mock_vocab_file: Path,
        mock_text_tokenizer: pytest.Mock,
        mock_model: pytest.Mock,
    ) -> None:
        """decode_to_midi should return bytes."""
        mock_t5_tokenizer.from_pretrained.return_value = mock_text_tokenizer
        mock_transformer.return_value = mock_model

        config = Text2MidiGeneratorConfig(
            model_path="",
            midi_vocab_path=str(mock_vocab_file),
        )
        generator = Text2MidiGenerator(config)

        result = generator.decode_to_midi([0, 1, 2, 3])

        assert isinstance(result, bytes)

    def test_decode_to_midi_handles_unknown_tokens(
        self,
        mock_torch: pytest.Mock,
        mock_t5_tokenizer: pytest.Mock,
        mock_transformer: pytest.Mock,
        mock_vocab_file: Path,
        mock_text_tokenizer: pytest.Mock,
        mock_model: pytest.Mock,
    ) -> None:
        """decode_to_midi should handle tokens not in vocabulary."""
        mock_t5_tokenizer.from_pretrained.return_value = mock_text_tokenizer
        mock_transformer.return_value = mock_model

        config = Text2MidiGeneratorConfig(
            model_path="",
            midi_vocab_path=str(mock_vocab_file),
        )
        generator = Text2MidiGenerator(config)

        result = generator.decode_to_midi([0, 1, 999])  # 999 not in vocab

        assert isinstance(result, bytes)
        assert b"UNK_999" in result


class TestText2MidiGeneratorWorkflow:
    """End-to-end workflow tests with mocked model."""

    def test_full_generation_workflow(
        self,
        mock_torch: pytest.Mock,
        mock_t5_tokenizer: pytest.Mock,
        mock_transformer: pytest.Mock,
        mock_vocab_file: Path,
        mock_text_tokenizer: pytest.Mock,
        mock_model: pytest.Mock,
        mocker: pytest.MockFixture,
    ) -> None:
        """Full workflow: load -> generate_step -> decode_to_midi."""
        mock_t5_tokenizer.from_pretrained.return_value = mock_text_tokenizer
        mock_transformer.return_value = mock_model

        # Setup model to return specific tokens
        mock_model.generate.return_value[0].tolist.return_value = [50, 51, 52]

        config = Text2MidiGeneratorConfig(
            model_path="",
            midi_vocab_path=str(mock_vocab_file),
        )
        generator = Text2MidiGenerator(config)

        new_tokens = generator.generate_step(
            technical_prompt="tempo:80 key:C_major instruments:piano",
            current_tokens=[],
            num_tokens=3,
        )

        midi_bytes = generator.decode_to_midi(new_tokens)

        assert isinstance(new_tokens, list)
        assert len(new_tokens) == 3
        assert isinstance(midi_bytes, bytes)
