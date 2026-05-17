"""
Unit tests for InMemoryFluidSynthEngine.

Tests for in-memory MIDI to audio synthesis using pretty_midi.
"""

import io
from unittest.mock import MagicMock, Mock, patch

import pytest

from adapters.audio.fluidsynth_memory import InMemoryFluidSynthEngine
from adapters.exceptions import ConfigurationError


class TestInMemoryFluidSynthEngineInit:
    """Tests for InMemoryFluidSynthEngine initialization."""

    def test_init_with_default_sample_rate(self):
        """Engine initializes with default 48000 Hz sample rate."""
        engine = InMemoryFluidSynthEngine()
        
        assert engine.sample_rate == 48000

    def test_init_with_custom_sample_rate(self):
        """Engine accepts custom sample rate."""
        engine = InMemoryFluidSynthEngine(sample_rate=44100)
        
        assert engine.sample_rate == 44100

    @patch("adapters.audio.fluidsynth_memory.pretty_midi")
    def test_init_succeeds_when_pretty_midi_available(self, mock_pretty_midi):
        """Engine initializes successfully when pretty_midi is available."""
        engine = InMemoryFluidSynthEngine()
        
        assert engine is not None


class TestInMemoryFluidSynthEngineRender:
    """Tests for the render method."""

    @pytest.fixture
    def mock_pretty_midi(self):
        """Mock pretty_midi module."""
        with patch("adapters.audio.fluidsynth_memory.pretty_midi") as mock:
            yield mock

    @pytest.fixture
    def engine(self, mock_pretty_midi):
        """Create engine with mocked pretty_midi."""
        return InMemoryFluidSynthEngine()

    def test_render_returns_bytes(self, engine, mock_pretty_midi):
        """render() returns AudioSamples (bytes)."""
        # Setup mock
        mock_pm = MagicMock()
        mock_midi = MagicMock()
        mock_pretty_midi.PrettyMIDI.return_value = mock_midi
        mock_midi.synthesize.return_value = MagicMock(
            tobytes=lambda: b"\x00\x00\x80?"  # 1.0 in float32 LE
        )
        
        result = engine.render([60, 64, 67])
        
        assert isinstance(result, bytes)

    def test_render_synthesizes_at_48khz(self, engine, mock_pretty_midi):
        """render() synthesizes at the configured sample rate."""
        mock_pm = MagicMock()
        mock_midi = MagicMock()
        mock_pretty_midi.PrettyMIDI.return_value = mock_midi
        mock_midi.synthesize.return_value = MagicMock(tobytes=lambda: b"")
        
        engine.render([60, 64, 67])
        
        mock_midi.synthesize.assert_called_once_with(fs=48000)

    def test_render_creates_midi_in_memory(self, engine, mock_pretty_midi):
        """render() creates MIDI in-memory without disk writes."""
        mock_midi = MagicMock()
        mock_pretty_midi.PrettyMIDI.return_value = mock_midi
        mock_pretty_midi.Instrument.return_value = mock_midi
        mock_midi.synthesize.return_value = MagicMock(tobytes=lambda: b"audio")
        
        # render() creates a new MIDI object (no file I/O)
        engine.render([60, 64, 67])
        
        # Verify PrettyMIDI was called (creates new, not loading from file)
        mock_pretty_midi.PrettyMIDI.assert_called_once()
        
        # Verify Instrument was created (notes added in memory)
        mock_pretty_midi.Instrument.assert_called_once()

    def test_render_converts_tokens_to_midi_bytes(self, engine, mock_pretty_midi):
        """render() properly converts token list to MIDI representation."""
        mock_midi = MagicMock()
        mock_pretty_midi.PrettyMIDI.return_value = mock_midi
        mock_midi.synthesize.return_value = MagicMock(tobytes=lambda: b"test_audio")
        
        tokens = [60, 64, 67, 72]
        result = engine.render(tokens)
        
        assert result == b"test_audio"

    def test_render_wraps_exceptions_in_generator_error(self, engine, mock_pretty_midi):
        """render() wraps exceptions in ConfigurationError."""
        mock_pretty_midi.PrettyMIDI.side_effect = Exception("Synthesis failed")
        
        with pytest.raises(ConfigurationError) as exc_info:
            engine.render([60, 64, 67])
        
        assert "Audio synthesis failed" in str(exc_info.value)


class TestInMemoryFluidSynthEngineFromMidiBytes:
    """Tests for synthesizing from raw MIDI bytes."""

    @pytest.fixture
    def mock_pretty_midi(self):
        """Mock pretty_midi module."""
        with patch("adapters.audio.fluidsynth_memory.pretty_midi") as mock:
            yield mock

    def test_synthesize_from_bytes_returns_audio(self, mock_pretty_midi):
        """synthesize_from_bytes() converts MIDI bytes to audio bytes."""
        mock_midi = MagicMock()
        mock_pretty_midi.PrettyMIDI.return_value = mock_midi
        mock_midi.synthesize.return_value = MagicMock(
            tobytes=lambda: b"\x00\x00\x80?\x00\x00\x00@"
        )
        
        engine = InMemoryFluidSynthEngine()
        midi_bytes = b"MThd..."  # Simulated MIDI file bytes
        result = engine.synthesize_from_bytes(midi_bytes)
        
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_synthesize_from_bytes_uses_bytesio(self, mock_pretty_midi):
        """synthesize_from_bytes() loads from BytesIO (no disk)."""
        mock_midi = MagicMock()
        captured_args = []
        
        def capture_arg(arg):
            captured_args.append(arg)
            return mock_midi
        
        mock_pretty_midi.PrettyMIDI.side_effect = capture_arg
        mock_midi.synthesize.return_value = MagicMock(tobytes=lambda: b"")
        
        engine = InMemoryFluidSynthEngine()
        engine.synthesize_from_bytes(b"MThd...")
        
        assert isinstance(captured_args[0], io.BytesIO)


class TestInMemoryFluidSynthEngineIntegration:
    """Integration-style tests with mocked synthesis."""

    @pytest.fixture
    def mock_pretty_midi(self):
        """Mock pretty_midi module."""
        with patch("adapters.audio.fluidsynth_memory.pretty_midi") as mock:
            yield mock

    def test_full_synthesis_workflow(self, mock_pretty_midi):
        """Test complete workflow from tokens to audio bytes."""
        # Setup mock to simulate synthesis
        mock_midi = MagicMock()
        mock_pretty_midi.PrettyMIDI.return_value = mock_midi
        
        # Simulate float32 audio data (1 second of silence at 48kHz mono)
        import struct
        samples = [0.0] * 48000
        audio_bytes = b"".join(struct.pack("<f", s) for s in samples)
        mock_midi.synthesize.return_value = MagicMock(tobytes=lambda: audio_bytes)
        
        engine = InMemoryFluidSynthEngine()
        result = engine.render([60, 64, 67, 72, 76])
        
        # Result should be the audio bytes
        assert result == audio_bytes
        assert len(result) == 48000 * 4  # 48000 float32 samples = 192000 bytes
