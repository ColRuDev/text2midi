"""
In-Memory FluidSynth Engine - Synthesizes MIDI to audio without disk I/O.

This adapter renders MIDI tokens to audio using pretty_midi's synthesize method,
keeping all data in memory via io.BytesIO. It implements the AudioRenderer protocol.

Architecture:
    - Implements domain.interfaces.AudioRenderer
    - Uses pretty_midi for MIDI parsing and synthesis
    - Uses io.BytesIO for in-memory MIDI buffer
    - Wraps all exceptions in adapters.exceptions.ConfigurationError
"""

from __future__ import annotations

import io
from typing import List

import pretty_midi

from adapters.exceptions import ConfigurationError
from domain.entities import TokenId


class InMemoryFluidSynthEngine:
    """
    In-memory audio synthesis engine using pretty_midi.
    
    This engine synthesizes MIDI data to audio samples entirely in memory,
    avoiding disk I/O by using io.BytesIO for intermediate storage.
    
    The adapter:
    - Converts MIDI tokens/bytes to audio using pretty_midi.synthesize()
    - Outputs raw PCM float32 mono audio at the configured sample rate
    - Wraps all synthesis errors in ConfigurationError
    
    Attributes:
        sample_rate: Audio sample rate in Hz (default: 48000 for CLAP compatibility)
    
    Example:
        >>> engine = InMemoryFluidSynthEngine()
        >>> audio_bytes = engine.render([60, 64, 67])  # Returns float32 PCM bytes
    """
    
    def __init__(self, sample_rate: int = 48000) -> None:
        """
        Initialize the synthesis engine.
        
        Args:
            sample_rate: Target sample rate in Hz. Default 48000 for CLAP.
        """
        self.sample_rate = sample_rate
    
    def render(self, tokens: List[TokenId]) -> bytes:
        """
        Render MIDI tokens to audio samples.
        
        This method converts a sequence of MIDI tokens to audio bytes
        using in-memory synthesis.
        
        Note: This is a simplified implementation. For full token-to-MIDI
        conversion, you'll need a tokenizer instance (e.g., from miditok).
        Currently returns empty audio for raw token lists.
        
        Args:
            tokens: MIDI token sequence to render.
        
        Returns:
            Audio samples as bytes (PCM float32 mono at sample_rate Hz).
        
        Raises:
            ConfigurationError: If synthesis fails.
        """
        try:
            # Create a simple PrettyMIDI object for testing
            # In production, tokens would be decoded via miditok first
            midi = pretty_midi.PrettyMIDI()
            
            # Create an instrument program (piano)
            instrument = pretty_midi.Instrument(program=0)
            
            # Convert tokens to notes (simplified - assumes tokens are note values)
            # In real implementation, this would use the tokenizer's decode method
            for i, token in enumerate(tokens):
                if 0 <= token <= 127:  # Valid MIDI note range
                    note = pretty_midi.Note(
                        velocity=100,
                        pitch=token,
                        start=i * 0.5,  # 0.5 second per note
                        end=(i + 1) * 0.5,
                    )
                    instrument.notes.append(note)
            
            midi.instruments.append(instrument)
            
            # Synthesize to audio
            audio = midi.synthesize(fs=self.sample_rate)
            
            # Convert to bytes (float32 PCM)
            return audio.tobytes()
            
        except Exception as e:
            raise ConfigurationError(f"Audio synthesis failed: {e}") from e
    
    def synthesize_from_bytes(self, midi_bytes: bytes) -> bytes:
        """
        Synthesize raw MIDI file bytes to audio.
        
        This is the primary method for converting existing MIDI data
        to audio without writing to disk.
        
        Args:
            midi_bytes: Raw MIDI file bytes (Standard MIDI Format).
        
        Returns:
            Audio samples as bytes (PCM float32 mono at sample_rate Hz).
        
        Raises:
            ConfigurationError: If synthesis fails.
        """
        try:
            # Load MIDI from memory buffer
            buffer = io.BytesIO(midi_bytes)
            midi = pretty_midi.PrettyMIDI(buffer)
            
            # Synthesize to audio
            audio = midi.synthesize(fs=self.sample_rate)
            
            # Convert to bytes (float32 PCM)
            return audio.tobytes()
            
        except Exception as e:
            raise ConfigurationError(f"Audio synthesis failed: {e}") from e
