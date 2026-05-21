"""
Generator adapters for MIDI token generation.

This package contains adapters that implement the MidiGenerator and
BatchMidiGenerator interfaces, providing concrete implementations
for different model backends.
"""

from adapters.generators.midillm_generator import (
    MidiLLMGenerator,
    MidiLLMGeneratorConfig,
)
from adapters.generators.text2midi_generator import (
    Text2MidiGenerator,
    Text2MidiGeneratorConfig,
)

__all__ = [
    "MidiLLMGenerator",
    "MidiLLMGeneratorConfig",
    "Text2MidiGenerator",
    "Text2MidiGeneratorConfig",
]
