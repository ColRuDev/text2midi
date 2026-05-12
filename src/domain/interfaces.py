"""
Domain interfaces - Abstract base classes defining system contracts.

This module defines the ports (interfaces) that the domain layer exposes.
Implementations (adapters) will be provided by the infrastructure layer.
All interfaces are framework-agnostic by design.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Protocol

from .entities import (
    ClapPromptSource,
    Intent,
    MidiBytes,
    MidiSequence,
    PromptText,
    TokenId,
)


class LLMTranslator(ABC):
    """
    Abstract interface for translating user intent to technical prompts.

    The LLMTranslator converts natural language intent into structured
    technical prompts that can guide the MIDI token generation process.
    Multiple variations can be generated for beam search exploration.
    """

    @abstractmethod
    def translate(self, intent: Intent, num_variations: int) -> List[PromptText]:
        """
        Translate a user intent into one or more technical prompts.

        Args:
            intent: The user's creative intent in natural language.
            num_variations: Number of prompt variations to generate.
                More variations increase diversity but also computational cost.

        Returns:
            A list of technical prompt strings, one per variation.
            Each prompt should be structured for MIDI token generation.

        Example:
            >>> translator.translate(Intent("A sunrise at the beach"), 3)
            ["tempo:80 key:C_major instruments:piano style:ambient...",
             "tempo:70 key:G_major instruments:synth style:peaceful...",
             "tempo:90 key:F_major instruments:strings style:calm..."]
        """
        ...


class MidiGenerator(ABC):
    """
    Abstract interface for MIDI token generation and decoding.

    The MidiGenerator is responsible for:
    1. Generating MIDI tokens autoregressively given a technical prompt
    2. Decoding token sequences into actual MIDI file bytes
    """

    @abstractmethod
    def generate_step(
        self,
        technical_prompt: PromptText,
        current_tokens: List[TokenId],
        num_tokens: int,
    ) -> List[TokenId]:
        """
        Generate the next batch of MIDI tokens.

        This is an autoregressive step that generates new tokens conditioned
        on the technical prompt and previously generated tokens.

        Args:
            technical_prompt: The structured prompt guiding generation.
            current_tokens: Tokens generated in previous steps (may be empty
                for the first step).
            num_tokens: Number of NEW tokens to generate in this step.

        Returns:
            Only the NEWLY GENERATED tokens (NOT the full sequence).
            The caller is responsible for appending to current_tokens.
            This avoids unnecessary list copying in beam search.

        Example:
            >>> new_tokens = generator.generate_step(
            ...     "tempo:80 key:C_major instruments:piano",
            ...     [60, 64, 67, 72],  # Existing tokens
            ...     16
            ... )
            >>> len(new_tokens)
            16  # Only new tokens
            >>> full_sequence = [60, 64, 67, 72] + new_tokens
        """
        ...

    @abstractmethod
    def decode_to_midi(self, tokens: List[TokenId]) -> MidiBytes:
        """
        Decode a token sequence into MIDI file bytes.

        Converts the abstract token representation into a standard MIDI
        file that can be played, saved, or further processed.

        Args:
            tokens: The complete token sequence to decode.

        Returns:
            Raw MIDI file bytes (Standard MIDI Format, type 0 or 1)
            ready to be written to disk or streamed.

        Example:
            >>> midi_bytes = generator.decode_to_midi([60, 64, 67, 72, ...])
            >>> with open("output.mid", "wb") as f:
            ...     f.write(midi_bytes)
        """
        ...


class AudioRenderer(Protocol):
    """
    Protocol for rendering MIDI to audio samples.

    Used by Evaluator to get audio for CLAP-based evaluation.
    This is a Protocol (structural typing) to allow flexible implementations.
    """

    def render(self, tokens: List[TokenId]) -> AudioSamples:
        """
        Render MIDI tokens to audio samples.

        Args:
            tokens: MIDI token sequence to render.

        Returns:
            Audio samples ready for CLAP evaluation (48kHz mono float32).
        """
        ...


type AudioSamples = bytes  # PCM float32 mono 48kHz - expected by CLAP


class Evaluator(ABC):
    """
    Abstract interface for evaluating generated MIDI sequences.

    The Evaluator scores MIDI sequences based on how well they match
    the intent and musical quality criteria. This score is used
    to guide beam search toward better outputs.

    Implementations may combine multiple reward signals:
    - CLAP-based audio-text similarity
    - Music theory consistency (key, harmony)
    - Note distribution plausibility

    The `clap_prompt_source` attribute controls which text is used for
    CLAP evaluation. Change this with a single line to switch between
    technical prompt (more precise) and original intent (more natural).
    """

    clap_prompt_source: str = ClapPromptSource.TECHNICAL

    def set_clap_prompt_source(self, source: str) -> None:
        """
        Set the CLAP evaluation prompt source.

        Args:
            source: ClapPromptSource.TECHNICAL or ClapPromptSource.ORIGINAL

        Raises:
            ValueError: If source is invalid.

        Example:
            >>> evaluator.set_clap_prompt_source(ClapPromptSource.ORIGINAL)
        """
        if not ClapPromptSource.is_valid(source):
            raise ValueError(
                f"Invalid CLAP prompt source: {source}. "
                f"Use ClapPromptSource.TECHNICAL or ClapPromptSource.ORIGINAL"
            )
        self.clap_prompt_source = source

    def get_clap_prompt(self, sequence: MidiSequence, intent: Intent) -> PromptText:
        """
        Get the text to use for CLAP evaluation based on current configuration.

        Args:
            sequence: Contains technical_prompt.
            intent: Contains original user text.

        Returns:
            The appropriate prompt text for CLAP evaluation.
        """
        if self.clap_prompt_source == ClapPromptSource.ORIGINAL:
            return intent.text
        return sequence.technical_prompt

    @abstractmethod
    def evaluate(
        self, sequence: MidiSequence, audio_data: AudioSamples, intent: Intent
    ) -> float:
        """
        Evaluate a MIDI sequence against quality criteria.

        Computes a reward score that reflects how well the generated
        sequence matches the intent and adheres to musical conventions.

        Args:
            sequence: The MidiSequence to evaluate, containing the technical
                prompt and generated tokens.
            audio_data: Synthesized audio from the MIDI, in CLAP-compatible
                format (48kHz mono float32 PCM).
                Use AudioRenderer protocol to convert tokens to audio.
            intent: The original user intent. Used for CLAP evaluation
                if clap_prompt_source is ORIGINAL.

        Returns:
            A float reward score in range [0, 1]. Higher is better.

        Example:
            >>> sequence = MidiSequence(
            ...     technical_prompt="tempo:80 key:C_major...",
            ...     tokens=[60, 64, 67, ...],
            ... )
            >>> audio = renderer.render(sequence.tokens)  # 48kHz mono float32
            >>> score = evaluator.evaluate(sequence, audio, intent)
            >>> sequence.reward = score
        """
        ...
