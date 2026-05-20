"""
Text2Midi Generator Adapter - Implements MidiGenerator using custom Transformer.

This adapter wraps a custom Encoder-Decoder Transformer architecture for
text-to-MIDI generation. It uses T5Tokenizer for text encoding and a custom
pickle-based vocabulary for MIDI token decoding.

Architecture:
    - Implements domain.interfaces.MidiGenerator
    - Uses vendored Transformer model (Encoder-Decoder architecture)
    - Uses T5Tokenizer for text input encoding
    - Uses custom pickle vocabulary for MIDI token decoding
    - Auto-detects CUDA/MPS availability for device placement
    - Wraps all external exceptions in adapters.exceptions.GeneratorError
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import torch
from transformers import T5Tokenizer  # type: ignore

from adapters.exceptions import GeneratorError
from domain.entities import MidiBytes, PromptText, TokenId
from domain.interfaces import MidiGenerator
from models.transformer_model import Transformer


@dataclass
class Text2MidiGeneratorConfig:
    """
    Configuration for Text2MidiGenerator.

    Attributes:
        model_path: Path to the pretrained model weights (.bin file).
        text_tokenizer_path: Path or name for T5 tokenizer (default: "google/flan-t5-base").
        midi_vocab_path: Path to the pickle file containing MIDI vocabulary.
        device: Device placement - "auto" (default), "cuda", "mps", or "cpu".
            Auto-detects CUDA/MPS availability and uses GPU if available.

    Raises:
        ValueError: If device is not one of "auto", "cuda", "mps", or "cpu".

    Example:
        >>> config = Text2MidiGeneratorConfig(
        ...     model_path="/models/text2midi/pytorch_model.bin",
        ...     midi_vocab_path="/models/text2midi/vocab_remi.pkl",
        ... )
        >>> generator = Text2MidiGenerator(config)
    """

    model_path: str = "models/text2midi/pytorch_model.bin"
    text_tokenizer_path: str = "google/flan-t5-base"
    midi_vocab_path: str = "models/text2midi/vocab_remi.pkl"
    device: str = "auto"
    temperature: float = 0.5

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        valid_devices = ("auto", "cuda", "mps", "cpu")
        if self.device not in valid_devices:
            raise ValueError(
                f"device must be one of {valid_devices}, got '{self.device}'"
            )


class Text2MidiGenerator(MidiGenerator):
    """
    Adapter implementing MidiGenerator using custom Transformer architecture.

    This generator uses a custom Encoder-Decoder Transformer where:
    - The encoder is a pretrained T5 model for text understanding
    - The decoder is a custom transformer for MIDI token generation
    - Separate tokenizers are used for text (T5) and MIDI (custom pickle vocab)

    The adapter:
    - Auto-detects CUDA/MPS and places model on GPU if available
    - Generates MIDI tokens auto-regressively using custom inference loop
    - Decodes tokens using the custom MIDI vocabulary
    - Wraps all external exceptions in GeneratorError

    Example:
        >>> config = Text2MidiGeneratorConfig(
        ...     model_path="/models/text2midi/pytorch_model.bin",
        ...     midi_vocab_path="/models/text2midi/vocab_remi.pkl",
        ... )
        >>> generator = Text2MidiGenerator(config)
        >>> new_tokens = generator.generate_step(
        ...     "tempo:80 key:C_major instruments:piano",
        ...     [],
        ...     100,
        ... )
        >>> midi_bytes = generator.decode_to_midi(new_tokens)
    """

    def __init__(self, config: Text2MidiGeneratorConfig) -> None:
        """
        Initialize the generator with configuration.

        Args:
            config: Configuration containing model and vocabulary paths.

        Raises:
            GeneratorError: If model, tokenizer, or vocabulary loading fails.
        """
        self._config = config
        self._device = self._resolve_device()
        self._text_tokenizer = self._load_text_tokenizer()
        self._midi_tokenizer = self._load_midi_tokenizer()
        self._midi_vocab = getattr(self._midi_tokenizer, "vocab", self._midi_tokenizer)
        self._inv_midi_vocab = (
            {v: k for k, v in self._midi_vocab.items()}
            if isinstance(self._midi_vocab, dict)
            else {}
        )
        self._model = self._load_model()

    def _resolve_device(self) -> torch.device:
        """
        Determine the device for model placement.

        Returns:
            torch.device for "cuda", "mps", or "cpu".
        """
        if self._config.device == "auto":
            if torch.cuda.is_available():
                device_name = "cuda"
            elif torch.backends.mps.is_available():
                device_name = "mps"
            else:
                device_name = "cpu"
        else:
            device_name = self._config.device

        return torch.device(device_name)

    def _load_text_tokenizer(self) -> T5Tokenizer:
        """
        Load the T5 tokenizer for text encoding.

        Returns:
            Loaded T5Tokenizer instance.

        Raises:
            GeneratorError: If tokenizer loading fails.
        """
        try:
            return T5Tokenizer.from_pretrained(self._config.text_tokenizer_path)
        except Exception as e:
            raise GeneratorError(
                f"Failed to load text tokenizer from {self._config.text_tokenizer_path}: {e}"
            ) from e

    def _load_midi_tokenizer(self) -> Any:
        """
        Load the MIDI tokenizer (usually a miditok object) from pickle file.

        Returns:
            The loaded MIDI tokenizer object.

        Raises:
            GeneratorError: If vocabulary loading fails.
        """
        try:
            vocab_path = Path(self._config.midi_vocab_path)
            if not vocab_path.exists():
                raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

            with open(vocab_path, "rb") as f:
                tokenizer = pickle.load(f)

            return tokenizer
        except GeneratorError:
            raise
        except Exception as e:
            raise GeneratorError(
                f"Failed to load MIDI tokenizer from {self._config.midi_vocab_path}: {e}"
            ) from e

    def _load_model(self) -> Transformer:
        """
        Load the Transformer model from the configured path.

        Returns:
            Loaded Transformer instance on the correct device.

        Raises:
            GeneratorError: If model loading fails.
        """
        try:
            vocab_size = len(self._midi_vocab)

            # Initialize model with architecture parameters matching the pretrained weights
            # These are the defaults from the original text2midi model
            model = Transformer(
                n_vocab=vocab_size,
                d_model=768,
                nhead=8,
                max_len=2048,
                num_decoder_layers=18,
                dim_feedforward=1024,
                dropout=0.1,
                device=self._device,
            )

            # Load pretrained weights if path is provided
            if self._config.model_path:
                model_path = Path(self._config.model_path)
                if not model_path.exists():
                    raise FileNotFoundError(f"Model file not found: {model_path}")
                state_dict = torch.load(model_path, map_location=self._device)
                model.load_state_dict(state_dict)

            model.eval()
            return model
        except GeneratorError:
            raise
        except Exception as e:
            raise GeneratorError(
                f"Failed to load model from {self._config.model_path}: {e}"
            ) from e

    def generate_step(
        self,
        technical_prompt: PromptText,
        current_tokens: List[TokenId],
        num_tokens: int,
    ) -> List[TokenId]:
        """
        Generate the next batch of MIDI tokens.

        Performs auto-regressive generation using the custom Transformer,
        returning only the newly generated tokens.

        Args:
            technical_prompt: Structured prompt guiding generation.
            current_tokens: Tokens from previous steps (may be empty).
            num_tokens: Number of NEW tokens to generate.

        Returns:
            List of only the newly generated TokenIds.

        Raises:
            GeneratorError: If generation fails.
        """
        try:
            # Encode the prompt using T5 tokenizer
            inputs = self._text_tokenizer(
                technical_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            input_ids = inputs["input_ids"].to(self._device)
            attention_mask = inputs["attention_mask"].to(self._device)

            with torch.no_grad():
                # For continuation, we must prepend the BOS token (1) to the current sequence
                # so the transformer recognizes the start of the sequence context.
                if current_tokens:
                    # Token ID 1 is BOS_None in vocab_remi.pkl
                    context_tokens = [1] + current_tokens
                    tgt_fin = torch.tensor(context_tokens, dtype=torch.long, device=self._device)
                else:
                    tgt_fin = None

                # Delegate the progressive generation loop and sampling math
                # directly to the vendored model implementation
                new_tokens_tensor = self._model.generate(
                    src=input_ids,
                    src_mask=attention_mask,
                    max_len=num_tokens,
                    temperature=self._config.temperature,
                    tgt_fin=tgt_fin,
                )

            # Convert the returned 2D tensor batch to a 1D list of python ints
            return new_tokens_tensor[0].tolist()
        except Exception as e:
            raise GeneratorError(f"Token generation failed: {e}") from e

    def decode_to_midi(self, tokens: List[TokenId]) -> MidiBytes:
        """
        Decode a token sequence into MIDI file bytes.

        Converts the token sequence using the custom MIDI vocabulary.
        For full MIDI file generation, this requires the miditok library
        to be properly configured with the vocabulary.

        Args:
            tokens: Complete token sequence to decode.

        Returns:
            Raw MIDI file bytes, or placeholder bytes if full decoding unavailable.

        Raises:
            GeneratorError: If MIDI decoding fails.
        """
        try:
            if hasattr(self._midi_tokenizer, "decode"):
                import os
                import tempfile

                # Apply Monkey Patch for Miditok v3+ compatibility with v2.x vocab
                if hasattr(self._midi_tokenizer, "config"):
                    if not hasattr(self._midi_tokenizer.config, "additional_tokens"):
                        self._midi_tokenizer.config.additional_tokens = []
                    
                    attrs_to_fix = {
                        "use_velocities": True,
                        "use_note_duration_programs": [],
                        "use_programs": True,
                        "use_chords": False,
                        "use_rests": False,
                        "use_tempos": True,
                        "use_time_signatures": True,
                        "use_sustain_pedals": False,
                        "use_pitch_bends": False,
                        "use_pitch_intervals": False,
                        "program_changes": False,
                        "default_note_duration": 0.5,
                    }
                    for attr, value in attrs_to_fix.items():
                        setattr(self._midi_tokenizer.config, attr, value)

                # Decode tokens to a miditok.Midi (or similar) object
                midi_obj = self._midi_tokenizer.decode(tokens)

                # We need to get the raw bytes. Most MIDI libraries support dumping to a file.
                if hasattr(midi_obj, "dump_midi"):
                    fd, temp_path = tempfile.mkstemp(suffix=".mid")
                    os.close(fd)
                    try:
                        midi_obj.dump_midi(temp_path)
                        with open(temp_path, "rb") as f:
                            midi_bytes = f.read()
                        return midi_bytes
                    finally:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)

                elif hasattr(midi_obj, "dump"):
                    fd, temp_path = tempfile.mkstemp(suffix=".mid")
                    os.close(fd)
                    try:
                        midi_obj.dump(temp_path)
                        with open(temp_path, "rb") as f:
                            midi_bytes = f.read()
                        return midi_bytes
                    finally:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)

            # Fallback placeholder if decode is missing
            events = [self._inv_midi_vocab.get(t, f"UNK_{t}") for t in tokens]
            event_str = " ".join(events)
            return f"MIDI_EVENTS: {event_str}".encode("utf-8")
        except Exception as e:
            raise GeneratorError(f"MIDI decoding failed: {e}") from e
