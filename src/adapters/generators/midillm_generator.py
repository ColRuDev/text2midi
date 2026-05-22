"""
MidiLLM Generator Adapter - Implements BatchMidiGenerator using LLM backends.

This adapter wraps vLLM or HuggingFace Transformers for batch MIDI generation.
It implements the BatchMidiGenerator interface for Best-of-N search use cases.

Architecture:
    - Implements domain.interfaces.BatchMidiGenerator
    - Supports vLLM and Transformers backends (configurable)
    - Uses T5Tokenizer for text encoding
    - Uses custom pickle vocabulary for MIDI token decoding
    - Auto-detects CUDA/MPS availability for device placement
    - Wraps all external exceptions in adapters.exceptions.GeneratorError
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Set

from adapters.exceptions import GeneratorError
from domain.entities import MidiBytes, PromptText, TokenId
from domain.interfaces import BatchMidiGenerator

# Allowed types for safe pickle loading (MIDI tokenizer objects)
# Includes builtin types and numpy types needed for miditok v2/v3 compatibility
_SAFE_PICKLE_TYPES: Set[str] = {
    # miditok tokenizer classes
    "miditok.MIDITokenizer",
    "miditok.tokenizations.REMI.REMI",
    "miditok.tokenizations.TSD.TSD",
    "miditok.tokenizations.MIDIPlus.MIDIPlus",
    # builtin types needed for pickle reconstruction
    "builtins.int",
    "builtins.dict",
    "builtins.list",
    "builtins.tuple",
    "builtins.set",
    "builtins.str",
    "builtins.float",
    "builtins.bool",
    "builtins.NoneType",
    # numpy types needed for miditok v2/v3 vocabularies
    "numpy.ndarray",
    "numpy.core.multiarray._reconstruct",
    "numpy.core.multiarray.scalar",
    "numpy.dtype",
    "numpy.core._dtype_ctypes",
    "miditok.tokenizations.remi.REMI",
    "miditok.classes.TokenizerConfig",
}


class RestrictedUnpickler(pickle.Unpickler):
    """
    Secure unpickler that validates types DURING deserialization.

    Prevents RCE attacks by rejecting any object that is not in the
    allowed types list. The check happens BEFORE the object is instantiated,
    not after loading when malicious code may have already executed.
    """

    def find_class(self, module: str, name: str) -> Any:
        """Validate that the requested class is in the allowed list."""
        full_name = f"{module}.{name}"
        # Check if this is an allowed type using exact match
        if full_name in _SAFE_PICKLE_TYPES:
            return super().find_class(module, name)
        raise GeneratorError(
            f"Unsafe pickle class: {full_name}. "
            f"Expected one of: {_SAFE_PICKLE_TYPES}"
        )


@dataclass
class MidiLLMGeneratorConfig:
    """
    Configuration for MidiLLMGenerator.

    Attributes:
        model_name: Model name or path for the LLM (e.g., "google/flan-t5-base").
        backend: Backend to use - "vllm", "transformers", or "mock" for testing.
        midi_vocab_path: Path to the pickle file containing MIDI vocabulary.
        device: Device placement - "auto" (default), "cuda", "mps", or "cpu".
        max_new_tokens: Maximum tokens to generate per sequence.
        temperature: Sampling temperature for generation.
        num_return_sequences: Number of sequences to return per batch call.

    Raises:
        ValueError: If device is not one of "auto", "cuda", "mps", or "cpu".
    """

    model_name: str = "slseanwu/MIDI-LLM_Llama-3.2-1B"
    backend: str = "transformers"
    midi_vocab_path: str = "models/text2midi/vocab_remi.pkl"
    device: str = "auto"
    max_new_tokens: int = 1024
    temperature: float = 0.8
    num_return_sequences: int = 1

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        valid_devices = ("auto", "cuda", "mps", "cpu")
        if self.device not in valid_devices:
            raise ValueError(
                f"device must be one of {valid_devices}, got '{self.device}'"
            )
        valid_backends = ("vllm", "transformers", "mock")
        if self.backend not in valid_backends:
            raise ValueError(
                f"backend must be one of {valid_backends}, got '{self.backend}'"
            )


class MidiLLMGenerator(BatchMidiGenerator):
    """
    Adapter implementing BatchMidiGenerator using LLM backends.

    This generator:
    - Supports vLLM and Transformers backends
    - Generates multiple complete MIDI sequences in a batch
    - Decodes token sequences using the custom MIDI vocabulary
    - Wraps all external exceptions in GeneratorError

    The adapter is designed for Best-of-N generation where multiple
    complete sequences are generated and the best one is selected.
    """

    def __init__(self, config: MidiLLMGeneratorConfig) -> None:
        """
        Initialize the generator with configuration.

        Args:
            config: Configuration containing model and vocabulary settings.

        Raises:
            GeneratorError: If model, tokenizer, or vocabulary loading fails.
        """
        self._config = config
        self._device = self._resolve_device()
        self._tokenizer = self._load_tokenizer()
        # Initialize _midi_tokenizer before loading vocab to handle early return
        self._midi_tokenizer = None
        self._midi_vocab = self._load_midi_vocab()
        self._inv_midi_vocab = (
            {v: k for k, v in self._midi_vocab.items()}
            if isinstance(self._midi_vocab, dict)
            else {}
        )
        
        # Inject vocabulary mapping into the domain layer for heuristics evaluation
        from domain.remi_vocab import set_inverted_vocab
        set_inverted_vocab(self._inv_midi_vocab)
        
        self._model = self._load_model()

    def _resolve_device(self) -> str:
        """
        Determine the device for model placement.

        Returns:
            Device string: "cuda", "mps", or "cpu".
        """
        # Skip torch import for mock backend
        if self._config.backend == "mock":
            return "cpu"

        import torch

        if self._config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return self._config.device

    def _load_tokenizer(self) -> Any:
        """
        Load the text tokenizer.

        Returns:
            Loaded tokenizer instance.

        Raises:
            GeneratorError: If tokenizer loading fails.
        """
        try:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(
                self._config.model_name,
                pad_token="<|eot_id|>",
            )
        except Exception as e:
            raise GeneratorError(
                f"Failed to load tokenizer from {self._config.model_name}: {e}"
            ) from e

    def _load_midi_vocab(self) -> dict:
        """
        Load the MIDI vocabulary from pickle file.

        Returns:
            Dictionary mapping token names to IDs.

        Raises:
            GeneratorError: If vocabulary loading fails or contains unsafe types.
        """
        try:
            vocab_path = Path(self._config.midi_vocab_path)
            if not vocab_path.exists():
                # Return a minimal vocab if file doesn't exist (for testing)
                return {"PAD": 0, "BOS": 1, "EOS": 2}

            with open(vocab_path, "rb") as f:
                # Use RestrictedUnpickler to validate types DURING deserialization
                # This prevents RCE attacks by checking before object instantiation
                tokenizer = RestrictedUnpickler(f).load()

            # Assign tokenizer for later use in decode_to_midi
            self._midi_tokenizer = tokenizer

            # Apply Miditok v3+ compatibility monkey-patch during initialization
            # to avoid race conditions from patching during decode
            self._apply_miditok_patch()

            # Handle both dict and miditok tokenizer objects
            if hasattr(tokenizer, "vocab"):
                return tokenizer.vocab
            return tokenizer
        except GeneratorError:
            raise
        except Exception as e:
            raise GeneratorError(
                f"Failed to load MIDI vocab from {self._config.midi_vocab_path}: {e}"
            ) from e

    def _apply_miditok_patch(self) -> None:
        """
        Apply Miditok v3+ compatibility monkey-patch for v2.x vocab.

        This patch adds missing attributes that Miditok v3+ expects but
        v2.x vocabularies don't have. Called once during initialization
        to avoid race conditions from patching during decode.
        """
        if self._midi_tokenizer and hasattr(self._midi_tokenizer, "config"):
            config = self._midi_tokenizer.config
            if not hasattr(config, "additional_tokens"):
                config.additional_tokens = []

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
                setattr(config, attr, value)

    def _load_model(self) -> Any:
        """
        Load the model based on the configured backend.

        Returns:
            Loaded model instance.

        Raises:
            GeneratorError: If model loading fails.
        """
        if self._config.backend == "mock":
            return None  # No actual model for mock backend

        try:
            import torch

            if self._config.backend == "vllm":
                try:
                    from vllm import LLM

                    return LLM(model=self._config.model_name)
                except ImportError:
                    raise GeneratorError(
                        "vLLM backend requested but vllm is not installed. "
                        "Install it with: pip install vllm"
                    )
            else:  # transformers
                from transformers import AutoModelForCausalLM

                model = AutoModelForCausalLM.from_pretrained(self._config.model_name)
                model.to(self._device)
                model.eval()
                return model
        except GeneratorError:
            raise
        except Exception as e:
            raise GeneratorError(
                f"Failed to load model from {self._config.model_name}: {e}"
            ) from e

    def generate_batch(
        self,
        technical_prompt: PromptText,
        num_outputs: int,
    ) -> List[List[TokenId]]:
        """
        Generate multiple complete MIDI token sequences in a single batch.

        Args:
            technical_prompt: Structured prompt guiding generation.
            num_outputs: Number of complete sequences to generate.

        Returns:
            List of token sequences, each a complete MIDI representation.

        Raises:
            GeneratorError: If generation fails.
        """
        try:
            if self._config.backend == "mock":
                return self._generate_tokens(technical_prompt, num_outputs)

            return self._generate_with_backend(technical_prompt, num_outputs)
        except GeneratorError:
            raise
        except Exception as e:
            raise GeneratorError(f"Batch generation failed: {e}") from e

    def _generate_tokens(
        self, technical_prompt: PromptText, num_outputs: int
    ) -> List[List[TokenId]]:
        """
        Internal method for token generation with mock backend.

        Args:
            technical_prompt: Prompt for generation.
            num_outputs: Number of sequences.

        Returns:
            List of token lists.
        """
        # Mock implementation for testing
        sequences = []
        for i in range(num_outputs):
            # Generate a simple sequence: BOS, some tokens, EOS
            seq = [1, 10 + i, 20 + i, 30 + i, 2]
            sequences.append(seq)
        return sequences

    def _generate_with_backend(
        self, technical_prompt: PromptText, num_outputs: int
    ) -> List[List[TokenId]]:
        """
        Generate using the configured backend (vLLM or Transformers).

        Args:
            technical_prompt: Prompt for generation.
            num_outputs: Number of sequences.

        Returns:
            List of token lists.
        """
        import torch

        # Add space to prompt to match training format
        full_prompt = technical_prompt + " "

        # Encode the prompt
        inputs = self._tokenizer(
            full_prompt,
            return_tensors="pt",
            padding=False,
        )
        input_ids = inputs["input_ids"].to(self._device)
        
        # Add MIDI BOS token (AMT_GPT2_BOS_ID=1) shifted by LLAMA_VOCAB_SIZE (128256)
        llama_vocab_size = 128256
        midi_bos = torch.tensor([[1 + llama_vocab_size]], device=self._device)
        input_ids = torch.cat([input_ids, midi_bos], dim=1)

        if self._config.backend == "vllm":
            # vLLM generation with SamplingParams
            from vllm import SamplingParams

            sampling_params = SamplingParams(
                max_tokens=self._config.max_new_tokens,
                temperature=self._config.temperature,
                n=num_outputs,
            )
            outputs = self._model.generate(
                prompts=[technical_prompt],
                sampling_params=sampling_params,
            )
            # Extract token IDs from vLLM outputs
            # vLLM returns RequestOutput with outputs list of CompletionOutput
            sequences = []
            max_vocab_id = max(self._inv_midi_vocab.keys()) if self._inv_midi_vocab else 600
            for output in outputs:
                for completion in output.outputs:
                    seq = [max(0, min(t - llama_vocab_size, max_vocab_id)) for t in completion.token_ids]
                    sequences.append(seq)
            return sequences
        else:
            # Transformers generation
            attention_mask = torch.ones_like(input_ids)
            with torch.no_grad():
                outputs = self._model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self._config.max_new_tokens,
                    temperature=self._config.temperature,
                    num_return_sequences=num_outputs,
                    do_sample=num_outputs > 1,
                    pad_token_id=self._tokenizer.pad_token_id,
                )

            # Convert tensor outputs to lists
            # Note: Causal LM outputs include input prompt tokens; slice them off
            sequences = []
            max_vocab_id = max(self._inv_midi_vocab.keys()) if self._inv_midi_vocab else 600
            for i in range(num_outputs):
                seq = outputs[i][input_ids.shape[-1]:].tolist()
                seq = [max(0, min(t - llama_vocab_size, max_vocab_id)) for t in seq]
                sequences.append(seq)

            return sequences

    def decode_to_midi(self, tokens: List[TokenId]) -> MidiBytes:
        """
        Decode a token sequence into MIDI file bytes.

        Args:
            tokens: Complete token sequence to decode.

        Returns:
            Raw MIDI file bytes, or placeholder bytes if full decoding unavailable.

        Raises:
            GeneratorError: If MIDI decoding fails.
        """
        try:
            # Try to use miditok decoder if available
            if self._midi_tokenizer and hasattr(self._midi_tokenizer, "decode"):
                return self._decode_tokens(tokens)

            # Fallback: return token events as string
            events = [self._inv_midi_vocab.get(t, f"UNK_{t}") for t in tokens]
            event_str = " ".join(events)
            return f"MIDI_EVENTS: {event_str}".encode("utf-8")
        except Exception as e:
            raise GeneratorError(f"MIDI decoding failed: {e}") from e

    def _decode_tokens(self, tokens: List[TokenId]) -> MidiBytes:
        """
        Decode tokens using miditok if available.

        Args:
            tokens: Token sequence to decode.

        Returns:
            MIDI bytes.
        """
        import os
        import tempfile

        midi_obj = self._midi_tokenizer.decode(tokens)

        if hasattr(midi_obj, "dump_midi"):
            fd, temp_path = tempfile.mkstemp(suffix=".mid")
            os.close(fd)
            try:
                midi_obj.dump_midi(temp_path)
                with open(temp_path, "rb") as f:
                    return f.read()
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        elif hasattr(midi_obj, "dump"):
            fd, temp_path = tempfile.mkstemp(suffix=".mid")
            os.close(fd)
            try:
                midi_obj.dump(temp_path)
                with open(temp_path, "rb") as f:
                    return f.read()
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        # Fallback
        events = [self._inv_midi_vocab.get(t, f"UNK_{t}") for t in tokens]
        return f"MIDI_EVENTS: {' '.join(events)}".encode("utf-8")
