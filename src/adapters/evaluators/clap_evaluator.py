"""
CLAP Evaluator - Audio-text similarity evaluation using LAION CLAP.

This adapter evaluates the alignment between generated audio and text prompts
using the CLAP (Contrastive Language-Audio Pretraining) model.

Architecture:
    - Implements domain.interfaces.Evaluator
    - Uses LAION CLAP for audio-text similarity scoring
    - Supports both technical prompts and original intent for evaluation
    - Provides fallback scoring when CLAP is unavailable
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from domain.entities import ClapPromptSource

if TYPE_CHECKING:
    from domain.entities import Intent, MidiSequence
    from domain.interfaces import AudioSamples

# Try to import CLAP, handle gracefully if not available
try:
    import laion_clap  # type: ignore
    CLAP_AVAILABLE = True
except ImportError:
    laion_clap = None  # type: ignore
    CLAP_AVAILABLE = False


class ClapEvaluator:
    """
    Evaluator using LAION CLAP for audio-text similarity.
    
    This evaluator computes a relevance score between generated audio
    and a text prompt using the CLAP model's multimodal embeddings.
    
    The adapter:
    - Uses CLAP to embed both audio and text
    - Computes cosine similarity between embeddings
    - Returns a normalized score in [0, 1] range
    - Falls back to 0.5 score when CLAP is unavailable
    
    Attributes:
        clap_prompt_source: Controls which text is used for evaluation.
            Defaults to ClapPromptSource.TECHNICAL.
    
    Example:
        >>> evaluator = ClapEvaluator()
        >>> score = evaluator.evaluate(sequence, audio_bytes, intent)
    """
    
    clap_prompt_source: str = ClapPromptSource.TECHNICAL
    
    def __init__(self, model_path: Optional[str] = None) -> None:
        """
        Initialize the CLAP evaluator.
        
        Args:
            model_path: Optional path to CLAP model weights.
                Uses default model if not provided.
        """
        self._model = None
        self._model_path = model_path
        
        if CLAP_AVAILABLE and laion_clap is not None:
            try:
                # Load CLAP model (using default if no path provided)
                self._model = laion_clap.CLAP_Module(enable_fusion=False)
                self._model.load_ckpt()
            except Exception:
                # Model loading failed, will use fallback
                pass
    
    def set_clap_prompt_source(self, source: str) -> None:
        """
        Set the CLAP evaluation prompt source.
        
        Args:
            source: ClapPromptSource.TECHNICAL or ClapPromptSource.ORIGINAL
        
        Raises:
            ValueError: If source is invalid.
        """
        if not ClapPromptSource.is_valid(source):
            raise ValueError(
                f"Invalid CLAP prompt source: {source}. "
                f"Use ClapPromptSource.TECHNICAL or ClapPromptSource.ORIGINAL"
            )
        self.clap_prompt_source = source
    
    def get_clap_prompt(self, sequence: "MidiSequence", intent: "Intent") -> str:
        """
        Get the text to use for CLAP evaluation.
        
        Args:
            sequence: Contains technical_prompt.
            intent: Contains original user text.
        
        Returns:
            The appropriate prompt text for CLAP evaluation.
        """
        if self.clap_prompt_source == ClapPromptSource.ORIGINAL:
            return intent.text
        return sequence.technical_prompt
    
    def evaluate(
        self,
        sequence: "MidiSequence",
        audio_data: "AudioSamples",
        intent: "Intent",
    ) -> float:
        """
        Evaluate audio-text similarity using CLAP.
        
        Computes the cosine similarity between the audio embedding
        and the text embedding to produce a relevance score.
        
        Args:
            sequence: The MidiSequence containing the technical prompt.
            audio_data: Synthesized audio bytes (float32 PCM).
            intent: The original user intent for comparison.
        
        Returns:
            A float score in [0, 1] range. Higher means better alignment.
            Returns 0.5 if CLAP is unavailable.
        """
        if self._model is None:
            # CLAP not available, return fallback score
            return 0.5
        
        try:
            import numpy as np
            import torch
            import torch.nn.functional as F
            
            # Get the text prompt for evaluation
            text_prompt = self.get_clap_prompt(sequence, intent)
            
            # Convert raw float32 PCM bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.float32)
            
            # If audio is empty or too short, return 0.5
            if len(audio_array) == 0:
                return 0.5
                
            # Need to reshape audio for CLAP: (batch_size, num_samples)
            # CLAP expects 48kHz audio by default
            # Add an extra dimension: (1, num_samples)
            if hasattr(np, 'expand_dims'):
                audio_array = np.expand_dims(audio_array, axis=0)
            else:
                audio_array = np.array([audio_array])
                
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio_array)
            
            # Get audio and text embeddings using the actual laion-clap API
            # use_tensor=True to get PyTorch tensors back for easy cosine similarity
            audio_embedding = self._model.get_audio_embedding_from_data(x=audio_tensor, use_tensor=True)
            text_embedding = self._model.get_text_embedding([text_prompt], use_tensor=True)
            
            # Compute cosine similarity
            similarity = F.cosine_similarity(audio_embedding, text_embedding, dim=1)
            
            # Normalize to [0, 1] range (cosine similarity is in [-1, 1])
            score = (similarity.cpu().item() + 1.0) / 2.0
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            # Evaluation failed, return fallback
            import logging
            logging.getLogger(__name__).warning(f"CLAP evaluation failed: {e}")
            return 0.5
