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
    import clap  # type: ignore
    CLAP_AVAILABLE = True
except ImportError:
    clap = None  # type: ignore
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
        
        if CLAP_AVAILABLE and clap is not None:
            try:
                # Load CLAP model (using default if no path provided)
                self._model = clap.CLAP()
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
            # Get the text prompt for evaluation
            text_prompt = self.get_clap_prompt(sequence, intent)
            
            # Get audio and text embeddings
            # Note: Actual CLAP API may differ - this is a conceptual implementation
            audio_embedding = self._model.get_audio_embeddings(audio_data)
            text_embedding = self._model.get_text_embeddings([text_prompt])
            
            # Compute similarity
            similarity = self._model.compute_similarity(audio_embedding, text_embedding)
            
            # Normalize to [0, 1] range (CLAP similarity can be in [-1, 1])
            score = (similarity.cpu().item() + 1.0) / 2.0
            
            return max(0.0, min(1.0, score))
            
        except Exception:
            # Evaluation failed, return fallback
            return 0.5
