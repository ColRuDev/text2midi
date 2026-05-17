"""
Composite Evaluator - Orchestrates multiple evaluators with weighted scoring.

This adapter combines ClapEvaluator and HeuristicsEvaluator using weights
from GenerationProfile to produce a unified evaluation score.

Architecture:
    - Implements domain.interfaces.Evaluator
    - Uses GenerationProfile weights for score combination
    - Delegates to ClapEvaluator and HeuristicsEvaluator
    - Provides fallback behavior when evaluators are unavailable
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Protocol

from domain.entities import ClapPromptSource, GenerationProfile

if TYPE_CHECKING:
    from domain.entities import Intent, MidiSequence
    from domain.interfaces import AudioSamples, Evaluator


class CompositeEvaluator:
    """
    Evaluator that combines CLAP and heuristics scores.
    
    This evaluator orchestrates multiple evaluation strategies,
    blending their scores according to GenerationProfile weights.
    
    The adapter:
    - Delegates audio-text evaluation to ClapEvaluator
    - Delegates music theory evaluation to HeuristicsEvaluator
    - Combines scores using weights from GenerationProfile
    - Handles missing evaluators gracefully
    
    Attributes:
        clap_prompt_source: Propagated to ClapEvaluator.
    
    Example:
        >>> profile = GenerationProfile(clap_weight=0.6, key_weight=0.2, note_weight=0.2)
        >>> evaluator = CompositeEvaluator(clap, heuristics, profile)
        >>> score = evaluator.evaluate(sequence, audio, intent)
    """
    
    clap_prompt_source: str = ClapPromptSource.TECHNICAL
    
    def __init__(
        self,
        clap_evaluator: Optional["Evaluator"] = None,
        heuristics_evaluator: Optional["Evaluator"] = None,
        profile: Optional[GenerationProfile] = None,
    ) -> None:
        """
        Initialize the composite evaluator.
        
        Args:
            clap_evaluator: Evaluator for audio-text similarity.
            heuristics_evaluator: Evaluator for music theory analysis.
            profile: GenerationProfile containing weights.
        """
        self._clap_evaluator = clap_evaluator
        self._heuristics_evaluator = heuristics_evaluator
        self._profile = profile or GenerationProfile()
    
    def set_clap_prompt_source(self, source: str) -> None:
        """
        Set the CLAP evaluation prompt source.
        
        Propagates to the ClapEvaluator if available.
        
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
        
        # Propagate to CLAP evaluator
        if self._clap_evaluator is not None:
            self._clap_evaluator.set_clap_prompt_source(source)
    
    def evaluate(
        self,
        sequence: "MidiSequence",
        audio_data: "AudioSamples",
        intent: "Intent",
    ) -> float:
        """
        Evaluate using weighted combination of evaluators.
        
        Computes scores from both evaluators and combines them
        using the weights from GenerationProfile.
        
        Args:
            sequence: The MidiSequence to evaluate.
            audio_data: Synthesized audio bytes.
            intent: The original user intent.
        
        Returns:
            A float score in [0, 1] range. Returns 0.0 if no evaluators
            are available.
        """
        clap_score = 0.0
        heuristics_score = 0.0
        
        # Get CLAP score
        if self._clap_evaluator is not None:
            clap_score = self._clap_evaluator.evaluate(sequence, audio_data, intent)
        
        # Get heuristics score
        if self._heuristics_evaluator is not None:
            heuristics_score = self._heuristics_evaluator.evaluate(
                sequence, audio_data, intent
            )
        
        # Check if we have any evaluators
        if self._clap_evaluator is None and self._heuristics_evaluator is None:
            return 0.0
        
        # Get weights from profile
        clap_weight = self._profile.clap_weight
        # Heuristics weight combines key_weight and note_weight
        heuristics_weight = self._profile.key_weight + self._profile.note_weight
        
        # Normalize weights
        total_weight = clap_weight + heuristics_weight
        if total_weight == 0:
            return 0.0
        
        # Compute weighted score
        weighted_score = (
            clap_score * clap_weight + heuristics_score * heuristics_weight
        ) / total_weight
        
        return weighted_score
