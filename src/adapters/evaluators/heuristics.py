"""
Heuristics Evaluator - Music theory-based MIDI evaluation.

This adapter evaluates MIDI sequences using music theory heuristics
such as key consistency, note distribution, and harmonic structure.

Architecture:
    - Implements domain.interfaces.Evaluator
    - Uses rule-based heuristics for music theory analysis
    - No external dependencies (pure Python)
    - Returns scores based on key/interval analysis
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from domain.entities import Intent, MidiSequence
    from domain.interfaces import AudioSamples


# Major scale intervals (semitones from root)
MAJOR_SCALE_INTERVALS = {0, 2, 4, 5, 7, 9, 11}

# Minor scale intervals (semitones from root)
MINOR_SCALE_INTERVALS = {0, 2, 3, 5, 7, 8, 10}


class HeuristicsEvaluator:
    """
    Evaluator using music theory heuristics.
    
    This evaluator analyzes MIDI sequences for key consistency,
    note distribution, and other musical properties.
    
    The adapter:
    - Parses key information from technical_prompt
    - Checks if notes belong to the specified key
    - Computes a score based on music theory rules
    - Handles missing or malformed prompts gracefully
    
    Example:
        >>> evaluator = HeuristicsEvaluator()
        >>> score = evaluator.evaluate(sequence, audio_bytes, intent)
    """
    
    def evaluate(
        self,
        sequence: "MidiSequence",
        audio_data: "AudioSamples",
        intent: "Intent",
    ) -> float:
        """
        Evaluate MIDI sequence using music theory heuristics.
        
        Analyzes the token sequence for key consistency and
        musical structure quality.
        
        Args:
            sequence: The MidiSequence to evaluate.
            audio_data: Audio bytes (not used for heuristics).
            intent: The original intent (not used for heuristics).
        
        Returns:
            A float score in [0, 1] range based on key consistency
            and note distribution.
        """
        tokens = sequence.tokens
        
        if not tokens:
            # Empty sequence gets low score
            return 0.1
        
        # Parse key from technical prompt
        key_info = self._parse_key_from_prompt(sequence.technical_prompt)
        
        # Compute key consistency score
        key_score = self._compute_key_consistency(tokens, key_info)
        
        # Compute note distribution score
        note_score = self._compute_note_distribution(tokens)
        
        # Combine scores (equal weighting)
        combined_score = (key_score + note_score) / 2.0
        
        return combined_score
    
    def _parse_key_from_prompt(self, prompt: str) -> dict:
        """
        Parse key information from technical prompt.
        
        Args:
            prompt: Technical prompt string (e.g., "tempo:80 key:C_major ...")
        
        Returns:
            Dict with 'root' (0-11) and 'scale' ('major' or 'minor').
        """
        # Default to C major
        result = {"root": 0, "scale": "major"}
        
        # Look for key specification in prompt
        prompt_lower = prompt.lower()
        
        # Check for minor
        if "minor" in prompt_lower:
            result["scale"] = "minor"
        else:
            result["scale"] = "major"
        
        # Try to extract root note
        note_names = ["c", "c#", "d", "d#", "e", "f", "f#", "g", "g#", "a", "a#", "b"]
        flat_names = ["c", "db", "d", "eb", "e", "f", "gb", "g", "ab", "a", "bb", "b"]
        
        for i, name in enumerate(note_names):
            if f"key:{name}" in prompt_lower or f"key: {name}" in prompt_lower:
                result["root"] = i
                break
            # Check flat names
            if i < len(flat_names) and f"key:{flat_names[i]}" in prompt_lower:
                result["root"] = i
                break
        
        return result
    
    def _compute_key_consistency(self, tokens: List[int], key_info: dict) -> float:
        """
        Compute how well tokens fit the specified key.
        
        Args:
            tokens: List of MIDI note values.
            key_info: Dict with 'root' and 'scale'.
        
        Returns:
            Score in [0, 1] based on percentage of in-key notes.
        """
        root = key_info["root"]
        scale = key_info["scale"]
        
        # Select scale intervals
        if scale == "minor":
            scale_intervals = MINOR_SCALE_INTERVALS
        else:
            scale_intervals = MAJOR_SCALE_INTERVALS
        
        # Count notes in key
        in_key_count = 0
        total_valid_notes = 0
        
        for token in tokens:
            # Only consider valid MIDI note range
            if 0 <= token <= 127:
                total_valid_notes += 1
                # Compute interval from root
                interval = (token - root) % 12
                if interval in scale_intervals:
                    in_key_count += 1
        
        if total_valid_notes == 0:
            return 0.1
        
        return in_key_count / total_valid_notes
    
    def _compute_note_distribution(self, tokens: List[int]) -> float:
        """
        Compute note distribution quality.
        
        A good distribution has variety without too much repetition.
        
        Args:
            tokens: List of MIDI note values.
        
        Returns:
            Score in [0, 1] based on note variety.
        """
        if not tokens:
            return 0.1
        
        # Get unique notes
        unique_notes = set(t for t in tokens if 0 <= t <= 127)
        
        if not unique_notes:
            return 0.1
        
        # Score based on variety (more unique notes = better, up to a point)
        # Ideal is around 7-12 unique notes for a melody
        variety_score = min(len(unique_notes) / 12.0, 1.0)
        
        return variety_score
