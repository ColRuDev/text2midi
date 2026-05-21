"""
Token-Level Heuristics Evaluator - Memory-based MIDI token validation.

This use case implements fast, in-memory token evaluation for MIDI generation,
operating directly on REMI TokenId arrays using an inverted vocabulary mapping.

Architecture:
    - Implements domain.interfaces.Evaluator
    - Uses inverted vocabulary for token parsing (no disk I/O)
    - Applies exponential penalties for constraint violations
    - Supports strict instrument control via configuration

The evaluator parses Pitch_X, Program_X, and TimeSig_X/Y events from the
token stream to enforce key/scale intervals, requested instruments, and
time signatures deterministically during beam search.
"""

from __future__ import annotations

import re
from typing import Optional, Set

from domain.entities import ClapPromptSource, Intent, MidiSequence
from domain.interfaces import AudioSamples, Evaluator


# Major scale intervals (semitones from root)
MAJOR_SCALE_INTERVALS = {0, 2, 4, 5, 7, 9, 11}

# Minor scale intervals (semitones from root)
MINOR_SCALE_INTERVALS = {0, 2, 3, 5, 7, 8, 10}

# Note name to pitch class mapping
NOTE_NAMES = {
    "c": 0, "c#": 1, "db": 1, "d": 2, "d#": 3, "eb": 3,
    "e": 4, "f": 5, "f#": 6, "gb": 6, "g": 7, "g#": 8,
    "ab": 8, "a": 9, "a#": 10, "bb": 10, "b": 11,
}


class TokenHeuristics(Evaluator):
    """
    Memory-based token-level heuristics evaluator.

    This evaluator operates directly on TokenId arrays using an inverted
    vocabulary mapping, avoiding disk I/O for fast evaluation during beam search.

    Attributes:
        vocab_mapping: Dict mapping TokenId (int) -> EventName (str).
        strict_instruments: When True, penalize unrequested instruments.
        requested_programs: Set of requested MIDI program numbers (0-127, -1 for drums).
        key_weight: Weight for key consistency score.
        note_weight: Weight for time signature/instrument consistency scores.
        clap_prompt_source: Inherited from Evaluator interface.

    Example:
        >>> from domain.remi_vocab import get_inverted_vocab
        >>> evaluator = TokenHeuristics(vocab_mapping=get_inverted_vocab())
        >>> score = evaluator.evaluate(sequence, audio_data, intent)
    """

    def __init__(
        self,
        vocab_mapping: dict[int, str],
        strict_instruments: bool = False,
        requested_programs: Optional[Set[int]] = None,
        key_weight: float = 0.5,
        note_weight: float = 0.5,
    ) -> None:
        """
        Initialize TokenHeuristics with vocabulary and configuration.

        Args:
            vocab_mapping: Inverted vocabulary mapping TokenId -> EventName.
            strict_instruments: Whether to penalize unrequested instruments.
            requested_programs: Set of requested MIDI program numbers.
            key_weight: Weight for key consistency score (default 0.5).
            note_weight: Weight for time signature/instrument scores (default 0.5).
        """
        self._vocab_mapping = vocab_mapping
        self._strict_instruments = strict_instruments
        self._requested_programs = requested_programs or set()
        self._key_weight = key_weight
        self._note_weight = note_weight
        self.clap_prompt_source: str = ClapPromptSource.TECHNICAL

    def evaluate(
        self,
        sequence: MidiSequence,
        audio_data: AudioSamples,
        intent: Intent,
    ) -> float:
        """
        Evaluate MIDI sequence using memory-based token heuristics.

        Parses tokens in memory and applies penalties for:
        - Key/scale violations (Pitch tokens)
        - Unrequested instruments (Program tokens, when strict_instruments=True)
        - Time signature mismatches (TimeSig tokens)

        Args:
            sequence: The MidiSequence to evaluate, containing tokens.
            audio_data: Audio bytes (not used for token heuristics).
            intent: Original user intent (not used for heuristics).

        Returns:
            A float reward score in range [0, 1]. Higher is better.
        """
        tokens = sequence.tokens

        if not tokens:
            # Empty sequence gets low score
            return 0.1

        # Parse constraints from technical prompt
        key_info = self._parse_key_from_prompt(sequence.technical_prompt)
        requested_timesig = self._parse_timesig_from_prompt(sequence.technical_prompt)

        # Resolve requested programs: use instance-level if set, otherwise parse from prompt
        requested_programs = self._requested_programs
        if not requested_programs:
            parsed_programs = self._parse_programs_from_prompt(sequence.technical_prompt)
            if parsed_programs:
                requested_programs = parsed_programs

        # Compute individual scores
        key_score = self._compute_key_consistency(tokens, key_info)
        timesig_score = self._compute_timesig_consistency(tokens, requested_timesig)
        instrument_score = self._compute_instrument_consistency(tokens, requested_programs)

        # Combine scores with injected weights from GenerationProfile
        # note_weight is split between timesig and instrument scores
        total_weight = self._key_weight + self._note_weight
        if total_weight == 0:
            return 0.5  # Neutral score if no weights

        # Split note_weight evenly between timesig and instrument
        timesig_portion = self._note_weight / 2
        instrument_portion = self._note_weight / 2

        combined_score = (
            key_score * self._key_weight +
            timesig_score * timesig_portion +
            instrument_score * instrument_portion
        ) / total_weight

        return combined_score

    def _parse_key_from_prompt(self, prompt: str) -> Optional[dict]:
        """
        Parse key information from technical prompt.

        Args:
            prompt: Technical prompt string (e.g., "tempo:80 key:C_major ..." or natural language).

        Returns:
            Dict with 'root' (0-11) and 'scale' ('major' or 'minor'), or None
            if no key was specified in the prompt.
        """
        prompt_lower = prompt.lower()

        # Try to match natural language format "key of C major"
        key_match = re.search(r"key of\s+([a-g][#b]?)\s+(major|minor)", prompt_lower)
        if key_match:
            note_name = key_match.group(1)
            scale = key_match.group(2)
            if note_name in NOTE_NAMES:
                return {"root": NOTE_NAMES[note_name], "scale": scale}

        # Try to extract root note and scale together from strict key:X_scale pattern
        key_match = re.search(r"key:\s*([a-g][#b]?)[_ -](major|minor)", prompt_lower)
        if key_match:
            note_name = key_match.group(1)
            scale = key_match.group(2)
            if note_name not in NOTE_NAMES:
                return None  # Invalid key name
            return {"root": NOTE_NAMES[note_name], "scale": scale}

        # Fall back to just root note (default to major scale)
        key_match = re.search(r"key:\s*([a-g][#b]?)", prompt_lower)
        if not key_match:
            return None  # No key specified

        note_name = key_match.group(1)
        if note_name not in NOTE_NAMES:
            return None  # Invalid key name

        return {"root": NOTE_NAMES[note_name], "scale": "major"}

    def _parse_timesig_from_prompt(self, prompt: str) -> Optional[str]:
        """
        Parse time signature from technical prompt.

        Args:
            prompt: Technical prompt string.

        Returns:
            Time signature string (e.g., "4/4") or None.
        """
        prompt_lower = prompt.lower()

        # Look for natural language fractions (e.g., "4/4 time signature") or strict tags
        timesig_match = re.search(r"\b(\d+/\d+)\b", prompt_lower)
        if timesig_match:
            return timesig_match.group(1)

        return None

    def _parse_programs_from_prompt(self, prompt: str) -> Set[int]:
        """
        Parse requested instruments from technical prompt.

        Args:
            prompt: Technical prompt string.

        Returns:
            Set of MIDI program numbers.
        """
        programs: Set[int] = set()

        prompt_lower = prompt.lower()

        # Natural language extraction: check if instrument names exist in the prompt
        # We replace underscores with spaces in INSTRUMENT_TO_PROGRAM keys
        for inst_key, program in INSTRUMENT_TO_PROGRAM.items():
            search_str = inst_key.replace("_", " ")
            # Use word boundaries to avoid matching "bass" inside "bassoon"
            if re.search(r"\b" + re.escape(search_str) + r"\b", prompt_lower):
                programs.add(program)

        # Fallback to strict instruments:X pattern
        # Use lookahead to stop before next key (e.g., "tempo:") or end of string
        instruments_match = re.search(
            r"instruments:\s*([a-z0-9_, ]+?)(?=\s+[a-z]+:|$)", prompt_lower
        )
        if instruments_match:
            instruments_str = instruments_match.group(1)
            # Parse instrument names
            for name in instruments_str.split(","):
                name = name.strip()
                if name in INSTRUMENT_TO_PROGRAM:
                    programs.add(INSTRUMENT_TO_PROGRAM[name])

        return programs

    def _compute_key_consistency(self, tokens: list[int], key_info: Optional[dict]) -> float:
        """
        Compute how well pitch tokens fit the specified key.

        Args:
            tokens: List of TokenIds.
            key_info: Dict with 'root' and 'scale', or None if no key specified.

        Returns:
            Score in [0, 1] based on percentage of in-key pitch tokens.
            Returns 0.5 (neutral) if no key was specified.
        """
        # Neutral score if no key constraint was specified
        if key_info is None:
            return 0.5

        root = key_info["root"]
        scale = key_info["scale"]

        # Select scale intervals
        scale_intervals = (
            MINOR_SCALE_INTERVALS if scale == "minor"
            else MAJOR_SCALE_INTERVALS
        )

        # Find all pitch tokens
        pitch_tokens = []
        for token in tokens:
            event_name = self._vocab_mapping.get(token, "")
            if event_name.startswith("Pitch_"):
                # Extract pitch value from "Pitch_X"
                try:
                    pitch = int(event_name.split("_")[1])
                    pitch_tokens.append(pitch)
                except (IndexError, ValueError):
                    continue

        if not pitch_tokens:
            return 0.5  # Neutral score if no pitch tokens

        # Count in-key pitches
        in_key_count = 0
        for pitch in pitch_tokens:
            interval = (pitch - root) % 12
            if interval in scale_intervals:
                in_key_count += 1

        # Compute score with exponential penalty for out-of-key
        # More out-of-key = exponentially worse score
        out_of_key_ratio = 1.0 - (in_key_count / len(pitch_tokens))
        score = 1.0 - (out_of_key_ratio ** 2)  # Exponential penalty

        return max(0.1, score)

    def _compute_timesig_consistency(
        self, tokens: list[int], requested_timesig: Optional[str]
    ) -> float:
        """
        Compute time signature consistency score.

        Args:
            tokens: List of TokenIds.
            requested_timesig: Requested time signature (e.g., "4/4").

        Returns:
            Score in [0, 1].
        """
        if not requested_timesig:
            return 0.5  # Neutral score if no time signature specified

        # Find all TimeSig tokens in the sequence
        timesig_tokens = []
        for token in tokens:
            event_name = self._vocab_mapping.get(token, "")
            if event_name.startswith("TimeSig_"):
                # Extract time signature from "TimeSig_X/Y"
                timesig = event_name.replace("TimeSig_", "")
                timesig_tokens.append(timesig)

        if not timesig_tokens:
            return 0.5  # Neutral score if no TimeSig tokens found

        # Check if first TimeSig matches requested
        first_timesig = timesig_tokens[0]
        if first_timesig == requested_timesig:
            return 1.0

        # Apply exponential penalty for mismatch
        return 0.3

    def _compute_instrument_consistency(
        self, tokens: list[int], requested_programs: Set[int]
    ) -> float:
        """
        Compute instrument consistency score.

        If strict_instruments is True, penalize branches containing
        instruments not in the requested set.

        Args:
            tokens: List of TokenIds.
            requested_programs: Set of requested MIDI program numbers.

        Returns:
            Score in [0, 1].
        """
        # Find all Program tokens in the sequence
        found_programs: Set[int] = set()
        for token in tokens:
            event_name = self._vocab_mapping.get(token, "")
            if event_name.startswith("Program_"):
                # Extract program number from "Program_X"
                try:
                    program = int(event_name.split("_")[1])
                    found_programs.add(program)
                except (IndexError, ValueError):
                    continue

        if not found_programs:
            return 0.5  # Neutral score if no Program tokens

        # If strict mode is off, return neutral score
        if not self._strict_instruments:
            return 0.5

        # If no requested programs specified, return neutral score
        if not requested_programs:
            return 0.5

        # Check if all found programs are in requested set
        unrequested = found_programs - requested_programs

        if not unrequested:
            return 1.0  # All instruments match

        # Apply severe penalty for unrequested instruments
        # Strict mode means STRICT. Any hallucinated instrument drops the score to minimum.
        return 0.1


# Instrument name to MIDI program number mapping
INSTRUMENT_TO_PROGRAM = {
    "piano": 0,
    "acoustic_piano": 0,
    "bright_piano": 1,
    "electric_piano": 2,
    "honky_tonk": 3,
    "rhodes": 4,
    "clavinet": 7,
    "celesta": 8,
    "glockenspiel": 9,
    "music_box": 10,
    "vibraphone": 11,
    "marimba": 12,
    "xylophone": 13,
    "tubular_bells": 14,
    "organ": 16,
    "hammond_organ": 16,
    "pipe_organ": 19,
    "accordion": 21,
    "harmonica": 22,
    "guitar": 24,
    "acoustic_guitar": 24,
    "electric_guitar": 27,
    "bass": 32,
    "acoustic_bass": 32,
    "electric_bass": 33,
    "violin": 40,
    "viola": 41,
    "cello": 42,
    "contrabass": 43,
    "strings": 48,
    "ensemble": 48,
    "synth_strings": 50,
    "choir": 52,
    "voice": 52,
    "trumpet": 56,
    "trombone": 57,
    "tuba": 58,
    "french_horn": 60,
    "brass": 61,
    "saxophone": 66,
    "sax": 66,
    "oboe": 68,
    "english_horn": 69,
    "bassoon": 70,
    "clarinet": 71,
    "flute": 73,
    "piccolo": 72,
    "recorder": 74,
    "pan_flute": 75,
    "bottle": 76,
    "shakuhachi": 77,
    "whistle": 78,
    "ocarina": 79,
    "synth_lead": 80,
    "lead": 80,
    "synth_pad": 88,
    "pad": 88,
    "synth_fx": 96,
    "sitar": 104,
    "banjo": 105,
    "shamisen": 106,
    "koto": 107,
    "kalimba": 108,
    "bagpipe": 109,
    "fiddle": 110,
    "shanai": 111,
    "tinkle": 112,
    "agogo": 113,
    "steel_drums": 114,
    "woodblock": 115,
    "taiko": 116,
    "melodic_tom": 117,
    "synth_drum": 118,
    "reverse_cymbal": 119,
    "fret_noise": 120,
    "breath_noise": 121,
    "seashore": 122,
    "bird": 123,
    "telephone": 124,
    "helicopter": 125,
    "applause": 126,
    "gunshot": 127,
    "drums": -1,
    "drum": -1,
    "percussion": -1,
}
