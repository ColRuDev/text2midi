"""
Centralized test fixtures and mock classes for pytest.

This module provides shared mock implementations of domain interfaces
to avoid duplication across test files.
"""

from typing import List

import pytest

from domain.entities import Intent, MidiSequence
from domain.interfaces import AudioSamples, MidiBytes, PromptText, TokenId


class MockTranslator:
    """Mock LLMTranslator for testing."""

    def __init__(self, prompts: List[PromptText] | None = None):
        self.prompts = prompts or ["prompt1", "prompt2"]
        self.call_count = 0

    def translate(self, intent: Intent, num_variations: int) -> List[PromptText]:
        self.call_count += 1
        return self.prompts[:num_variations]


class MockGenerator:
    """Mock MidiGenerator for testing."""

    def __init__(
        self,
        tokens_per_call: int = 10,
        should_fail: bool = False,
        fail_after: int = 999,
    ):
        self.tokens_per_call = tokens_per_call
        self.should_fail = should_fail
        self.fail_after = fail_after
        self.call_count = 0
        self.total_tokens_generated = 0

    def generate_step(
        self,
        technical_prompt: PromptText,
        current_tokens: List[TokenId],
        num_tokens: int,
    ) -> List[TokenId]:
        self.call_count += 1
        self.total_tokens_generated += num_tokens

        if self.should_fail and self.call_count > self.fail_after:
            raise RuntimeError("Simulated generation failure")

        start = len(current_tokens)
        return list(range(start, start + num_tokens))

    def decode_to_midi(self, tokens: List[TokenId]) -> MidiBytes:
        return b"MIDI_DATA_" + bytes(str(len(tokens)), "utf-8")


class MockBatchGenerator:
    """Mock BatchMidiGenerator for testing."""

    def __init__(self, sequences: List[List[TokenId]] | None = None):
        self.sequences = sequences or [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        self.call_count = 0
        self.last_prompt = ""
        self.last_num_outputs = 0

    def generate_batch(
        self, technical_prompt: PromptText, num_outputs: int
    ) -> List[List[TokenId]]:
        self.call_count += 1
        self.last_prompt = technical_prompt
        self.last_num_outputs = num_outputs
        return self.sequences[:num_outputs]

    def decode_to_midi(self, tokens: List[TokenId]) -> MidiBytes:
        return b"MIDI_" + bytes(str(len(tokens)), "utf-8")


class MockAudioRenderer:
    """Mock AudioRenderer for testing."""

    def render(self, tokens: List[TokenId]) -> AudioSamples:
        return b"AUDIO_" + bytes(str(len(tokens)), "utf-8")


class MockEvaluator:
    """Mock Evaluator for testing."""

    def __init__(self, rewards: List[float] | None = None, reward: float | None = None):
        # Support both single reward and list of rewards
        if reward is not None:
            self.rewards = [reward]
        else:
            self.rewards = rewards or [0.5, 0.7, 0.6]
        self.call_count = 0
        self.evaluations: List[tuple] = []

    def evaluate(
        self,
        sequence: MidiSequence,
        audio_data: AudioSamples,
        intent: Intent,
    ) -> float:
        self.call_count += 1
        reward = self.rewards[self.call_count % len(self.rewards)]
        self.evaluations.append((sequence.technical_prompt, reward))
        return reward


# Pytest fixtures that return the mock classes themselves,
# allowing tests to instantiate them as needed
@pytest.fixture
def mock_translator_class():
    """Return the MockTranslator class for test instantiation."""
    return MockTranslator


@pytest.fixture
def mock_generator_class():
    """Return the MockGenerator class for test instantiation."""
    return MockGenerator


@pytest.fixture
def mock_batch_generator_class():
    """Return the MockBatchGenerator class for test instantiation."""
    return MockBatchGenerator


@pytest.fixture
def mock_evaluator_class():
    """Return the MockEvaluator class for test instantiation."""
    return MockEvaluator


@pytest.fixture
def mock_audio_renderer_class():
    """Return the MockAudioRenderer class for test instantiation."""
    return MockAudioRenderer
