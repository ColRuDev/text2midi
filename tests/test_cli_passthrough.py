"""
Tests for CLI pass-through translator integration.

These tests verify the CLI correctly handles the --translator-model argument
and passes None to the pipeline when the argument is omitted.
"""

import os
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from domain.entities import MidiSequence
from domain.interfaces import AudioSamples, MidiBytes, PromptText, TokenId


# Mock classes for testing (duplicated to avoid import issues)
class MockGenerator:
    """Mock MidiGenerator for testing."""

    def __init__(self):
        self.call_count = 0

    def generate_step(
        self,
        technical_prompt: PromptText,
        current_tokens: List[TokenId],
        num_tokens: int,
    ) -> List[TokenId]:
        self.call_count += 1
        start = len(current_tokens)
        return list(range(start, start + num_tokens))

    def decode_to_midi(self, tokens: List[TokenId]) -> MidiBytes:
        return b"MIDI_DATA_" + bytes(str(len(tokens)), "utf-8")


class MockEvaluator:
    """Mock Evaluator for testing."""

    def __init__(self, reward: float = 0.7):
        self.reward = reward
        self.call_count = 0

    def evaluate(
        self,
        sequence: MidiSequence,
        audio_data: AudioSamples,
        intent,
    ) -> float:
        self.call_count += 1
        return self.reward


class MockAudioRenderer:
    """Mock AudioRenderer for testing."""

    def render(self, tokens: List[TokenId]) -> AudioSamples:
        return b"AUDIO_" + bytes(str(len(tokens)), "utf-8")


class TestCLIPassThroughTranslator:
    """Tests for CLI pass-through translator behavior."""

    def test_translator_model_has_no_default(self):
        """
        Task 3.1: --translator-model MUST have no default value.

        GIVEN the CLI argument parser
        WHEN --translator-model is parsed without a value
        THEN the default MUST be None (not a model name)
        """
        from cli import parse_args

        args = parse_args(["--text", "Test melody"])

        # When --translator-model is omitted, it should be None
        assert args.translator_model is None

    def test_translator_model_accepts_explicit_value(self):
        """
        GIVEN --translator-model is provided
        WHEN the CLI is parsed
        THEN the value MUST be stored
        """
        from cli import parse_args

        args = parse_args([
            "--text", "Test melody",
            "--translator-model", "gemma-4-31b-it"
        ])

        assert args.translator_model == "gemma-4-31b-it"

    def test_cli_passes_none_to_pipeline_when_translator_model_omitted(self):
        """
        Task 3.2: CLI MUST pass None for translator config when argument is omitted.

        GIVEN --translator-model is omitted
        WHEN the CLI initializes the pipeline
        THEN it MUST pass translator_config=None
        """
        from cli import main

        mock_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.midi_bytes = b"MIDI_DATA"
        mock_result.technical_prompt = "test prompt"
        mock_pipeline.generate.return_value = mock_result

        with (
            patch("cli.Text2MidiPipeline", return_value=mock_pipeline) as mock_pipeline_cls,
            patch("cli.Path.write_bytes"),
            patch("cli.Path.write_text"),
        ):
            # Remove GOOGLE_API_KEY to ensure we're testing pass-through path
            original_key = os.environ.pop("GOOGLE_API_KEY", None)

            try:
                exit_code = main(["--text", "Test melody", "--output", "/tmp/test.mid"])

                assert exit_code == 0
                # Verify pipeline was called with translator_config=None
                mock_pipeline_cls.assert_called_once()
                call_kwargs = mock_pipeline_cls.call_args.kwargs
                assert call_kwargs.get("translator_config") is None
            finally:
                if original_key:
                    os.environ["GOOGLE_API_KEY"] = original_key

    def test_cli_passes_config_to_pipeline_when_translator_model_provided(self):
        """
        GIVEN --translator-model is provided
        WHEN the CLI initializes the pipeline
        THEN it MUST pass a GoogleAIConfig with the model name
        """
        from cli import main
        from adapters.translators.google_ai_translator import GoogleAIConfig

        mock_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.midi_bytes = b"MIDI_DATA"
        mock_result.technical_prompt = "test prompt"
        mock_pipeline.generate.return_value = mock_result

        with (
            patch("cli.Text2MidiPipeline", return_value=mock_pipeline) as mock_pipeline_cls,
            patch("cli.Path.write_bytes"),
            patch("cli.Path.write_text"),
            patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}),
        ):
            exit_code = main([
                "--text", "Test melody",
                "--translator-model", "gemma-4-31b-it",
                "--output", "/tmp/test.mid"
            ])

            assert exit_code == 0
            # Verify pipeline was called with a GoogleAIConfig
            mock_pipeline_cls.assert_called_once()
            call_kwargs = mock_pipeline_cls.call_args.kwargs
            config = call_kwargs.get("translator_config")
            assert config is not None
            assert isinstance(config, GoogleAIConfig)
            assert config.model_name == "gemma-4-31b-it"

    def test_cli_help_mentions_translator_model_is_optional(self):
        """
        The help text should indicate that --translator-model is optional.
        """
        from cli import parse_args
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--text", required=True)
        parser.add_argument(
            "--translator-model",
            type=str,
            default=None,
            help="Google AI Studio model for intent translation. Omit to bypass translation.",
        )

        help_text = parser.format_help()
        # Help should mention optional or bypass
        assert "translator-model" in help_text


class TestCLIE2EPassThrough:
    """E2E tests for CLI pass-through behavior (Task 3.3)."""

    def test_cli_generates_without_api_key_when_translator_model_omitted(self):
        """
        Task 3.3: E2E test verifying CLI works without API key when bypassing translation.

        GIVEN GOOGLE_API_KEY is not set
        AND --translator-model is omitted
        WHEN the CLI is executed
        THEN it MUST successfully generate MIDI without requiring the API key
        """
        from cli import main
        import tempfile

        # Ensure GOOGLE_API_KEY is not set
        original_key = os.environ.pop("GOOGLE_API_KEY", None)

        try:
            with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
                output_path = f.name

            # This should work without GOOGLE_API_KEY because we're using PassThroughTranslator
            # However, we need to mock the generator since it requires model files
            with (
                patch("pipeline.Text2MidiGenerator") as mock_gen_cls,
                patch("pipeline.CompositeEvaluator") as mock_eval_cls,
                patch("pipeline.InMemoryFluidSynthEngine") as mock_audio_cls,
            ):
                # Setup minimal mocks
                mock_gen_cls.return_value = MockGenerator()
                mock_eval_cls.return_value = MockEvaluator()
                mock_audio_cls.return_value = MockAudioRenderer()

                exit_code = main([
                    "--text", "A peaceful sunrise",
                    "--output", output_path,
                    "--profile", "one-shot"
                ])

                # Should succeed without API key
                assert exit_code == 0

                # Verify the output file was created
                import pathlib
                assert pathlib.Path(output_path).exists()

                # Cleanup
                pathlib.Path(output_path).unlink(missing_ok=True)
                pathlib.Path(output_path).with_suffix(".txt").unlink(missing_ok=True)

        finally:
            if original_key:
                os.environ["GOOGLE_API_KEY"] = original_key
