"""
Tests for CLI module.

Tests validate argument parsing and MIDI file output.
"""

import os
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

from domain.entities import GenerationProfile


class MockPipeline:
    """Mock Text2MidiPipeline for testing."""

    def __init__(self, midi_bytes: bytes = b"MOCK_MIDI_DATA"):
        self.midi_bytes = midi_bytes
        self.last_text = None
        self.last_profile = None

    def generate(self, text: str, profile: GenerationProfile) -> bytes:
        self.last_text = text
        self.last_profile = profile
        return self.midi_bytes


class TestCLIArgumentParsing(unittest.TestCase):
    """Test suite for CLI argument parsing."""

    def test_cli_module_exists(self):
        """
        AC3.1: cli.py module must exist.
        """
        import cli

        self.assertTrue(hasattr(cli, "main"))

    def test_parse_text_argument(self):
        """
        AC3.1: CLI must parse --text argument.
        """
        from cli import parse_args

        args = parse_args(["--text", "A peaceful sunrise"])

        self.assertEqual(args.text, "A peaceful sunrise")

    def test_parse_profile_argument(self):
        """
        AC3.1: CLI must parse --profile argument.
        """
        from cli import parse_args

        args = parse_args(["--text", "test", "--profile", "one-shot"])

        self.assertEqual(args.profile, "one-shot")

    def test_default_profile_is_balanced(self):
        """
        AC3.1: Default profile should be 'balanced'.
        """
        from cli import parse_args

        args = parse_args(["--text", "test"])

        self.assertEqual(args.profile, "balanced")

    def test_parse_output_argument(self):
        """
        AC3.1: CLI must parse --output argument.
        """
        from cli import parse_args

        args = parse_args(["--text", "test", "--output", "output.mid"])

        self.assertEqual(args.output, "output.mid")

    def test_default_output_is_output_mid(self):
        """
        AC3.1: Default output should be 'output.mid'.
        """
        from cli import parse_args

        args = parse_args(["--text", "test"])

        self.assertEqual(args.output, "output.mid")


class TestCLIExecution(unittest.TestCase):
    """Test suite for CLI execution flow."""

    def test_main_creates_pipeline(self):
        """
        AC3.2: CLI must instantiate Text2MidiPipeline.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.mid"

            with patch("cli.Text2MidiPipeline") as mock_pipeline_cls:
                mock_pipeline = MockPipeline()
                mock_pipeline_cls.return_value = mock_pipeline

                from cli import main

                main(["--text", "test melody", "--output", str(output_path)])

                mock_pipeline_cls.assert_called_once()

    def test_main_calls_generate_with_correct_arguments(self):
        """
        AC3.2: CLI must call generate with text and profile.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.mid"

            mock_pipeline = MockPipeline()

            with patch("cli.Text2MidiPipeline") as mock_pipeline_cls:
                mock_pipeline_cls.return_value = mock_pipeline

                from cli import main

                main(["--text", "test melody", "--profile", "one-shot", "--output", str(output_path)])

                self.assertEqual(mock_pipeline.last_text, "test melody")
                self.assertIsInstance(mock_pipeline.last_profile, GenerationProfile)

    def test_main_writes_midi_to_disk(self):
        """
        AC3.3: CLI must write MIDI bytes to output file.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.mid"
            expected_midi = b"MIDI_BINARY_DATA_12345"

            mock_pipeline = MockPipeline(midi_bytes=expected_midi)

            with patch("cli.Text2MidiPipeline") as mock_pipeline_cls:
                mock_pipeline_cls.return_value = mock_pipeline

                from cli import main

                main(["--text", "test", "--output", str(output_path)])

            # Verify file was written
            self.assertTrue(output_path.exists())
            self.assertEqual(output_path.read_bytes(), expected_midi)

    def test_main_with_default_output_path(self):
        """
        AC3.3: CLI must write to default output.mid if not specified.
        """
        mock_pipeline = MockPipeline(midi_bytes=b"TEST_MIDI")

        with patch("cli.Text2MidiPipeline") as mock_pipeline_cls:
            mock_pipeline_cls.return_value = mock_pipeline

            from cli import main

            # Change to temp dir to avoid polluting project
            with tempfile.TemporaryDirectory() as tmpdir:
                original_cwd = os.getcwd()
                try:
                    os.chdir(tmpdir)
                    main(["--text", "test"])

                    # Should have created output.mid
                    self.assertTrue(Path("output.mid").exists())
                finally:
                    os.chdir(original_cwd)
                    # Cleanup
                    if Path("output.mid").exists():
                        Path("output.mid").unlink()

    def test_main_with_all_profiles(self):
        """
        AC3.2: CLI must work with all predefined profiles.
        """
        profiles = ["one-shot", "balanced", "deep-search"]

        for profile_name in profiles:
            with self.subTest(profile=profile_name):
                with tempfile.TemporaryDirectory() as tmpdir:
                    output_path = Path(tmpdir) / f"test_{profile_name}.mid"

                    mock_pipeline = MockPipeline()

                    with patch("cli.Text2MidiPipeline") as mock_pipeline_cls:
                        mock_pipeline_cls.return_value = mock_pipeline

                        from cli import main

                        main([
                            "--text", "test",
                            "--profile", profile_name,
                            "--output", str(output_path),
                        ])

                        self.assertTrue(output_path.exists())


class TestCLIErrorHandling(unittest.TestCase):
    """Test suite for CLI error handling."""

    def test_invalid_profile_raises_error(self):
        """
        CLI should raise error for invalid profile name.
        """
        from cli import main

        with self.assertRaises(SystemExit):
            main(["--text", "test", "--profile", "invalid-profile"])

    def test_missing_text_raises_error(self):
        """
        CLI should raise error if text is not provided.
        """
        from cli import main

        with self.assertRaises(SystemExit):
            main([])


if __name__ == "__main__":
    unittest.main()
