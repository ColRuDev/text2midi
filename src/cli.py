"""
Command Line Interface for Text2Midi.

This module provides the CLI entry point for text-to-MIDI generation.
It parses user arguments, instantiates the pipeline, and writes MIDI to disk.

Architecture:
    - Uses argparse for argument parsing
    - Creates Text2MidiPipeline instance
    - Maps profile names to GenerationProfile instances
    - Writes binary MIDI output to disk

Usage:
    python -m cli --text "A peaceful sunrise" --profile balanced --output output.mid
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from config.profiles import BALANCED, DEEP_SEARCH, ONE_SHOT, PROFILES
from domain.entities import GenerationProfile
from pipeline import Text2MidiPipeline

logger = logging.getLogger(__name__)

# Mapping of profile names to profile instances
PROFILE_MAP: dict[str, GenerationProfile] = PROFILES


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command line arguments.

    Args:
        args: Command line arguments (defaults to sys.argv[1:]).

    Returns:
        Parsed arguments namespace with:
        - text: The natural language description of desired music
        - profile: Profile name ('one-shot', 'balanced', 'deep-search')
        - output: Output file path for MIDI file

    Example:
        >>> args = parse_args(["--text", "A melody", "--profile", "one-shot"])
        >>> args.text
        'A melody'
        >>> args.profile
        'one-shot'
    """
    parser = argparse.ArgumentParser(
        prog="text2midi",
        description="Generate MIDI from natural language descriptions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  text2midi --text "A peaceful sunrise at the beach"
  text2midi --text "Jazz piano" --profile one-shot --output jazz.mid
  text2midi --text "Symphony" --profile deep-search --output symphony.mid
        """,
    )

    parser.add_argument(
        "--text",
        "-t",
        type=str,
        required=True,
        help="Natural language description of the music to generate.",
    )

    parser.add_argument(
        "--profile",
        "-p",
        type=str,
        choices=list(PROFILE_MAP.keys()),
        default="balanced",
        help="Generation profile: 'one-shot' (fast), 'balanced' (default), or 'deep-search' (quality).",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output.mid",
        help="Output MIDI file path (default: output.mid).",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging.",
    )

    return parser.parse_args(args)


def main(argv: list[str] | None = None) -> int:
    """
    Main entry point for the CLI.

    Parses arguments, creates the pipeline, generates MIDI, and writes to disk.

    Args:
        argv: Command line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code: 0 for success, 1 for error.

    Example:
        >>> exit_code = main(["--text", "test", "--output", "test.mid"])
        >>> exit_code
        0
    """
    # Parse arguments
    args = parse_args(argv)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Resolve profile
    profile = PROFILE_MAP.get(args.profile)
    if profile is None:
        logger.error(f"Invalid profile: {args.profile}")
        print(f"Error: Invalid profile '{args.profile}'. Choose from: {list(PROFILE_MAP.keys())}", file=sys.stderr)
        return 1

    output_path = Path(args.output)

    try:
        # Create pipeline (loads heavy adapters once)
        logger.info("Initializing pipeline...")
        pipeline = Text2MidiPipeline()

        # Generate MIDI
        logger.info(f"Generating MIDI for: '{args.text}'")
        midi_bytes = pipeline.generate(text=args.text, profile=profile)

        # Write to disk
        logger.info(f"Writing MIDI to: {output_path}")
        output_path.write_bytes(midi_bytes)

        print(f"Successfully generated MIDI: {output_path}")
        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return 1

    except Exception as e:
        logger.exception(f"Generation failed: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
