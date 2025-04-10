#!/usr/bin/env python
"""
Digital Presenter - Main script to run the entire pipeline.
"""

import argparse
from pathlib import Path

from src.transcript import process_presentation
from src.voice import process_transcripts
from src.utils import get_project_paths, ensure_directory


def main():
    """Run the Digital Presenter pipeline."""
    parser = argparse.ArgumentParser(
        description="Digital Presenter - Generate transcripts and audio for presentations"
    )
    parser.add_argument(
        "--pptx",
        type=str,
        help="Path to PowerPoint file (if not provided, will look in data/input)",
    )
    parser.add_argument(
        "--skip-transcript", action="store_true", help="Skip transcript generation"
    )
    parser.add_argument(
        "--skip-audio", action="store_true", help="Skip audio generation"
    )
    parser.add_argument(
        "--language",
        type=str,
        choices=["chinese", "english"],
        default="chinese",
        help="Language for transcript generation (default: chinese)",
    )
    args = parser.parse_args()

    # Get project paths
    paths = get_project_paths()
    ensure_directory(paths["input_dir"])
    ensure_directory(paths["output_dir"])

    # Find PowerPoint file
    pptx_path = None
    if args.pptx:
        pptx_path = Path(args.pptx)
        if not pptx_path.exists():
            print(f"Error: File {pptx_path} does not exist")
            return
    else:
        # Look for PowerPoint files in the input directory
        pptx_files = list(paths["input_dir"].glob("*.pptx"))
        if not pptx_files:
            print(f"No PowerPoint files found in {paths['input_dir']}")
            return
        pptx_path = pptx_files[0]
        if len(pptx_files) > 1:
            print(f"Multiple PowerPoint files found. Using {pptx_path.name}")

    # Generate transcripts
    transcript_file = None
    if not args.skip_transcript:
        print(f"Generating transcripts for {pptx_path.name} in {args.language}...")
        output_file = paths["output_dir"] / f"{pptx_path.stem}_transcripts.json"
        process_presentation(pptx_path, output_file, target_language=args.language)
        transcript_file = output_file
    else:
        # Find existing transcript file
        transcript_files = list(
            paths["output_dir"].glob(f"{pptx_path.stem}_transcripts.json")
        )
        if transcript_files:
            transcript_file = transcript_files[0]
        else:
            print(f"No transcript file found for {pptx_path.name}")
            return

    # Generate audio
    if not args.skip_audio and transcript_file:
        print(f"Generating audio from transcript {transcript_file.name}...")
        output_dir = paths["output_dir"] / f"{pptx_path.stem}_audio"
        process_transcripts(transcript_file, output_dir)

    print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
