#!/usr/bin/env python
"""
Digital Presenter - Main script to run the entire pipeline.
"""

import argparse
from pathlib import Path

from src.transcript import process_presentation
from src.voice import process_pptx_for_audio
from src.utils import get_project_paths, ensure_directory


def main():
    """Run the Digital Presenter pipeline."""
    parser = argparse.ArgumentParser(
        description="Digital Presenter - Generate transcripts and audio for presentations"
    )
    parser.add_argument(
        "--pptx",
        type=str,
        help="Path to PowerPoint file (if not provided, will look in data/raw)",
    )
    parser.add_argument(
        "--skip-audio", action="store_true", help="Skip audio generation"
    )
    parser.add_argument(
        "--direct-audio", action="store_true", 
        help="Generate audio directly from PowerPoint notes without generating separate transcript file"
    )
    parser.add_argument(
        "--language",
        type=str,
        choices=["chinese", "english"],
        default="chinese",
        help="Language for transcript generation (default: chinese)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Model to use for transcript generation (default: gpt-4o)",
    )
    parser.add_argument(
        "--tts-provider",
        type=str,
        choices=["minimax", "openai"],
        default="minimax",
        help="Text-to-speech provider to use (default: minimax, which is better for Chinese; openai is better for English)",
    )
    args = parser.parse_args()

    # Get project paths
    paths = get_project_paths()
    ensure_directory(paths["raw_dir"])
    ensure_directory(paths["noted_dir"])
    ensure_directory(paths["audio_dir"])

    # Find PowerPoint file
    pptx_path = None
    if args.pptx:
        pptx_path = Path(args.pptx)
        if not pptx_path.exists():
            print(f"Error: File {pptx_path} does not exist")
            return
    else:
        # Look for PowerPoint files in the raw directory
        pptx_files = list(paths["raw_dir"].glob("*.pptx"))
        if not pptx_files:
            print(f"No PowerPoint files found in {paths['raw_dir']}")
            return
        pptx_path = pptx_files[0]
        if len(pptx_files) > 1:
            print(f"Multiple PowerPoint files found. Using {pptx_path.name}")

    # Direct audio generation from PowerPoint notes
    if args.direct_audio:
        # Check if the file is in the noted directory
        if "noted" in str(pptx_path):
            noted_pptx = pptx_path
        else:
            # Look for the noted version in the noted directory
            noted_pptx = paths["noted_dir"] / f"{pptx_path.stem}_noted.pptx"
            if not noted_pptx.exists():
                print(f"No noted version of {pptx_path.name} found in {paths['noted_dir']}")
                print("Please run the transcript generation process first.")
                return
        
        print(f"Generating audio directly from {noted_pptx.name} notes...")
        print(f"Using {args.tts_provider} for TTS generation")
        
        # Save audio files to the same directory as the noted PPTX
        process_pptx_for_audio(noted_pptx, None, provider=args.tts_provider)
        print("Pipeline completed successfully!")
        return

    # Generate transcripts
    print(f"Generating transcripts for {pptx_path.name} in {args.language} using {args.model}...")
    output_dir = paths["noted_dir"]
    noted_pptx = process_presentation(pptx_path, output_dir, target_language=args.language, model=args.model)
    
    # Generate audio if not skipped
    if not args.skip_audio:
        print(f"Generating audio from {noted_pptx.name}...")
        print(f"Using {args.tts_provider} for TTS generation")
        
        # Save audio files to the same directory as the noted PPTX
        process_pptx_for_audio(noted_pptx, None, provider=args.tts_provider)

    print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
