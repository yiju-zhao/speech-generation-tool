#!/usr/bin/env python
"""
Digital Presenter - Main script to run the entire pipeline.
"""

from pathlib import Path

from src.transcript import process_presentation as standard_process_presentation
from src.transcript_processor import process_presentation_with_storm
from src.voice import process_pptx_for_audio
from src.utils import get_project_paths, ensure_directory
from config.parse_args import parse_args


def main():
    """Run the Digital Presenter pipeline."""
    args = parse_args()

    # Extract arguments
    pptx_path_arg = args.pptx
    skip_audio = args.skip_audio
    direct_audio = args.direct_audio
    language = args.language
    model = args.model
    tts_provider = args.tts_provider
    llm_provider = args.llm_provider
    use_storm = args.use_storm
    disable_search = args.disable_search
    slides=args.slides

    # Get project paths
    paths = get_project_paths()
    ensure_directory(paths["raw_dir"])
    ensure_directory(paths["processed_dir"])
    ensure_directory(paths["audio_dir"])

    # Find PowerPoint file
    pptx_path = None
    if pptx_path_arg:
        pptx_path = Path(pptx_path_arg)
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
    if direct_audio:
        # Check if the file is in the noted directory
        if "noted" in str(pptx_path):
            noted_pptx = pptx_path
        else:
            # Look for the noted version in the noted directory
            noted_pptx = paths["processed_dir"] / f"{pptx_path.stem}_noted.pptx"
            if not noted_pptx.exists():
                print(
                    f"No noted version of {pptx_path.name} found in {paths['processed_dir']}"
                )
                print("Please run the transcript generation process first.")
                return

        print(f"Generating audio directly from {noted_pptx.name} notes...")
        print(f"Using {tts_provider} for TTS generation")

        # Save audio files to the same directory as the noted PPTX
        process_pptx_for_audio(noted_pptx, None, provider=tts_provider, page_range=slides)
        print("Pipeline completed successfully!")
        return

    # Generate transcripts
    if use_storm:
        print(
            f"Generating transcripts with Storm-enhanced approach for {pptx_path.name}..."
        )
        print(f"Language: {language}, Model: {model}, Provider: {llm_provider}")
        print(
            f"Web search for fact verification: {'DISABLED' if disable_search else 'ENABLED'}"
        )

        noted_pptx_path = process_presentation_with_storm(
            pptx_path=pptx_path,
            output_base_dir=paths["processed_dir"],
            target_language=language,
            model=model,
            enable_search=not disable_search,
            llm_provider=llm_provider,
            slides=slides,
        )
    else:
        print(f"Generating transcripts with standard approach for {pptx_path.name}...")
        print(f"Language: {language}, Model: {model}, Provider: {llm_provider}")
        noted_pptx_path = standard_process_presentation(
            pptx_path=pptx_path,
            output_base_dir=paths["processed_dir"],
            target_language=language,
            model=model,
            llm_provider=llm_provider,
            slides=slides,
        )

    # Generate audio if not skipped
    if not skip_audio:
        print("Generating audio from transcripts...")
        print(f"Using {tts_provider} for TTS generation")

        # Save audio files to the same directory as the noted PPTX
        process_pptx_for_audio(noted_pptx_path, None, provider=tts_provider)

    print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
