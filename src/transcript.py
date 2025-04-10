"""
Transcript generation module for Digital Presenter.
"""

import os
from pathlib import Path
import openai
from pptx import Presentation

from .utils import load_config, ensure_directory, get_project_paths, save_json


def extract_slide_content(slide):
    """Extract all text content from a slide."""
    slide_content = []
    for shape in slide.shapes:
        if hasattr(shape, "text") and shape.text.strip():
            slide_content.append(shape.text.strip())

    return "\n".join(slide_content) if slide_content else None


def generate_transcript(slide_content, previous_transcript=None, llm_client=None):
    """Generate a transcript for a slide using LLM."""
    if not llm_client:
        config = load_config()
        llm_client = openai.OpenAI(api_key=config["openai_api_key"])

    # Add previous slide's transcript as context if available
    context = ""
    if previous_transcript:
        context = f"\nPrevious slide's transcript:\n{previous_transcript}\n"

    # Generate transcript using LLM
    prompt = f"""Please generate a natural, conversational transcript for the following presentation slide content. 
    Make it sound like someone giving a presentation, with proper transitions and explanations.
    {context}
    Current slide content:
    {slide_content}
    
    Transcript:"""

    response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    )

    return response.choices[0].message.content.strip()


def process_presentation(pptx_path, output_file=None):
    """Process a presentation and generate transcripts for all slides."""
    # Load the presentation
    presentation = Presentation(pptx_path)

    # Initialize OpenAI client
    config = load_config()
    llm_client = openai.OpenAI(api_key=config["openai_api_key"])

    # Process each slide
    transcripts = []
    for idx, slide in enumerate(presentation.slides, 1):
        # Extract all text from the slide
        slide_content = extract_slide_content(slide)

        if not slide_content:
            continue

        # Get previous transcript if available
        previous_transcript = transcripts[-1]["transcript"] if transcripts else None

        # Generate transcript
        transcript = generate_transcript(slide_content, previous_transcript, llm_client)

        # Store results
        slide_data = {
            "slide_number": idx,
            "original_content": slide_content,
            "transcript": transcript,
        }
        transcripts.append(slide_data)

        print(f"Processed slide {idx}")

    # Save transcripts to a JSON file
    if not output_file:
        paths = get_project_paths()
        ensure_directory(paths["output_dir"])
        output_file = paths["output_dir"] / "presentation_transcripts.json"

    save_json(transcripts, output_file)
    print(f"\nTranscripts have been saved to {output_file}")

    return transcripts


def main():
    """Main function to run the transcript generator."""
    paths = get_project_paths()
    ensure_directory(paths["input_dir"])
    ensure_directory(paths["output_dir"])

    # Find PowerPoint files in the input directory
    pptx_files = list(paths["input_dir"].glob("*.pptx"))

    if not pptx_files:
        print(f"No PowerPoint files found in {paths['input_dir']}")
        return

    # Process each PowerPoint file
    for pptx_file in pptx_files:
        print(f"Processing {pptx_file.name}...")
        output_file = paths["output_dir"] / f"{pptx_file.stem}_transcripts.json"
        process_presentation(pptx_file, output_file)


if __name__ == "__main__":
    main()
