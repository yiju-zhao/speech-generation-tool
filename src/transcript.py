"""
Transcript generation module for Digital Presenter.
"""

import openai
from pptx import Presentation

from .utils import (
    load_config,
    ensure_directory,
    get_project_paths,
    save_json,
    write_transcripts_to_pptx
)


def extract_slide_content(slide):
    """Extract all text content from a slide."""
    slide_content = []
    for shape in slide.shapes:
        if hasattr(shape, "text") and shape.text.strip():
            slide_content.append(shape.text.strip())

    return "\n".join(slide_content) if slide_content else None


def generate_transcript(slide_content, previous_transcripts=None, llm_client=None, target_language="chinese"):
    """Generate a transcript for a slide using LLM."""
    # Add all previous slides' transcripts as context if available
    context = ""
    if previous_transcripts:
        context = f"{previous_transcripts}\n"

    # Create a single prompt that specifies the output language
    unified_prompt = f"""
    Please generate a natural, conversational transcript for the following presentation slide content. Make it sound like someone giving a presentation, with proper transitions and explanations. DO NOT include any additional text or comments.

    OUTPUT LANGUAGE: {target_language.upper()}
    ONLY output the transcript for speech content. NO instruction or content that can not convert to a voice.
    
    Current slide content:
    {slide_content}

    Please generate the transcript based on the current slide content to extend the previous slides' transcripts {context}
    """
    
    # Use the client that was passed in
    response = llm_client.chat.completions.create(
        model="deepseek-chat" if target_language == "chinese" else "gpt-4o-mini",
        messages=[
            {"role": "user", "content": unified_prompt}
        ],
        max_tokens=2000,
        stream=False
    )
    
    return response.choices[0].message.content.strip()


def process_presentation(pptx_path, output_base_dir=None, target_language="chinese"):
    """Process a presentation and generate transcripts for all slides."""
    # Load the presentation
    presentation = Presentation(pptx_path)
    
    # Set up output directory structure
    if not output_base_dir:
        paths = get_project_paths()
        output_base_dir = paths["output_dir"] / pptx_path.stem
    ensure_directory(output_base_dir)

    # Initialize the appropriate LLM client based on target language
    config = load_config()
    if target_language == "chinese":
        # Use DeepSeek for Chinese
        llm_client = openai.OpenAI(
            api_key=config["deepseek_api_key"], 
            base_url="https://api.deepseek.com"
        )
    else:
        # Use OpenAI for English
        llm_client = openai.OpenAI(api_key=config["openai_api_key"])

    # Process each slide
    transcripts = []
    previous_transcripts = []
    for idx, slide in enumerate(presentation.slides, 1):
        # Extract all text from the slide
        slide_content = extract_slide_content(slide)

        if not slide_content:
            continue

        # Generate transcript
        transcript = generate_transcript(slide_content, previous_transcripts, llm_client, target_language)

        # Store results
        slide_data = {
            "slide_number": idx,
            "original_content": slide_content,
            "transcript": transcript,
        }
        transcripts.append(slide_data)
        previous_transcripts = previous_transcripts + [transcript]

        print(f"Processed slide {idx}")

    # Save complete transcripts to a JSON file
    complete_transcript_file = output_base_dir / "transcript.json"
    save_json(transcripts, complete_transcript_file)
    print(f"\nTranscript has been saved to {complete_transcript_file}")
    
    # Write transcripts back to the PPTX file
    noted_pptx_path = write_transcripts_to_pptx(pptx_path, transcripts, output_base_dir)
    print(f"PPTX with transcripts has been saved to {noted_pptx_path}")

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
        output_dir = paths["output_dir"] / pptx_file.stem
        process_presentation(pptx_file, output_dir)


if __name__ == "__main__":
    main()
