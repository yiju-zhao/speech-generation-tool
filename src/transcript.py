"""
Transcript generation module for Digital Presenter.
"""

import openai
from pptx import Presentation

from .utils import (
    load_config,
    ensure_directory,
    get_project_paths,
    write_transcripts_to_pptx
)


def extract_slide_content(slide):
    """Extract all text content from a slide."""
    slide_content = []
    
    # Process all shapes in the slide
    for shape in slide.shapes:
        # Handle direct text attributes
        if hasattr(shape, "text") and shape.text.strip():
            slide_content.append(shape.text.strip())
        
        # Handle tables
        if shape.has_table:
            table = shape.table
            for row in table.rows:
                row_text = []
                for cell in row.cells:
                    if cell.text.strip():
                        row_text.append(cell.text.strip())
                if row_text:
                    slide_content.append(" | ".join(row_text))
        
        # Handle text frames (which might contain paragraphs with runs)
        if hasattr(shape, "text_frame"):
            text_frame = shape.text_frame
            for paragraph in text_frame.paragraphs:
                paragraph_text = paragraph.text.strip()
                if paragraph_text:
                    slide_content.append(paragraph_text)
    
    return "\n".join(slide_content) if slide_content else None


def generate_transcript(slide_content, previous_transcripts=None, llm_client=None, target_language="chinese"):
    """Generate a transcript for a slide using LLM."""
    # Add all previous slides' transcripts as context if available
    context = ""
    if previous_transcripts:
        context = f"{previous_transcripts}\n"

    # Check if the slide_content contains both slide content and unpolished notes
    has_unpolished_notes = "UNPOLISHED NOTES" in slide_content
    
    # Create a prompt that specifies the output language and provides clear instructions
    if has_unpolished_notes:
        unified_prompt = f"""
        You are a professional presenter.

        OUTPUT LANGUAGE: {target_language.upper()}

        DO NOT include any additional text or comments.
        
        Use the unpolished notes as a guide to understand the intended message, but incorporate information from the slide content as well. Create a coherent, flowing transcript that combines both sources.
        
        Current slide content:
        {slide_content}
        
        Previous slides' transcripts:
        {context}

        Please generate the transcript for this slide based on the above content and make a natural and smooth transition from the previous slides' transcripts:
        """
    else:
        unified_prompt = f"""
        You are a professional presenter.

        Make it sound like someone giving a presentation, with proper transitions and explanations. 

        DO NOT include any additional text or comments.

        OUTPUT LANGUAGE: {target_language.upper()}
        ONLY output the transcript for speech content. NO instruction or content that can not convert to a voice.
        
        Current slide content:
        {slide_content}

        Previous slides' transcripts:
        {context}
        
        Please generate the transcript for this slide based on the current slide content and make a natural and smooth transition from the previous slides' transcripts:
        """
    
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
        output_base_dir = paths["noted_dir"]
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
        
        # Check for unpolished notes in the slide
        unpolished_notes = ""
        if slide.has_notes_slide:
            notes_text = slide.notes_slide.notes_text_frame.text
            if notes_text.strip():
                unpolished_notes = notes_text.strip()
                print(f"Found unpolished notes for slide {idx}")

        if not slide_content and not unpolished_notes:
            print(f"No content found for slide {idx}, skipping.")
            continue

        # Generate transcript
        # If we have unpolished notes, use them as a guide along with slide content
        if unpolished_notes:
            # Create a prompt that includes both unpolished notes and slide content
            combined_content = f"""
            SLIDE CONTENT:
            {slide_content if slide_content else "No slide content available"}
            
            UNPOLISHED NOTES (USE AS GUIDE):
            {unpolished_notes}
            """
            transcript = generate_transcript(combined_content, previous_transcripts, llm_client, target_language)
        else:
            # If no unpolished notes, just use slide content
            transcript = generate_transcript(slide_content, previous_transcripts, llm_client, target_language)

        # Store results
        slide_data = {
            "slide_number": idx,
            "original_content": slide_content,
            "unpolished_notes": unpolished_notes,
            "transcript": transcript,
        }
        transcripts.append(slide_data)
        previous_transcripts = previous_transcripts + [transcript]

        print(f"Processed slide {idx}")

    # Write transcripts back to the PPTX file
    noted_pptx_path = write_transcripts_to_pptx(pptx_path, transcripts, output_base_dir)
    print(f"PPTX with transcripts has been saved to {noted_pptx_path}")

    return noted_pptx_path


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
        print("This will extract content from slides and any unpolished notes, then generate transcripts.")
        print("The transcripts will be saved to the notes section of a new PPTX file in the output directory.")
        
        output_dir = paths["output_dir"] / pptx_file.stem
        process_presentation(pptx_file, output_dir)
        
        print(f"Transcript generation complete for {pptx_file.name}")
        print(f"Check the output directory for the PPTX file with transcripts in the notes section.")


if __name__ == "__main__":
    main()
