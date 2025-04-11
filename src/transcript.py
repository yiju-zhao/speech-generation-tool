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
from .llm import get_llm_provider


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


def generate_transcript(slide_content, previous_transcripts=None, llm_client=None, target_language=None, model=None):
    """Generate a transcript for a slide using LLM."""
    # Add all previous slides' transcripts as context if available
    context = ""
    if previous_transcripts:
        context = f"{previous_transcripts}\n"
    
    # Create a unified prompt that handles both cases
    prompt = f"""
    **Role**: Technical Speech Architect  
    **Objective**: Create TTS-ready transcripts with uncompromised accuracy  

    **Core Directives**:  
    1. Preserve _exact_ terminology + numerical precision  
    2. Optimize complex concepts for vocal clarity  
    3. Maintain academic rigor in spoken format  

    **Input Sources**:  
    - Content: {slide_content}  
    - Context: {context}  

    **Output Specifications**:  
    `Language`: {target_language.upper() if target_language else "ENGLISH"}  
    `Format`: Pure speech text (no markup/annotations)  

    **Technical Protocols**:  
    ✓ Term fidelity: Use source terms verbatim  
    ✓ Data integrity: Exact values + full SI units  
    ✓ First-use acronyms: "Advanced Reactor Concept (ARC)"  
    ✓ Concept links: "This correlates directly with <previous_term>"  
    ✓ Transition logic: "Consequently...", "The evidence establishes..."  

    **Prohibited Elements**:  
    - Approximations ("~", "about")  
    - Subjective commentary ("surprisingly")  
    - Non-verbal cues (parentheticals, pauses)  
    - Casual paraphrasing  

    **Structural Guidelines**:  
    1. **Opening**: Bridge from {context} using exact technical references  
    2. **Flow**:  
    - State quantitative relationships verbally  
    - Use 3-tier vocal hierarchy: Concept → Mechanism → Evidence  
    - Employ precision transitions: "The critical progression..."  
    3. **Complex Data**:  
    - "Three components emerge: (1) <term>, (2) <term>..."  
    - "Sequential analysis reveals: First... Second... Crucially..."  

    **Safety Imperatives**:  
    ‼ Preserve safety-critical wording verbatim  
    ‼ Verbalize technical caveats explicitly  
    ‼ If term conflict: Prioritize slide terminology       
    """
    
    return llm_client.generate(prompt, model)


def process_presentation(pptx_path, output_base_dir=None, target_language=None, model=None):
    """Process a presentation and generate transcripts for all slides."""
    # Load the presentation
    presentation = Presentation(pptx_path)
    
    # Set up output directory structure
    if not output_base_dir:
        paths = get_project_paths()
        output_base_dir = paths["noted_dir"]
    ensure_directory(output_base_dir)

    # Initialize the appropriate LLM client based on model choice
    config = load_config()
    llm_client = get_llm_provider(model, config)

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

        # Combine slide content and unpolished notes if available
        combined_content = []
        if slide_content:
            combined_content.append(f"SLIDE CONTENT:\n{slide_content}")
        if unpolished_notes:
            combined_content.append(f"UNPOLISHED NOTES (USE AS GUIDE):\n{unpolished_notes}")
        
        # Generate transcript with combined content
        transcript = generate_transcript(
            "\n\n".join(combined_content),
            previous_transcripts,
            llm_client,
            target_language,
            model
        )

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
