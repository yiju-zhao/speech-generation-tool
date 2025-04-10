"""
Utility functions for the Digital Presenter project.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from pptx import Presentation


# Load environment variables
def load_config():
    """Load environment variables from .env file."""
    env_path = Path(__file__).parent.parent / "config" / ".env"
    load_dotenv(env_path)

    return {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "minimax_api_key": os.getenv("MINIMAX_API_KEY"),
        "deepseek_api_key": os.getenv("DEEPSEEK_API_KEY"),
    }


def ensure_directory(directory):
    """Ensure a directory exists, create it if it doesn't."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    return directory


def get_project_paths():
    """Get standard project paths."""
    base_dir = Path(__file__).parent.parent
    return {
        "base_dir": base_dir,
        "input_dir": base_dir / "data" / "input",
        "output_dir": base_dir / "data" / "output",
        "config_dir": base_dir / "config",
        "notebooks_dir": base_dir / "notebooks",
    }


def save_json(data, filename, directory=None):
    """Save data to a JSON file."""
    if directory:
        ensure_directory(directory)
        filepath = Path(directory) / filename
    else:
        filepath = Path(filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return filepath


def load_json(filepath):
    """Load data from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def write_transcripts_to_pptx(pptx_path, transcripts, output_dir=None):
    """
    Write transcripts back to the notes of each slide in the PPTX file.
    
    Args:
        pptx_path (Path or str): Path to the original PPTX file
        transcripts (list): List of dictionaries containing slide data with transcripts
        output_dir (Path or str, optional): Directory to save the output file. 
                                           If None, uses the same directory as the input file.
    
    Returns:
        Path: Path to the saved PPTX file with transcripts in notes
    """
    pptx_path = Path(pptx_path)
    
    # Load the presentation
    presentation = Presentation(pptx_path)
    
    # Set up output directory
    if output_dir:
        output_dir = Path(output_dir)
        ensure_directory(output_dir)
    else:
        output_dir = pptx_path.parent
    
    # Create output filename with "_noted" suffix
    output_filename = f"{pptx_path.stem}_noted{pptx_path.suffix}"
    output_path = output_dir / output_filename
    
    # Write transcripts to slide notes
    for i, slide in enumerate(presentation.slides):
        # Skip if no transcript for this slide
        if i >= len(transcripts):
            continue
            
        # Get the transcript for this slide
        transcript = transcripts[i].get("transcript", "")
        
        # Get or create notes slide - python-pptx creates it automatically if it doesn't exist
        notes_slide = slide.notes_slide
        
        # Add the transcript to notes
        text_frame = notes_slide.notes_text_frame
        text_frame.text = transcript
    
    # Save the presentation
    presentation.save(output_path)
    
    return output_path
