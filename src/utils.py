"""
Utility functions for the Digital Presenter project.
"""

import os
import json
import toml
from pathlib import Path
from dotenv import load_dotenv
from pptx import Presentation


# Load environment variables
def load_config():
    """Load configuration from config.toml file."""
    config_path = Path(__file__).parent.parent / "config" / "config.toml"
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found at {config_path}. "
            "Please create a config.toml file in the config directory."
        )
    
    try:
        config = toml.load(config_path)
        required_keys = ["openai_api_key", "minimax_api_key", "deepseek_api_key"]
        
        # Validate that all required keys are present
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise KeyError(
                f"Missing required configuration keys: {', '.join(missing_keys)}. "
                "Please check your config.toml file."
            )
        
        return config
    except toml.TomlDecodeError as e:
        raise ValueError(f"Error parsing config.toml: {str(e)}")


def ensure_directory(directory):
    """Ensure a directory exists, create it if it doesn't."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    return directory


def get_project_paths():
    """Get standard project paths."""
    base_dir = Path(__file__).parent.parent
    return {
        "base_dir": base_dir,
        "raw_dir": base_dir / "data" / "raw",
        "noted_dir": base_dir / "data" / "noted",
        "audio_dir": base_dir / "data" / "audio",
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


def extract_transcripts_from_pptx(pptx_path):
    """
    Extract transcripts from the notes section of each slide in a PowerPoint file.
    
    Args:
        pptx_path (Path or str): Path to the PowerPoint file
        
    Returns:
        list: A list of dictionaries containing slide number and transcript
    """
    pptx_path = Path(pptx_path)
    
    # Load the presentation
    presentation = Presentation(pptx_path)
    
    # Extract transcripts from notes
    transcripts = []
    for i, slide in enumerate(presentation.slides, 1):
        # Get the content from the slide
        slide_content = []
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                slide_content.append(shape.text.strip())
        
        original_content = "\n".join(slide_content) if slide_content else ""
        
        # Get transcript from notes
        transcript = ""
        if slide.has_notes_slide:
            notes_text = slide.notes_slide.notes_text_frame.text
            if notes_text.strip():
                transcript = notes_text.strip()
        
        slide_data = {
            "slide_number": i,
            "original_content": original_content,
            "transcript": transcript
        }
        
        transcripts.append(slide_data)
    
    return transcripts


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
