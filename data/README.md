# Data Directory Structure

This directory contains all the data files used by the Digital Presenter project.

## Directory Structure

- `raw/` - Original PowerPoint files before processing
- `noted/` - PowerPoint files with transcripts in the notes section
- `audio/` - Audio files generated from transcripts (though audio files are now saved in the same directory as the noted PPTX)

## Workflow

1. Place original PowerPoint files in the `raw/` directory
2. Run the transcript generation process to create PowerPoint files with transcripts in notes in `noted/`
3. Run the voice generation process to create audio files in the same directory as the noted PPTX files

## File Naming Conventions

- Original PowerPoint files: `filename.pptx`
- PowerPoint files with transcripts: `filename_noted.pptx`
- Audio files: `slide_X.mp3` (where X is the slide number) 