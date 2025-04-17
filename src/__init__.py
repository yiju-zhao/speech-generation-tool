"""
Digital Presenter - A tool for generating presentation transcripts and audio narration.
"""

__version__ = "0.1.0"

# Export the main functionality for easy imports
from .transcript_processor import process_presentation_with_storm
from .transcript_models import SlideInformation
from .transcript_generator import TranscriptGenerator, TranscriptReviewer
from .transcript_curator import TranscriptKnowledgeCurator
