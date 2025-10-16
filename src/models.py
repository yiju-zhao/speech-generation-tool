from dataclasses import dataclass, field
from typing import List, Dict

from .knowledge_base import KnowledgeItem


@dataclass
class SlideInformation:
    """Store information about a slide to prevent hallucinations."""

    queries: List[str] = field(default_factory=list)
    facts: List[str] = field(default_factory=list)
    original_content: str = ""
    unpolished_notes: str = ""
    slide_number: int = 0
    # Slide placement and type help guide generation/review
    position: str = ""   # one of: first, middle, last
    type: str = ""        # one of: title, content, thank_you
    knowledge_items: List[KnowledgeItem] = field(default_factory=list)
    search_results: List[Dict] = field(default_factory=list)
