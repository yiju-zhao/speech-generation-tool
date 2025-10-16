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
    knowledge_items: List[KnowledgeItem] = field(default_factory=list)
    search_results: List[Dict] = field(default_factory=list)
