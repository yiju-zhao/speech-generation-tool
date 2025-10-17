"""
Knowledge base management for storing Q&A items from web search.
Simplified to use direct JSON storage without embeddings or semantic retrieval.
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any

from src.web_search import TavilySearchRetriever


@dataclass
class KnowledgeItem:
    """A single knowledge item (Q&A pair) in the knowledge base."""
    content: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            "content": self.content,
            "source": self.source,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data):
        """Create from dictionary."""
        return cls(
            content=data["content"],
            source=data["source"],
            metadata=data.get("metadata", {}),
        )


class KnowledgeBase:
    """Simple storage for Q&A knowledge items without embedding/retrieval."""

    def __init__(self, knowledge_dir=None):
        """Initialize the knowledge base."""
        self.items = []

        if knowledge_dir:
            self.load_from_directory(knowledge_dir)

    def load_from_directory(self, directory_path):
        """Load knowledge items from all JSON files in a directory."""
        directory = Path(directory_path)
        if not directory.exists():
            logging.warning(f"Knowledge directory {directory} does not exist.")
            return

        count = 0
        for file_path in directory.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)

                # Handle different possible JSON structures
                if isinstance(data, list):
                    # List of knowledge items
                    for item_data in data:
                        self.add_item(KnowledgeItem.from_dict(item_data))
                    count += len(data)
                elif isinstance(data, dict) and "items" in data:
                    # Dict with "items" key containing a list of knowledge items
                    for item_data in data["items"]:
                        self.add_item(KnowledgeItem.from_dict(item_data))
                    count += len(data["items"])
                elif "content" in data and "source" in data:
                    # Single knowledge item
                    self.add_item(KnowledgeItem.from_dict(data))
                    count += 1
            except Exception as e:
                logging.error(f"Error loading knowledge file {file_path}: {e}")

        logging.info(f"Loaded {count} knowledge items from {directory}")

    def add_item(self, item):
        """Add a knowledge item to the base."""
        self.items.append(item)

    def add_items(self, items):
        """Add multiple knowledge items to the base."""
        self.items.extend(items)

    def get_all_items(self):
        """Get all knowledge items."""
        return self.items

    def save_to_file(self, file_path, indent=2):
        """Save the knowledge base to a JSON file."""
        items_data = [item.to_dict() for item in self.items]
        with open(file_path, "w") as f:
            json.dump({"items": items_data}, f, indent=indent)
        logging.info(f"Saved {len(items_data)} knowledge items to {file_path}")

    def __len__(self):
        return len(self.items)


class KnowledgeRetriever:
    """Simple retriever that provides direct access to pre-curated Q&A knowledge items."""

    def __init__(self, knowledge_base_dir=None, web_search_enabled=True, tavily_api_key=None):
        self.knowledge_base = KnowledgeBase(knowledge_base_dir)
        self.web_search_enabled = web_search_enabled
        self.web_retriever = None

        if web_search_enabled:
            try:
                self.web_retriever = TavilySearchRetriever(api_key=tavily_api_key)
                if self.web_retriever.is_available:
                    logging.info("Web search retriever initialized successfully.")
                else:
                    logging.warning("Web search retriever not available. Will use knowledge base only.")
                    self.web_search_enabled = False
            except (ImportError, ValueError) as e:
                logging.warning(f"Web search retriever initialization failed: {e}")
                self.web_search_enabled = False

    def retrieve_for_slide(self, slide_content, slide_number, queries=None, max_items=5):
        """
        Retrieve Q&A knowledge for a slide.
        Returns items directly from the pre-loaded knowledge base without complex retrieval.
        """
        results = []
        logging.info(f"Retrieving knowledge for slide {slide_number}")

        # Simply return all available knowledge items from the base (limited set)
        if len(self.knowledge_base) > 0:
            logging.info(f"Found {len(self.knowledge_base)} items in knowledge base")
            # Return all items since it's a curated limited set
            results = self.knowledge_base.get_all_items()[:max_items]
            return results

        # If knowledge base is empty and web search is enabled, fetch from web
        if self.web_search_enabled and self.web_retriever and self.web_retriever.is_available and queries:
            logging.info(f"Fetching Q&A from web search for slide {slide_number}")
            try:
                valid_queries = [q for q in queries if q and len(q.strip()) > 0]
                if valid_queries:
                    # Get web search results
                    web_results = self.web_retriever.search_multiple(valid_queries[:3])

                    # Convert to knowledge items
                    for result in web_results[:max_items]:
                        content = result.get("content", "") or result.get("raw_content", "")
                        if content:
                            metadata = {
                                "title": result.get("title", ""),
                                "source_query": result.get("source_query", ""),
                                "slide_number": slide_number
                            }

                            # Check if it's a Q&A format answer
                            if result.get("is_ai_answer"):
                                metadata["qa"] = {
                                    "question": result.get("source_query", ""),
                                    "answer": content
                                }

                            item = KnowledgeItem(
                                content=content,
                                source=result.get("url", "Web Search"),
                                metadata=metadata
                            )
                            results.append(item)

                            # Add to knowledge base for future use
                            self.knowledge_base.add_item(item)

            except Exception as e:
                logging.error(f"Error fetching from web search: {e}")

        return results[:max_items]
