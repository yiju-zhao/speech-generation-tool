"""
Knowledge base management and retrieval for enhancing transcript generation with verified information.
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore
    cosine_similarity = None  # type: ignore

from src.web_search import TavilySearchRetriever


@dataclass
class KnowledgeItem:
    """A single knowledge item in the knowledge base."""
    content: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embeddings: Optional[List[float]] = None

    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            "content": self.content,
            "source": self.source,
            "metadata": self.metadata,
            # Don't store embeddings in JSON
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
    """Manages a collection of knowledge items and provides retrieval functionality."""

    def __init__(self, knowledge_dir=None):
        """Initialize the knowledge base."""
        self.items = []
        self.encoder = None
        self.embeddings = None

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
        # Invalidate cached embeddings when adding new items
        self.embeddings = None

    def add_items(self, items):
        """Add multiple knowledge items to the base."""
        self.items.extend(items)
        # Invalidate cached embeddings when adding new items
        self.embeddings = None

    def init_encoder(self):
        """Initialize the sentence encoder for semantic search."""

        if self.encoder is None:
            if SentenceTransformer is None:
                logging.warning("sentence-transformers not available; semantic search disabled. Falling back to keyword search.")
                self.encoder = None
                return
            try:
                self.encoder = SentenceTransformer("paraphrase-MiniLM-L6-v2")
                logging.info("Sentence encoder initialized successfully.")
            except Exception as e:
                logging.warning(f"Error initializing sentence encoder; falling back to keyword search: {e}")
                self.encoder = None

    def compute_embeddings(self):
        """Compute embeddings for all knowledge items."""
        if not self.items:
            logging.warning("No knowledge items to encode.")
            return

        self.init_encoder()
        if not self.encoder:
            # No encoder — skip embedding computation
            self.embeddings = None
            return

        contents = [item.content for item in self.items]
        self.embeddings = self.encoder.encode(contents)

        # Store embeddings in each item
        for i, item in enumerate(self.items):
            try:
                item.embeddings = self.embeddings[i].tolist()
            except Exception:
                item.embeddings = None

    def search(self, query, top_k=5):
        """Search the knowledge base with a natural language query."""
        if not self.items:
            logging.warning("No knowledge items to search.")
            return []

        # Compute embeddings if not already done
        if self.embeddings is None:
            self.compute_embeddings()

        # If encoder or embeddings are unavailable, fall back to keyword search
        if not self.encoder or self.embeddings is None or cosine_similarity is None:
            logging.info("Falling back to keyword search for knowledge base query.")
            return self.keyword_search(query, top_k=top_k)

        query_embedding = self.encoder.encode(query)

        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]

        return [self.items[i] for i in top_indices]

    def keyword_search(self, keywords, top_k=5):
        """Simple keyword-based search."""
        if not self.items:
            return []

        # Split the keywords string into individual keywords
        if isinstance(keywords, str):
            keywords = [k.strip().lower() for k in keywords.split() if k.strip()]

        results = []
        for item in self.items:
            content_lower = item.content.lower()
            score = sum(1 for keyword in keywords if keyword in content_lower)
            if score > 0:
                results.append((score, item))

        # Sort by score (descending) and return top_k items
        results.sort(reverse=True, key=lambda x: x[0])
        return [item for _, item in results[:top_k]]

    def save_to_file(self, file_path, indent=None):
        """Save the knowledge base to a JSON file."""
        items_data = [item.to_dict() for item in self.items]
        with open(file_path, "w") as f:
            json.dump({"items": items_data}, f, indent=indent)

    def __len__(self):
        return len(self.items)


class KnowledgeRetriever:
    """Class for retrieving relevant knowledge for transcript generation."""

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
        """Retrieve relevant knowledge for a slide."""
        results = []
        logging.info(f"Starting knowledge retrieval for slide {slide_number}")

        # First, try retrieving from the local knowledge base
        kb_results = []
        if len(self.knowledge_base) > 0:
            logging.info(f"Retrieving from knowledge base for slide {slide_number} ({len(self.knowledge_base)} items in KB)")
            try:
                kb_results = self.knowledge_base.search(slide_content, top_k=max_items)
                if kb_results:
                    logging.info(f"Retrieved {len(kb_results)} items from knowledge base")
                else:
                    logging.info("No matching items found in knowledge base")
            except Exception as e:
                logging.error(f"Error searching knowledge base: {e}")
            
            results.extend(kb_results)

        # If web search is enabled and retriever is available, fetch additional information
        web_results = []
        if (self.web_search_enabled and self.web_retriever and 
            self.web_retriever.is_available and queries):
            logging.info(f"Web search enabled for slide {slide_number}")
            try:
                # Filter out empty queries
                valid_queries = [q for q in queries if q and len(q.strip()) > 0]
                if not valid_queries:
                    logging.warning("No valid queries for web search")
                else:
                    logging.info(f"Performing web search with {len(valid_queries)} queries")
                    
                    # Try individual queries first for more targeted results
                    for query in valid_queries[:2]:  # Try first few queries individually
                        query_results = self.web_retriever.search(query)
                        if query_results:
                            for result in query_results:
                                result["source_query"] = query
                            web_results.extend(query_results)
                    
                    # If individual queries didn't yield results, try combined search
                    if not web_results:
                        web_results = self.web_retriever.search_multiple(valid_queries)
                    
                    if not web_results:
                        # Try fallback strategies if web search failed
                        error = getattr(self.web_retriever, 'get_last_error', lambda: None)()
                        if error:
                            logging.error(f"Web search failed with error: {error}")
                        else:
                            logging.warning("Web search returned no results")
                        
                        # Try fallback queries if we have slide content
                        if slide_content:
                            fallback_strategies = [
                                # First sentence only
                                slide_content.split(".", 1)[0] if "." in slide_content else slide_content[:60],
                                # Extract key terms
                                " ".join([w for w in slide_content.replace("\n", " ").split() if len(w) > 4][:8]),
                                # Title-like query
                                f"information about {' '.join(slide_content.split()[:3])}"
                            ]
                            
                            for strategy_idx, simplified_query in enumerate(fallback_strategies):
                                if not web_results:
                                    fallback_results = self.web_retriever.search(simplified_query)
                                    if fallback_results:
                                        web_results = fallback_results
                                        # Mark these as from fallback strategy
                                        for result in web_results:
                                            result["source_query"] = f"fallback_{strategy_idx+1}:{simplified_query}"
                                        break

                # Process web results and convert to knowledge items
                for result in web_results[:max_items]:
                    content = result.get("content", "") or result.get("raw_content", "")
                    url = result.get("url", "Unknown")
                    is_ai_answer = result.get("is_ai_answer", False)
                    source_query = result.get("source_query", "")
                    
                    if content:
                        # Create metadata
                        metadata = {
                            "title": result.get("title", ""),
                            "source_query": source_query,
                            "retrieval_method": "web_search",
                            "slide_number": slide_number
                        }
                        
                        if is_ai_answer:
                            # For AI-generated answers
                            metadata["is_ai_answer"] = True
                            metadata["retrieval_method"] = "tavily_ai_answer"
                            
                            # Create knowledge item for AI answer
                            ai_item = KnowledgeItem(
                                content=content,
                                source="Tavily AI Answer",
                                metadata=metadata
                            )
                            
                            # Add to results with high priority
                            if ai_item not in results:
                                results.insert(0, ai_item)
                            
                            # Add to knowledge base
                            try:
                                self.knowledge_base.add_item(ai_item)
                            except Exception as e:
                                logging.error(f"Failed to add AI answer to knowledge base: {e}")
                        else:
                            # Create regular knowledge item
                            item = KnowledgeItem(
                                content=content,
                                source=url,
                                metadata=metadata
                            )
                            
                            # Add to results and knowledge base
                            results.append(item)
                            try:
                                self.knowledge_base.add_item(item)
                            except Exception as e:
                                logging.error(f"Failed to add web result to knowledge base: {e}")
            
            except Exception as e:
                logging.error(f"Error retrieving from web search: {e}")

        # If we still have no results, create a placeholder knowledge item
        if not results and slide_content:
            logging.warning(f"No knowledge items found for slide {slide_number}. Creating placeholder.")
            placeholder_item = KnowledgeItem(
                content=f"No external knowledge found for slide {slide_number}. Using slide content only: {slide_content}",
                source="slide_content",
                metadata={
                    "title": f"Slide {slide_number} Content",
                    "retrieval_method": "fallback",
                    "slide_number": slide_number
                }
            )
            results.append(placeholder_item)
            
        # Sort results to prioritize non-placeholders
        sorted_results = sorted(
            results, 
            key=lambda x: 0 if x.source != "slide_content" else 1
        )
        
        return sorted_results[:max_items]  # Limit to max_items
