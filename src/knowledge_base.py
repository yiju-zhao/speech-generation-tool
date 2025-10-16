"""
Knowledge base management and retrieval for enhancing transcript generation with verified information.
"""

import json
import logging
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# Required libraries: pip install sentence-transformers faiss-cpu
import numpy as np
from sentence_transformers import SentenceTransformer
try:
    import faiss
except ImportError:
    raise ImportError("FAISS is not installed. Please run 'pip install faiss-cpu' or 'pip install faiss-gpu'.")

from src.web_search import TavilySearchRetriever


@dataclass
class KnowledgeItem:
    """A single knowledge item in the knowledge base."""
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
    """
    Manages a collection of knowledge items using a FAISS index for efficient, persistent search.
    """
    def __init__(self, knowledge_dir: str, model_name: str = "paraphrase-MiniLM-L6-v2"):
        """
        Initialize the knowledge base. If a pre-built index exists, it's loaded.
        Otherwise, a new index is created from the JSON files in the knowledge_dir.
        """
        self.knowledge_dir = Path(knowledge_dir)
        self.index_path = self.knowledge_dir / "knowledge.index"
        self.metadata_path = self.knowledge_dir / "knowledge_metadata.json"
        
        self.items: List[KnowledgeItem] = []
        self.index = None
        self.encoder = SentenceTransformer(model_name)

        if self.index_path.exists() and self.metadata_path.exists():
            logging.info(f"Loading existing knowledge base from {self.knowledge_dir}...")
            self._load_from_disk()
        elif self.knowledge_dir.exists():
            logging.warning(f"No existing index found. Building new knowledge base from {self.knowledge_dir}...")
            self._build_and_save_index()
        else:
            logging.error(f"Knowledge directory {self.knowledge_dir} does not exist. Knowledge base is empty.")

    def _load_from_disk(self):
        """Loads the FAISS index and metadata from disk."""
        try:
            self.index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, "r") as f:
                items_data = json.load(f)
                self.items = [KnowledgeItem.from_dict(data) for data in items_data]
            
            # Ensure the number of items matches the index size
            if self.index.ntotal != len(self.items):
                logging.error("Index and metadata mismatch. Rebuilding is recommended.")
                # You could force a rebuild here if desired
            else:
                logging.info(f"Successfully loaded {len(self.items)} items into the knowledge base.")
        except Exception as e:
            logging.error(f"Failed to load knowledge base from disk: {e}. It may be corrupted.")

    def _build_and_save_index(self):
        """Loads data from JSONs, builds a FAISS index, and saves it to disk."""
        # Step 1: Load all knowledge items from JSON files
        source_items = self._load_source_documents()
        if not source_items:
            logging.warning("No source documents found to build the knowledge base.")
            return

        self.items = source_items
        contents = [item.content for item in self.items]

        # Step 2: Compute embeddings
        logging.info(f"Computing embeddings for {len(contents)} documents...")
        embeddings = self.encoder.encode(contents, show_progress_bar=True)
        embedding_dim = embeddings.shape[1]

        # Step 3: Create and populate the FAISS index
        self.index = faiss.IndexFlatL2(embedding_dim) # Using a simple L2 distance index
        self.index.add(np.array(embeddings, dtype=np.float32))

        # Step 4: Save the index and metadata to disk
        try:
            faiss.write_index(self.index, str(self.index_path))
            with open(self.metadata_path, "w") as f:
                json.dump([item.to_dict() for item in self.items], f)
            logging.info(f"Successfully built and saved new knowledge base to {self.knowledge_dir}")
        except Exception as e:
            logging.error(f"Failed to save new knowledge base: {e}")

    def _load_source_documents(self) -> List[KnowledgeItem]:
        """Scans the directory for JSON files and loads them into KnowledgeItem objects."""
        all_items = []
        for file_path in self.knowledge_dir.glob("*.json"):
            # Exclude the metadata file we save
            if file_path.name == "knowledge_metadata.json":
                continue
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for item_data in data:
                        all_items.append(KnowledgeItem.from_dict(item_data))
                elif isinstance(data, dict) and "items" in data:
                    for item_data in data["items"]:
                        all_items.append(KnowledgeItem.from_dict(item_data))
            except Exception as e:
                logging.error(f"Error loading source knowledge file {file_path}: {e}")
        return all_items

    def search(self, query: str, top_k: int = 5) -> List[KnowledgeItem]:
        """Search the knowledge base with a natural language query using the FAISS index."""
        if not self.index or len(self.items) == 0:
            logging.warning("Knowledge base is not initialized or is empty. Cannot search.")
            return []

        query_embedding = self.encoder.encode([query])
        distances, indices = self.index.search(np.array(query_embedding, dtype=np.float32), top_k)
        
        # Filter out invalid indices if any
        valid_indices = [i for i in indices[0] if i != -1]
        return [self.items[i] for i in valid_indices]

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
        if self.knowledge_base and len(self.knowledge_base) > 0:
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
            
        # # Sort results to prioritize non-placeholders
        # sorted_results = sorted(
        #     results, 
        #     key=lambda x: 0 if x.source != "slide_content" else 1
        # )
        
        return results
