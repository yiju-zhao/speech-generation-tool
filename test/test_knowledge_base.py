#!/usr/bin/env python
"""
Test script for the knowledge base functionality.
"""

import sys
import os
from pathlib import Path
import argparse
import logging

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from src.knowledge_base import KnowledgeBase, KnowledgeItem, KnowledgeRetriever


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def create_test_knowledge_base(output_dir):
    """Create a test knowledge base for demonstration purposes."""
    kb = KnowledgeBase()

    # Add some sample knowledge items
    items = [
        KnowledgeItem(
            content="Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to intelligence displayed by animals and humans. In the field of computer science, AI research is defined as the study of 'intelligent agents', which refers to any system that perceives its environment and takes actions that maximize the chances of achieving specified goals.",
            source="Sample AI Definition",
            metadata={"topic": "AI", "type": "definition"},
        ),
        KnowledgeItem(
            content="Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves.",
            source="Sample ML Definition",
            metadata={"topic": "Machine Learning", "type": "definition"},
        ),
        KnowledgeItem(
            content="Deep learning is a subset of machine learning where artificial neural networks, algorithms inspired by the human brain, learn from large amounts of data. Deep learning allows machines to solve complex problems even when using a data set that is very diverse, unstructured and inter-connected.",
            source="Sample DL Definition",
            metadata={"topic": "Deep Learning", "type": "definition"},
        ),
        KnowledgeItem(
            content="Transformer models are a type of neural network architecture that have been particularly successful for natural language processing tasks. They use self-attention mechanisms to process input sequences in parallel, allowing them to capture long-range dependencies and achieve state-of-the-art performance on many language tasks.",
            source="Sample Transformer Definition",
            metadata={"topic": "Transformers", "type": "definition"},
        ),
        KnowledgeItem(
            content="Reinforcement learning is an area of machine learning concerned with how intelligent agents ought to take actions in an environment in order to maximize the notion of cumulative reward. Reinforcement learning differs from supervised learning in that labeled input/output pairs need not be presented, and sub-optimal actions need not be explicitly corrected.",
            source="Sample RL Definition",
            metadata={"topic": "Reinforcement Learning", "type": "definition"},
        ),
    ]

    kb.add_items(items)

    # Save the knowledge base
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "test_knowledge_base.json")
    kb.save_to_file(output_file)

    logging.info(f"Created test knowledge base with {len(kb.items)} items")
    logging.info(f"Saved to {output_file}")

    return output_file


def test_knowledge_retrieval(knowledge_base_path, query=None):
    """Test retrieving information from the knowledge base."""
    # Initialize knowledge base
    kb = KnowledgeBase()
    kb.load_from_directory(knowledge_base_path)

    logging.info(f"Loaded knowledge base with {len(kb.items)} items")

    # Get query from user if not provided
    if not query:
        query = input("Enter a search query: ")

    # Test vector search if available
    try:
        logging.info(f"Searching for: {query}")
        results = kb.search(query, top_k=2)

        logging.info(f"Found {len(results)} results using vector search")
        for i, item in enumerate(results, 1):
            logging.info(f"\nResult {i}:")
            logging.info(f"Source: {item.source}")
            logging.info(f"Content: {item.content[:150]}...")
    except ImportError:
        logging.warning("Vector search not available. Using keyword search instead.")

        # Fall back to keyword search
        results = kb.keyword_search(query, top_k=2)

        logging.info(f"Found {len(results)} results using keyword search")
        for i, item in enumerate(results, 1):
            logging.info(f"\nResult {i}:")
            logging.info(f"Source: {item.source}")
            logging.info(f"Content: {item.content[:150]}...")


def test_knowledge_retriever(knowledge_base_dir, slide_content=None):
    """Test the KnowledgeRetriever class with a sample slide."""
    # Create a knowledge retriever
    retriever = KnowledgeRetriever(
        knowledge_base_dir=knowledge_base_dir, web_search_enabled=False
    )

    # Use sample slide content if not provided
    if not slide_content:
        slide_content = """
        Artificial Intelligence and Machine Learning
        
        - Overview of AI techniques
        - Deep learning applications
        - Transformer models in NLP
        - Future directions in AI research
        """

    # Sample queries
    queries = [
        "What is artificial intelligence?",
        "How do transformer models work?",
        "What is deep learning?",
    ]

    logging.info("Testing knowledge retriever with sample slide content")
    logging.info(f"Slide content:\n{slide_content}\n")

    # Retrieve knowledge items
    items = retriever.retrieve_for_slide(slide_content, 1, queries, max_items=3)

    logging.info(f"Retrieved {len(items)} knowledge items")
    for i, item in enumerate(items, 1):
        logging.info(f"\nItem {i}:")
        logging.info(f"Source: {item.source}")
        logging.info(f"Content: {item.content[:150]}...")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test knowledge base functionality")
    parser.add_argument(
        "--create", action="store_true", help="Create a test knowledge base"
    )
    parser.add_argument(
        "--search", action="store_true", help="Test knowledge base search"
    )
    parser.add_argument(
        "--retrieve", action="store_true", help="Test knowledge retriever"
    )
    parser.add_argument(
        "--kb-dir",
        type=str,
        default="data/knowledge_base",
        help="Knowledge base directory",
    )
    parser.add_argument("--query", type=str, help="Search query")
    args = parser.parse_args()

    setup_logging()

    if args.create:
        create_test_knowledge_base(args.kb_dir)

    if args.search:
        test_knowledge_retrieval(args.kb_dir, args.query)

    if args.retrieve:
        test_knowledge_retriever(args.kb_dir)

    # If no options specified, run all tests
    if not (args.create or args.search or args.retrieve):
        kb_path = create_test_knowledge_base(args.kb_dir)
        test_knowledge_retrieval(kb_path, "What is deep learning?")
        test_knowledge_retriever(args.kb_dir)


if __name__ == "__main__":
    main()
