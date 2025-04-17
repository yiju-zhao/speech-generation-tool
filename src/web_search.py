"""
Web search utilities for retrieving information to enhance transcript generation.
"""

import os
import logging
import backoff
from typing import List, Dict, Any

from tavily import TavilyClient



class TavilySearchRetriever:
    """Retrieve information from web searches using Tavily."""

    def __init__(self, api_key=None, max_results=3):
        """
        Initialize the Tavily Search retriever.

        Args:
            api_key: Tavily API key (can also be set as TAVILY_API_KEY environment variable)
            max_results: Maximum number of search results to return
        """
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        self.max_results = max_results
        self.usage = 0
        self.tavily_client = None
        self.last_error = None

        if not self.api_key:
            logging.error("No Tavily API key provided. Web search will be disabled.")
            return

        # Initialize the Tavily client
        self.tavily_client = TavilyClient(api_key=self.api_key)
        logging.info("Tavily client initialized successfully")


    @property
    def is_available(self):
        """Check if the search retriever is available and properly configured."""
        return self.api_key is not None

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search the web using Tavily.

        Args:
            query: Search query string

        Returns:
            List of search results
        """
        if not self.is_available:
            logging.warning("Tavily search retriever is not available")
            return []

        if not query or len(query.strip()) == 0:
            logging.warning("Empty query provided to Tavily search")
            return []

        # Clean and normalize the query
        query = query.strip()
        
        # Log the query
        query_id = self.usage + 1
        logging.info(f"Tavily search query #{query_id}: {query}")
        self.usage += 1

        # Advanced search depth
        search_config = {
            "query": query, 
            "max_results": self.max_results, 
            "include_raw_content": True,
            "search_depth": "advanced",  # Use advanced search for better results
            "include_answer": "advanced",      # Request AI-generated answer
        }
        
        logging.info(f"Performing Tavily search with advanced depth: {query}")
        response_data = self.tavily_client.search(**search_config)
        results = response_data.get("results", [])
        
        # Extract the answer if available
        answer = response_data.get("answer", "")
        if answer:
            logging.info(f"Received AI-generated answer from Tavily: {answer[:100]}...")
        else:
            logging.info("No AI-generated answer received from Tavily")
        
        if not results:
            # If no results, try basic search with more results
            logging.info("Advanced search returned no results, trying basic search with more results")
            search_config.update({
                "search_depth": "basic",
                "max_results": min(10, self.max_results * 2)  # Try with more results
            })
            response_data = self.tavily_client.search(**search_config)
            results = response_data.get("results", [])
            # Update answer if one was provided
            if response_data.get("answer"):
                answer = response_data.get("answer")
                logging.info(f"Received AI-generated answer from basic search: {answer[:100]}...")
        
        # If still no results, try with different topic settings
        if not results:
            logging.info("Basic search returned no results, trying with 'news' topic")
            search_config.update({
                "topic": "news",
                "time_range": "week"
            })
            response_data = self.tavily_client.search(**search_config)
            results = response_data.get("results", [])
            # Update answer if one was provided
            if response_data.get("answer"):
                answer = response_data.get("answer")
                logging.info(f"Received AI-generated answer from news search: {answer[:100]}...")
        
        logging.info(f"Tavily search query #{query_id} returned {len(results)} results")
        
        # Log a brief overview of results
        for i, result in enumerate(results):
            title = result.get("title", "No title")
            url = result.get("url", "No URL")
            logging.info(f"  Result {i+1}: {title[:50]}... - {url}")
        
        # Add the AI-generated answer to the results as metadata
        if answer:
            # Create a special result entry for the AI answer
            answer_result = {
                "title": "Tavily AI-Generated Answer",
                "content": answer,
                "url": "tavily:ai-answer",
                "is_ai_answer": True,
                "source_query": query
            }
            # Add it to the beginning of results for priority
            results.insert(0, answer_result)
        
        return results


    def search_multiple(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Search multiple queries and collect results.

        Args:
            queries: List of search queries

        Returns:
            Combined search results from all queries
        """
        if not self.is_available:
            logging.warning("Tavily search retriever is not available for multi-search")
            return []

        if not queries:
            logging.warning("No queries provided to search_multiple")
            return []

        # Filter out empty queries
        valid_queries = [q for q in queries if q and len(q.strip()) > 0]
        if not valid_queries:
            logging.warning("No valid queries after filtering in search_multiple")
            return []

        results = []
        for query in valid_queries:
            try:
                query_results = self.search(query)
                if query_results:
                    # Add the query that generated these results
                    for result in query_results:
                        result["source_query"] = query
                    results.extend(query_results)
                else:
                    logging.warning(f"No results found for query: {query}")
            except Exception as e:
                self.last_error = str(e)
                logging.error(f"Error searching query '{query}': {e}")

        # Remove duplicates by URL
        seen_urls = set()
        unique_results = []
        for result in results:
            if result.get("url") not in seen_urls:
                seen_urls.add(result.get("url"))
                unique_results.append(result)

        logging.info(f"Combined search returned {len(unique_results)} unique results from {len(valid_queries)} queries")
        return unique_results
        
    def get_last_error(self):
        """
        Get the last error message from the Tavily search.
        
        Returns:
            The last error message or None if no errors have occurred
        """
        return self.last_error
