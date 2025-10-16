"""
Knowledge curation module for transcript generation
"""

import json
import logging  # Add explicit import for logging
import os
from pathlib import Path

from .knowledge_base import KnowledgeRetriever
from .models import SlideInformation


class TranscriptKnowledgeCurator:
    """Curates knowledge from slides to reduce hallucinations."""

    def __init__(
        self,
        llm_client,
        model=None,
        max_queries_per_slide=3,
        enable_search=True,
        knowledge_base_dir=None,
        tavily_api_key=None,
    ):
        self.llm_client = llm_client
        self.model = model
        self.max_queries_per_slide = max_queries_per_slide
        self.enable_search = enable_search
        self.knowledge_base_dir = knowledge_base_dir
        
        # Ensure knowledge base directory exists
        if knowledge_base_dir:
            self.ensure_knowledge_base_dir()

        # Initialize knowledge retriever with proper web search configuration
        self.knowledge_retriever = KnowledgeRetriever(
            knowledge_base_dir=knowledge_base_dir,
            web_search_enabled=enable_search,
            tavily_api_key=tavily_api_key
        )
        
        # Store web search results from the last retrieval
        self.last_web_results = []
        
        # Initialize knowledge base with placeholder if needed
        if knowledge_base_dir:
            self.initialize_knowledge_base()

    def ensure_knowledge_base_dir(self):
        """Ensure the knowledge base directory exists."""
        kb_dir = Path(self.knowledge_base_dir)
        if not kb_dir.exists():
            try:
                logging.info(f"Creating knowledge base directory: {kb_dir}")
                kb_dir.mkdir(parents=True, exist_ok=True)
                logging.info(f"Successfully created knowledge base directory")
            except Exception as e:
                logging.error(f"Failed to create knowledge base directory: {e}")
                
    def initialize_knowledge_base(self):
        """Initialize the knowledge base with basic information if it's empty."""
        kb_dir = Path(self.knowledge_base_dir)
        json_files = list(kb_dir.glob("*.json"))
        
        # If no JSON files exist, create a basic placeholder
        if not json_files:
            try:
                logging.info("No knowledge base files found. Creating initial placeholder.")
                
                # Create a basic knowledge item
                from .knowledge_base import KnowledgeItem
                import datetime
                
                # Simple placeholder knowledge
                initial_items = [
                    KnowledgeItem(
                        content="This is an initial knowledge item created when the knowledge base was initialized.",
                        source="system_initialization",
                        metadata={
                            "title": "Knowledge Base Initialization",
                            "retrieval_method": "initialization",
                            "creation_date": str(datetime.datetime.now())
                        }
                    )
                ]
                
                # Save to file
                initial_file = kb_dir / "initial_knowledge.json"
                
                # Convert to dictionaries for serialization
                items_data = [item.to_dict() for item in initial_items]
                
                with open(initial_file, "w") as f:
                    import json
                    json.dump({"items": items_data}, f, indent=4)
                    
                logging.info(f"Created initial knowledge base file: {initial_file}")
                
                # Now reload the knowledge base to include this file
                self.knowledge_retriever.knowledge_base.load_from_directory(self.knowledge_base_dir)
                logging.info("Reloaded knowledge base with initial file")
                
            except Exception as e:
                logging.error(f"Failed to initialize knowledge base: {e}")

    def generate_search_queries(self, slide_content, slide_number):
        """Generate search queries to verify slide content."""
        # Log the slide content for debugging (truncated for brevity)
        content_preview = slide_content[:100] + "..." if len(slide_content) > 100 else slide_content
        logging.info(f"Generating search queries for slide {slide_number} content: {content_preview}")
        
        # Enhanced prompt for better query generation focused on technical terms
        prompt = f"""
        You are an expert researcher analyzing slide {slide_number} content to identify key technical terms that need verification or clarification.
        
        SLIDE CONTENT:
        {slide_content}
        
        Your task is to create specific search queries ONLY for technical terms, complex concepts, or claims that require additional background knowledge. DO NOT generate queries for simple, common knowledge facts.
        
        IMPORTANT GUIDELINES:
        - First, assess if this slide contains any technical terms or complex concepts that actually need clarification
        - If the slide content is simple and contains no technical terms requiring clarification, return an empty list []
        - Generate up to {self.max_queries_per_slide} queries ONLY if there are technical terms that need background knowledge
        - Focus on terms that would benefit from expanded background knowledge
        - Make queries specific - target precise technical terms
        - Start queries with "explain", "define", or "what is" for better results
        - Include proper context for acronyms or ambiguous terms
        - DO NOT generate queries for basic concepts that don't need clarification
        
        Format your response as a JSON list of strings, with each string being a search query.
        If no queries are needed, return an empty list: []
        """

        result = self.llm_client.generate(prompt, self.model)
        
        # Try to parse as JSON with better error handling
        try:
            queries = json.loads(result)
            if not isinstance(queries, list):
                logging.warning(f"Query result not a list format: {result[:200]}")
                queries = []
        except json.JSONDecodeError as e:
            logging.warning(f"Failed to parse JSON query result: {e} - trying fallback extraction")
            # If not valid JSON, extract queries manually
            queries = []
            for line in result.split("\n"):
                line = line.strip()
                if line.startswith("-") or line.startswith("*"):
                    queries.append(line[1:].strip())
                elif len(line) > 10 and not line.startswith("#") and not line.startswith("{") and not line.startswith("}"):
                    queries.append(line)

        # Filter out empty queries and truncate to max queries
        queries = [q for q in queries if q and len(q.strip()) > 5][:self.max_queries_per_slide]
        
        # Log the generated queries
        if queries:
            for i, query in enumerate(queries):
                logging.info(f"Generated query {i+1} for slide {slide_number}: {query}")
        else:
            logging.info(f"No queries generated for slide {slide_number} - content may be simple or no technical terms found.")
        
        return queries

    def perform_knowledge_retrieval(self, slide_content, slide_number, queries):
        """Retrieve knowledge with proper web search integration"""
        if not queries:
            logging.warning(f"Empty queries provided for slide {slide_number}. Knowledge retrieval may fail.")
            # Extract key terms from slide content as backup
            if slide_content:
                import re
                words = re.findall(r'\b\w+\b', slide_content)
                significant_words = [w for w in words if len(w) > 5][:10]  # Get longer words
                if significant_words:
                    backup_query = " ".join(significant_words)
                    queries = [backup_query]
                    logging.info(f"Created backup query from significant terms: {backup_query}")
        
        # Log all queries being used for retrieval
        logging.info(f"Performing knowledge retrieval for slide {slide_number} with {len(queries)} queries:")
        for i, q in enumerate(queries):
            logging.info(f"  Query {i+1}: {q}")
        
        # Ensure no empty queries are passed to the search API
        filtered_queries = [q for q in queries if q and len(q.strip()) > 0]
        if len(filtered_queries) != len(queries):
            logging.warning(f"Removed {len(queries) - len(filtered_queries)} empty queries")
        
        # Enhance queries with slide number context for better results
        enhanced_queries = []
        for q in filtered_queries:
            # If query is very short, try to augment it with slide context
            if len(q) < 20 and slide_content:
                # Extract the first sentence from slide content to provide context
                first_sentence = slide_content.split(".", 1)[0] if "." in slide_content else slide_content[:50]
                enhanced_q = f"{q} - context: {first_sentence}"
                enhanced_queries.append(enhanced_q)
            else:
                enhanced_queries.append(q)
                
        results = self.knowledge_retriever.retrieve_for_slide(
            slide_content=slide_content,
            slide_number=slide_number,
            queries=enhanced_queries,
            max_items=5
        )
        
        logging.info(f"Retrieved {len(results)} knowledge items for slide {slide_number}")
        
        # Check if we got zero results despite having queries
        if not results and filtered_queries:
            logging.warning(f"No knowledge items found for slide {slide_number} despite queries. Trying broader terms.")
            # Try a more general search based on slide title or first sentence
            if slide_content:
                first_sentence = slide_content.split(".", 1)[0] if "." in slide_content else slide_content[:50]
                broader_query = first_sentence
                logging.info(f"Trying broader query: {broader_query}")
                
                # Try with a single broader query
                results = self.knowledge_retriever.retrieve_for_slide(
                    slide_content=slide_content,
                    slide_number=slide_number,
                    queries=[broader_query],
                    max_items=5
                )
                
                logging.info(f"Broader query retrieved {len(results)} knowledge items")
        
        # Save the retrieved knowledge to the knowledge base
        if results and self.knowledge_base_dir:
            self.save_slide_knowledge_to_file(slide_number, results)
        
        return results
        
    def save_slide_knowledge_to_file(self, slide_number, knowledge_items):
        """Save knowledge items for a slide to a dedicated file in the knowledge base."""
        if not self.knowledge_base_dir or not knowledge_items:
            return
            
        try:
            import json
            from pathlib import Path
            
            kb_dir = Path(self.knowledge_base_dir)
            
            # Create a filename based on the slide number
            slide_kb_filename = f"slide_{slide_number:03d}_knowledge.json"
            slide_kb_file = kb_dir / slide_kb_filename
            
            # Convert items to dictionaries for serialization
            items_data = []
            for item in knowledge_items:
                # Ensure the metadata includes the slide number
                if hasattr(item, "metadata"):
                    item.metadata["slide_number"] = slide_number
                
                # Add to the list
                if hasattr(item, "to_dict"):
                    items_data.append(item.to_dict())
                else:
                    # Fallback for non-KnowledgeItem objects
                    items_data.append({
                        "content": getattr(item, "content", str(item)),
                        "source": getattr(item, "source", "unknown"),
                        "metadata": getattr(item, "metadata", {"slide_number": slide_number})
                    })
            
            # Write to file
            with open(slide_kb_file, "w") as f:
                json.dump({"items": items_data}, f, indent=4)
                
            logging.info(f"Saved {len(items_data)} knowledge items for slide {slide_number} to {slide_kb_file}")
            
            # Reload the knowledge base to ensure it includes the new items
            self.knowledge_retriever.knowledge_base.load_from_directory(self.knowledge_base_dir)
            logging.info(f"Reloaded knowledge base after adding slide {slide_number} knowledge")
            
        except Exception as e:
            logging.error(f"Error saving slide knowledge to file: {e}")

    def extract_verified_facts(self, slide_content, queries, knowledge_items=None):
        """Extract verified facts from slide content and knowledge."""
        knowledge_info = ""
        ai_answer = None

        if knowledge_items:
            # Check if we have any Tavily AI-generated answers
            for item in knowledge_items:
                if hasattr(item, "metadata") and item.metadata.get("is_ai_answer"):
                    ai_answer = item.content
                    logging.info("Using Tavily AI-generated answer in fact verification")
                    break
            
            # Prepare knowledge info for the prompt
            knowledge_info = "RETRIEVED KNOWLEDGE:\n"
            
            # Start with AI answer if available
            if ai_answer:
                knowledge_info += f"[AI] Tavily AI Answer: {ai_answer}\n\n"
                
            # Add other knowledge items
            for idx, item in enumerate(knowledge_items, 1):
                if not (hasattr(item, "metadata") and item.metadata.get("is_ai_answer")):
                    knowledge_info += f"[{idx}] Source: {item.source}\n"
                    knowledge_info += (
                        f"Content: {item.content[:500]}...\n\n"  # Limit content length
                    )

        prompt = f"""
        You are an expert fact checker examining slide content and retrieved knowledge.
        
        SLIDE CONTENT:
        {slide_content}
        
        {knowledge_info}
        
        KEY INFORMATION NEEDS:
        {", ".join(queries)}
        
        Extract verified facts based on the slide content and retrieved knowledge. Focus on:
        1. Precise technical terminology
        2. Numerical data and statistics
        3. Names, dates, and proper nouns
        4. Technical processes or relationships
        
        {"IMPORTANT: Prioritize information from the Tavily AI Answer as it contains verified information." if ai_answer else ""}
        
        For each fact, indicate if it is verified by knowledge sources or only from the slide.
        
        Format your response as a JSON list of objects, each with these properties:
        - "fact": the verified fact
        - "source": "slide" if from slide only, "knowledge" if confirmed by knowledge sources, "both" if in both
        - "confidence": a number between 0-1 indicating confidence level
        
        Example: [
            {{"fact": "Example fact 1", "source": "both", "confidence": 0.95}},
            {{"fact": "Example fact 2", "source": "slide", "confidence": 0.7}}
        ]
        """

        result = self.llm_client.generate(prompt, self.model)
        facts = []

        try:
            # Try to parse as JSON
            facts_data = json.loads(result)
            if isinstance(facts_data, list):
                # Extract just the fact strings for backward compatibility
                facts = [item.get("fact") for item in facts_data if item.get("fact")]
        except:
            # If not valid JSON, extract facts manually
            for line in result.split("\n"):
                line = line.strip()
                if line.startswith("-") or line.startswith("*"):
                    facts.append(line[1:].strip())
                elif line.startswith("{") and line.endswith("}"):
                    # Might be a JSON object on a single line
                    try:
                        fact_obj = json.loads(line)
                        if "fact" in fact_obj:
                            facts.append(fact_obj["fact"])
                    except:
                        pass
                elif len(line) > 10 and not line.startswith("#"):
                    facts.append(line)

        # Log the number of facts extracted
        logging.info(f"Extracted {len(facts)} verified facts for content")
        return facts

    def synthesize_verified_content(self, slide_info: SlideInformation):
        """Create a verified content synthesis based on extracted facts."""
        facts_text = "\n".join([f"- {fact}" for fact in slide_info.facts])

        knowledge_context = ""
        ai_answer = None
        
        if slide_info.knowledge_items:
            # Check if we have any Tavily AI-generated answers
            for item in slide_info.knowledge_items:
                if hasattr(item, "metadata") and item.metadata.get("is_ai_answer"):
                    ai_answer = item.content
                    break
            
            # Start knowledge context with AI answer if available    
            knowledge_context = "KNOWLEDGE SOURCES:\n"
            if ai_answer:
                knowledge_context += f"[AI] Tavily AI Answer: {ai_answer[:500]}...\n\n"
            
            # Add other knowledge sources
            for idx, item in enumerate(slide_info.knowledge_items[:3], 1):
                if not (hasattr(item, "metadata") and item.metadata.get("is_ai_answer")):
                    knowledge_context += f"[{idx}] Source: {item.source}\n"
                    if hasattr(item, "metadata") and item.metadata.get("title"):
                        knowledge_context += f"Title: {item.metadata.get('title')}\n"

        prompt = f"""
        You are synthesizing verified information for slide {slide_info.slide_number}.
        
        ORIGINAL SLIDE CONTENT (use this to preserve structure and emphasis):
        {slide_info.original_content}
        
        VERIFIED FACTS (must all be included):
        {facts_text}
        
        {knowledge_context}
        
        UNPOLISHED NOTES (if available):
        {slide_info.unpolished_notes}
        
        {"IMPORTANT: The Tavily AI Answer provides verified information that should be prioritized in your synthesis." if ai_answer else ""}
        
        Create a coherent synthesis that:
        1. Includes EVERY verified fact (do not omit any verified content)
        2. Uses the ORIGINAL SLIDE CONTENT to guide structure and ordering
        3. Maintains technical precision; paraphrase allowed but do not change meaning
        4. Adds NO new facts beyond the verified facts
        5. Resolves conflicts in favor of verified facts if wording differs
        6. Uses natural, flowing language suitable for a presentation transcript
        
        Output a single concise paragraph that will serve as the factual foundation for the final transcript.
        """

        verified_content = self.llm_client.generate(prompt, self.model)
        
        # Log a preview of the verified content
        content_preview = verified_content[:100] + "..." if len(verified_content) > 100 else verified_content
        logging.info(f"Generated verified content with {'AI answer integration' if ai_answer else 'standard verification'}: {content_preview}")
        
        return verified_content 
