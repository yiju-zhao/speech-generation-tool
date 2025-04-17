"""
Transcript generation and review modules
"""

import json
import logging
from .knowledge_base import KnowledgeRetriever


class TranscriptGenerator:
    """Generates transcripts from verified slide content."""

    def __init__(self, llm_client, model=None, target_language=None, knowledge_base_dir=None, tavily_api_key=None):
        self.llm_client = llm_client
        self.model = model
        self.target_language = target_language
        self.knowledge_base_dir = knowledge_base_dir
        self.tavily_api_key = tavily_api_key
        
        # Initialize knowledge retriever for RAG if a knowledge base is provided
        self.knowledge_retriever = None
        if knowledge_base_dir:
            self.knowledge_retriever = KnowledgeRetriever(
                knowledge_base_dir=knowledge_base_dir,
                web_search_enabled=tavily_api_key is not None,
                tavily_api_key=tavily_api_key
            )
            logging.info(f"Initialized knowledge retriever with base directory: {knowledge_base_dir}")
    
    def retrieve_technical_knowledge(self, slide_content, slide_number):
        """Retrieve relevant technical knowledge for terms in the slide content."""
        if not self.knowledge_retriever or not slide_content:
            return []
            
        logging.info(f"Retrieving technical knowledge for slide {slide_number}")
        
        # Extract technical terms from the slide content
        technical_terms = self.extract_technical_terms(slide_content)
        
        if not technical_terms:
            logging.info(f"No technical terms identified in slide {slide_number}")
            return []
            
        # Create queries for each technical term
        queries = [f"explain {term}" for term in technical_terms]
        logging.info(f"Generated {len(queries)} queries for technical terms: {', '.join(technical_terms)}")
        
        # Retrieve knowledge using those queries
        knowledge_items = self.knowledge_retriever.retrieve_for_slide(
            slide_content=slide_content,
            slide_number=slide_number,
            queries=queries,
            max_items=5
        )
        
        logging.info(f"Retrieved {len(knowledge_items)} knowledge items for technical terms")
        return knowledge_items
    
    def extract_technical_terms(self, slide_content):
        """Extract technical terms from the slide content that need explanation."""
        if not slide_content:
            return []
            
        # Use the LLM to identify technical terms that need explanation
        prompt = f"""
        You are analyzing slide content to identify technical terms and concepts that would benefit from additional explanation.
        
        SLIDE CONTENT:
        {slide_content}
        
        Extract ONLY the technical terms and complex concepts that would need explanation for a general audience.
        Focus on specialized terminology, acronyms, domain-specific concepts, and technical processes.
        DO NOT include common terms or simple concepts that don't need explanation.
        
        Format your response as a JSON list of strings, each string being a technical term.
        Examples: ["Neural Networks", "LSTM", "Transformer Architecture", "Gradient Descent"]
        If no technical terms need explanation, return an empty list []
        """
        
        result = self.llm_client.generate(prompt, self.model)
        
        # Parse the response
        try:
            terms = json.loads(result)
            if not isinstance(terms, list):
                logging.warning(f"Technical term extraction result not a list format: {result[:200]}")
                terms = []
        except json.JSONDecodeError as e:
            logging.warning(f"Failed to parse technical term JSON result: {e} - trying fallback extraction")
            # If not valid JSON, extract terms manually
            terms = []
            for line in result.split("\n"):
                line = line.strip()
                if line.startswith("-") or line.startswith("*"):
                    terms.append(line[1:].strip())
                elif line.startswith('"') and line.endswith('"'):
                    terms.append(line.strip('"'))
        
        return terms

    def generate_transcript(self, verified_content, slide_info=None, previous_transcripts=None):
        """Generate a transcript for a slide using verified content.
        
        Args:
            verified_content: The verified content of the slide
            slide_info: Information about the slide (position, type, etc.)
            previous_transcripts: Previous slides' transcripts for context
        """
        # Add all previous slides' transcripts as context if available
        context = ""
        if previous_transcripts:
            context = "\n".join(previous_transcripts)

        # Default slide position and type
        slide_position = "middle"
        slide_type = "content"
        slide_number = 0
        
        # Extract slide position and type if available
        if slide_info:
            if hasattr(slide_info, 'position'):
                slide_position = slide_info.position
            elif isinstance(slide_info, dict) and 'position' in slide_info:
                slide_position = slide_info['position']
                
            if hasattr(slide_info, 'type'):
                slide_type = slide_info.type
            elif isinstance(slide_info, dict) and 'type' in slide_info:
                slide_type = slide_info['type']
                
            if hasattr(slide_info, 'slide_number'):
                slide_number = slide_info.slide_number
            elif isinstance(slide_info, dict) and 'slide_number' in slide_info:
                slide_number = slide_info['slide_number']
        
        # Extract knowledge context specifically for technical explanations
        technical_knowledge = ""
        
        # First, try to use knowledge items from slide_info if available
        if slide_info and hasattr(slide_info, 'knowledge_items') and slide_info.knowledge_items:
            tech_items = []
            
            # First look for AI answers which tend to be more comprehensive
            ai_answer = None
            for item in slide_info.knowledge_items:
                if hasattr(item, "metadata") and item.metadata.get("is_ai_answer"):
                    ai_answer = item.content
                    break
                    
            if ai_answer:
                technical_knowledge = f"""
                **TECHNICAL KNOWLEDGE FROM SEARCH**:
                This knowledge should ONLY be used for explaining technical terms or concepts, not for adding unrelated information:
                
                {ai_answer[:800] if len(ai_answer) > 800 else ai_answer}
                """
            else:
                # If no AI answer, use the most relevant knowledge items
                for idx, item in enumerate(slide_info.knowledge_items[:2]):
                    tech_items.append(f"- {item.content[:400]}...")
                
                if tech_items:
                    technical_knowledge = f"""
                    **TECHNICAL KNOWLEDGE FROM SEARCH**:
                    This knowledge should ONLY be used for explaining technical terms or concepts, not for adding unrelated information:
                    
                    {chr(10).join(tech_items)}
                    """
        
        # If no knowledge items in slide_info or they're insufficient, use RAG retrieval
        if not technical_knowledge and self.knowledge_retriever:
            # Actively retrieve technical knowledge using RAG
            rag_items = self.retrieve_technical_knowledge(verified_content, slide_number)
            
            if rag_items:
                # Extract AI answers first
                ai_answers = [item for item in rag_items if hasattr(item, "metadata") and item.metadata.get("is_ai_answer")]
                
                if ai_answers:
                    # Use the first AI answer
                    ai_content = ai_answers[0].content
                    technical_knowledge = f"""
                    **TECHNICAL KNOWLEDGE FROM KNOWLEDGE BASE (RAG)**:
                    This knowledge should ONLY be used for explaining technical terms or concepts, not for adding unrelated information:
                    
                    {ai_content[:800] if len(ai_content) > 800 else ai_content}
                    """
                else:
                    # Use regular knowledge items
                    rag_content = []
                    for idx, item in enumerate(rag_items[:3]):
                        term = item.metadata.get("source_query", "").replace("explain ", "") if hasattr(item, "metadata") else ""
                        content = item.content[:350] + "..." if len(item.content) > 350 else item.content
                        rag_content.append(f"Term: {term}\nExplanation: {content}")
                    
                    if rag_content:
                        technical_knowledge = f"""
                        **TECHNICAL KNOWLEDGE FROM KNOWLEDGE BASE (RAG)**:
                        This knowledge should ONLY be used for explaining technical terms or concepts, not for adding unrelated information:
                        
                        {chr(10).join(rag_content)}
                        """

        # Create a prompt focused on verified content with restricted use of knowledge base
        prompt = f"""
        **Role**: Technical Speech Architect  
        **Objective**: Create a TTS-ready transcript with perfect factual accuracy  

        **Current Slide Information**:
        - Position: {slide_position} (first, middle, last)
        - Type: {slide_type} (title, content, thank_you, q_and_a, etc.)

        **Core Directives**:  
        1. STRICTLY adhere to the slide content - use ONLY the VERIFIED CONTENT as your primary source
        2. Create a CONCISE speech transcript that sounds natural when spoken
        3. Preserve exact terminology and numerical precision
        4. When technical terms or complex concepts appear, you MAY use the knowledge base information ONLY to provide a brief explanation
        5. DO NOT add any information that isn't in the verified content unless it's explaining a technical term
        6. FOLLOW THE SAME ORDER as the original slide content - do not rearrange or reorganize information

        **Input Sources**:  
        - Primary Source (MUST follow): {verified_content}
        {technical_knowledge}
        - Previous Transcript Context: {context}  

        **CRITICAL INSTRUCTION**:
        - For simple, non-technical content: ONLY use the verified slide content
        - For technical terms/concepts: You may BRIEFLY clarify using knowledge base
        - NEVER add unrelated facts or tangential information
        - ALWAYS prioritize the verified slide content over external knowledge
        - MAINTAIN the same sequence and flow of information as presented in the original slide

        **Output Specifications**:  
        `Language`: {self.target_language.upper() if self.target_language else "ENGLISH"}  
        `Format`: Pure speech text (no markup/annotations)  

        **Technical Protocols**:  
        ✓ Fact adherence: STRICTLY use facts from verified content 
        ✓ Term fidelity: Use source terms verbatim
        ✓ Speech optimization: Concise, natural speech patterns
        ✓ Technical explanations: ONLY when needed, BRIEF, and DIRECTLY relevant
        ✓ Sequence adherence: FOLLOW the same order of information as the original slide content
        ✓ Slide-specific handling:
          - If title/first slide: Create a brief, engaging introduction
          - If thank you slide: Keep it simple, just a brief thank you message
          - If Q&A slide: Briefly mention the Q&A session is starting
          - If content slide: Focus on key information, avoid exhaustive detail

        **Prohibited Elements**:  
        - Any facts not in the verified content (except brief technical explanations)
        - Additional background information not directly explaining a technical term
        - Tangential information or examples not directly from the slide
        - Non-verbal cues (parentheticals, pauses)
        - Rearranging or reorganizing the flow of information from the original slide
        
        **FORMAT RESTRICTION (CRITICAL)**:
        1. Output ONLY the actual transcript text
        2. DO NOT include any introductory phrases like "Here is the transcript:" or "以下是转录内容:"
        3. DO NOT include any explanations or notes after the transcript
        4. DO NOT use quotation marks to wrap the entire transcript
        5. Return ONLY the pure speech text that would be read aloud
        """

        return self.llm_client.generate(prompt, self.model)


class TranscriptReviewer:
    """Reviews generated transcripts for hallucinations and accuracy."""

    def __init__(self, llm_client, model=None):
        self.llm_client = llm_client
        self.model = model

    def review_transcript(self, slide_info, transcript):
        """Check transcript against verified facts to detect hallucinations."""
        facts_text = "\n".join([f"- {fact}" for fact in slide_info.facts])
        
        # Extract slide position and type if available
        slide_position = "unknown"
        slide_type = "unknown"
        
        if hasattr(slide_info, 'position'):
            slide_position = slide_info.position
        elif isinstance(slide_info, dict) and 'position' in slide_info:
            slide_position = slide_info['position']
            
        if hasattr(slide_info, 'type'):
            slide_type = slide_info.type
        elif isinstance(slide_info, dict) and 'type' in slide_info:
            slide_type = slide_info['type']

        # Extract original slide content if available
        original_content = ""
        if hasattr(slide_info, 'content'):
            original_content = slide_info.content
        elif isinstance(slide_info, dict) and 'content' in slide_info:
            original_content = slide_info['content']

        prompt = f"""
        You are a transcript fact-checker with expertise in detecting hallucinations and ensuring presentation-ready speech.
        
        SLIDE INFORMATION:
        - Position: {slide_position} (first, middle, last)
        - Type: {slide_type} (title, content, thank_you, q_and_a, etc.)
        
        ORIGINAL SLIDE CONTENT:
        {original_content}
        
        VERIFIED FACTS:
        {facts_text}
        
        GENERATED TRANSCRIPT:
        {transcript}
        
        Analyze the transcript for accuracy and presentation quality:
        
        1. Is it supported by the verified facts without hallucinations?
        2. Is it appropriately concise and focused for a presentation?
        3. Is it appropriate for the slide type? (e.g., brief for thank you slides)
        4. Does it sound natural when spoken aloud?
        5. Does it follow the same order and structure as the original slide content?
        6. Is it engaging and effective for a presentation audience?
        
        Format your response as a JSON object with these fields:
        - "accurate" (boolean): Whether the transcript contains ONLY information from verified facts
        - "presentation_ready" (boolean): Whether the transcript is appropriately concise and focused
        - "follows_slide_order" (boolean): Whether the transcript follows the same order as the original slide content
        - "presentation_quality" (number, 1-10): Rating the overall quality and effectiveness as presentation speech
        - "hallucinations" (list): Any claims in the transcript not supported by verified facts
        - "style_issues" (list): Any issues with presentation style (too verbose, too academic, etc.)
        - "order_issues" (list): Any issues with the transcript not following the original slide content order
        - "corrections" (list): Suggested corrections for any inaccuracies
        - "revised_transcript" (string): A revised version that is factually accurate, presentation-ready, and follows the original order
        """

        result = self.llm_client.generate(prompt, self.model)
        try:
            # Try to parse as JSON
            review = json.loads(result)
        except:
            # If not valid JSON, return a simplified format
            review = {
                "accurate": False,
                "presentation_ready": False,
                "follows_slide_order": False,
                "presentation_quality": 5,
                "hallucinations": [],
                "style_issues": [],
                "order_issues": [],
                "corrections": [],
                "revised_transcript": transcript,
            }

            lines = result.split("\n")
            is_revised = False
            revised_transcript = []

            for line in lines:
                if "hallucination:" in line.lower() or "issue:" in line.lower():
                    review["hallucinations"].append(line)
                elif "style issue:" in line.lower():
                    review["style_issues"].append(line)
                elif "order issue:" in line.lower():
                    review["order_issues"].append(line)
                elif "correction:" in line.lower():
                    review["corrections"].append(line)
                elif line.startswith("Revised transcript:") or is_revised:
                    is_revised = True
                    revised_transcript.append(line)

            if revised_transcript:
                review["revised_transcript"] = "\n".join(revised_transcript)

        return review 