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
        **Length**: WORDLIMIT: Reduce the overlimit script to less than 200 words for highly informed slides, 30 words for title slides, and 50 words for thank you or Q&A slides. The revised transcript must be concise and fit within these limits.

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
        You are an expert Communications Coach and Speechwriter. Your task is to transform a dry, fact-based script into a polished, engaging, and natural-sounding speech, while maintaining 100% factual accuracy based on the provided `verified_facts`.

        **CORE DIRECTIVES**

        1.  **WORDLIMIT:** Strictly enforce word limits: under 200 for content slides, 30 for titles, 50 for thank you/Q&A. Be concise.
        2.  **100% Factually Grounded:** The final transcript MUST ONLY contain information explicitly supported by the `verified_facts`. Eliminate all hallucinations.
        3.  **Preserve Logical Structure:** The revised transcript's topic flow must exactly match the sequence in the `original_slide_content`.
        4.  **Natural & Engaging Tone:** The final script must sound like a confident, knowledgeable human speaking to an audience. This is critical.

        ---
        **ORAL POLISH STYLE GUIDE (Apply this during revision)**

        This is how you will transform the text from written to spoken language.

        * **Use Contractions:** Change "it is" to "it's," "we will" to "we'll," "do not" to "don't." Make it sound like a person talking.
        * **Add Conversational Transitions:** Inject phrases to guide the listener, like "So, let's dive in," "Now, what does this actually mean for us?", "The key takeaway here is...", or "Let's turn our attention to..."
        * **Address the Audience:** Use "you," "we," and "our" to create a direct connection with the audience.
        * **Simplify Complex Sentences:** Break down long, complex sentences into shorter, more digestible ones. Rephrase passive voice into active voice (e.g., change "An increase was seen" to "We saw an increase").
        * **Ask Rhetorical Questions:** Engage the audience by asking questions you intend to answer, like "So, what's the bottom line?" or "How did we get here?"

        **EXAMPLE OF TRANSFORMATION:**
        * **Dry/Written Text:** "The implementation of the new protocol resulted in a 30% reduction in system latency."
        * **Polished/Oral Transcript:** "So, after we rolled out the new protocol, what was the impact? We saw system latency drop by a massive 30%. That's a huge improvement for our users."

        ---
        **INPUT VARIABLES**

        * `slide_position`: {slide_position}
        * `slide_type`: {slide_type}
        * `original_slide_content`: {original_content}
        * `verified_facts`: {facts_text}
        * `generated_transcript`: {transcript}

        ---
        **TASK: 3-STEP PROCESS**

        **Step 1: Critical Analysis**
        Evaluate the `generated_transcript`.
        * **Factual Verification:** Does it align perfectly with `verified_facts`?
        * **Structural Alignment:** Does it follow the order of `original_slide_content`?
        * **Delivery Quality:** Is it concise and within the WORDLIMIT?

        **Step 2: Construct JSON Output**
        Fill out the JSON schema below with your detailed analysis.

        **Step 3: Revise and Polish**
        Create the `revised_transcript`. This is the most important step. **Apply the Oral Polish Style Guide** to transform the text into a presentation-ready script. It must be factually perfect and sound completely natural.

        ---
        **OUTPUT SCHEMA (JSON Only)**
        Your entire output must be a single, valid JSON object.

        {{
        "assessment": {{
            "is_factually_accurate": <boolean>,
            "follows_slide_order": <boolean>,
            "is_presentation_ready": <boolean>
        }},
        "presentation_quality_score": <number, 1-10>,
        "detailed_critique": {{
            "factual_errors": [
            {{
                "unsupported_claim": "<The specific claim from the transcript that is not supported by the facts>",
                "reasoning": "<A clear explanation of why the claim is a factual error or hallucination>"
            }}
            ],
            "style_and_tone_issues": [
            {{
                "issue_description": "<Description of the style problem (e.g., 'Overly verbose and exceeds time limit', 'Unnatural phrasing', 'Tone is too academic')>",
                "offending_text_example": "<The specific text from the transcript that demonstrates this issue>"
            }}
            ],
            "structural_errors": [
            {{
                "issue": "The transcript's structure does not follow the original slide's content order.",
                "details": "<Explanation of how the order is incorrect (e.g., 'It introduces the third point before the second point')>"
            }}
            ]
        }},
        "revised_transcript": "<The final, polished, 100% accurate, and presentation-ready transcript as a single string.>"
        }}
        """

        # (The rest of your code for this method remains the same)
        result = self.llm_client.generate(prompt, self.model)
        try:
            review = json.loads(result)
        except json.JSONDecodeError:
            review = {
                "assessment": {
                    "is_factually_accurate": False,
                    "follows_slide_order": False,
                    "is_presentation_ready": False,
                },
                "presentation_quality_score": 3,
                "detailed_critique": {
                    "factual_errors": [],
                    "style_and_tone_issues": [{"issue_description": "Failed to parse LLM response as JSON.", "offending_text_example": result[:200]}],
                    "structural_errors": []
                },
                "revised_transcript": transcript,
            }
            logging.error(f"Failed to parse review transcript JSON response: {result}")
        
        return review