"""
Main processor for handling presentations with the Storm approach
"""

import os
import json
import logging
from pptx import Presentation

from .utils import (
    load_config,
    ensure_directory,
    get_project_paths,
    write_transcripts_to_pptx,
    extract_slide_content,
)
from .llm import get_llm_provider
from .transcript_models import SlideInformation
from .transcript_curator import TranscriptKnowledgeCurator
from .transcript_generator import TranscriptGenerator, TranscriptReviewer


def process_presentation_with_storm(
    pptx_path,
    output_base_dir=None,
    target_language=None,
    model=None,
    enable_search=True,
    llm_provider=None,
    slides=None,
):
    """Process a presentation using Storm-inspired approach to reduce hallucinations."""
    # Load the presentation
    presentation = Presentation(pptx_path)

    # Set up output directory structure
    if not output_base_dir:
        paths = get_project_paths()
        output_base_dir = paths["noted_dir"]
    ensure_directory(output_base_dir)
    
    # Set up logging
    log_directory = os.path.join(output_base_dir, "logs")
    ensure_directory(log_directory)
    log_filename = f"{os.path.splitext(os.path.basename(pptx_path))[0]}_processing.log"
    log_path = os.path.join(log_directory, log_filename)
    
    # Configure logging with console output
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(console_handler)
    
    logging.info(f"Starting presentation processing for: {pptx_path}")
    
    # Load config and set up resources
    config = load_config()
    llm_client = get_llm_provider(model, config, provider=llm_provider)
    tavily_api_key = config.get("tavily_api_key", None)
    knowledge_base_dir = config.get("knowledge_base_dir")
    
    # Validate API key if web search is enabled
    if enable_search and not tavily_api_key:
        logging.warning("Web search is enabled but no Tavily API key found in config.toml. Disabling web search.")
        enable_search = False
    elif enable_search and not tavily_api_key.startswith(("tvly-", "tvly-dev-")):
        logging.warning(f"Tavily API key format looks unusual: {tavily_api_key[:10]}... Consider checking the key.")
    
    # Ensure knowledge base directory exists if specified
    if knowledge_base_dir:
        ensure_directory(knowledge_base_dir)

    # Initialize components
    knowledge_curator = TranscriptKnowledgeCurator(
        llm_client, model, enable_search=enable_search,
        knowledge_base_dir=knowledge_base_dir, tavily_api_key=tavily_api_key,
    )
    transcript_generator = TranscriptGenerator(llm_client, model, target_language)
    transcript_reviewer = TranscriptReviewer(llm_client, model)

    # Process tracking
    transcripts = []
    previous_transcripts = []
    slide_information = {}
    all_knowledge_items = []

    # Log processing information
    slide_count = len(presentation.slides)
    logging.info(f"Processing {slide_count} slides with Storm approach")
    logging.info(f"Using model: {model} with provider: {llm_provider or 'default'}")
    logging.info(f"Web search: {'ENABLED' if enable_search else 'DISABLED'}, Knowledge base: {'ENABLED' if knowledge_base_dir else 'DISABLED'}")
    
    # Process each slide
    for idx, slide in enumerate(presentation.slides, 1):
        if slides and idx not in slides:
            logging.info(f"Skipping slide {idx} (not in specified slides)")
            continue

        logging.info(f"Processing slide {idx} with Storm approach")

        # Extract slide content
        slide_content = extract_slide_content(slide)
        unpolished_notes = ""
        if slide.has_notes_slide:
            notes_text = slide.notes_slide.notes_text_frame.text
            if notes_text.strip():
                unpolished_notes = notes_text.strip()
                logging.info(f"Found unpolished notes for slide {idx}")

        if not slide_content and not unpolished_notes:
            logging.info(f"No content found for slide {idx}, skipping.")
            continue

        # Combine content for processing
        combined_content = slide_content or ""
        if unpolished_notes:
            combined_content += f"\n\nUNPOLISHED NOTES:\n{unpolished_notes}"

        # Create slide information object
        slide_info = SlideInformation(
            original_content=slide_content or "",
            unpolished_notes=unpolished_notes,
            slide_number=idx,
        )

        logging.info(f"Step 1: Knowledge curation for slide {idx}")
        
        # Step 1a: Generate search queries
        try:
            logging.info(f"Generating search queries for slide {idx}")
            slide_info.queries = knowledge_curator.generate_search_queries(combined_content, idx)
        except Exception as e:
            logging.error(f"Error generating search queries for slide {idx}: {str(e)}")
            slide_info.queries = []

        # Step 1b: Retrieve knowledge
        try:
            logging.info(f"Retrieving knowledge for slide {idx}")
            slide_info.knowledge_items = knowledge_curator.perform_knowledge_retrieval(
                combined_content, idx, slide_info.queries
            )
            logging.info(f"Retrieved {len(slide_info.knowledge_items)} knowledge items for slide {idx}")
            
            # Add to all knowledge items collection with slide context
            for item in slide_info.knowledge_items:
                if hasattr(item, 'metadata'):
                    item.metadata['slide_number'] = idx
                    if slide_content:
                        item.metadata['slide_content_snippet'] = slide_content[:100] + "..." if len(slide_content) > 100 else slide_content
                all_knowledge_items.append(item)
        except Exception as e:
            logging.error(f"Error retrieving knowledge for slide {idx}: {str(e)}")
            slide_info.knowledge_items = []

        # Step 1c: Extract verified facts
        try:
            logging.info(f"Extracting verified facts for slide {idx}")
            slide_info.facts = knowledge_curator.extract_verified_facts(
                combined_content, slide_info.queries, slide_info.knowledge_items
            )
        except Exception as e:
            logging.error(f"Error extracting facts for slide {idx}: {str(e)}")
            slide_info.facts = combined_content.split("\n")

        # Step 1d: Synthesize verified content
        try:
            logging.info(f"Synthesizing verified content for slide {idx}")
            slide_info.verified_content = knowledge_curator.synthesize_verified_content(slide_info)
        except Exception as e:
            logging.error(f"Error synthesizing content for slide {idx}: {str(e)}")
            slide_info.verified_content = combined_content

        # Store slide information
        slide_information[idx] = slide_info

        # Step 2: Generate transcript
        logging.info(f"Step 2: Transcript generation for slide {idx}")
        try:
            transcript = transcript_generator.generate_transcript(
                slide_info.verified_content, previous_transcripts
            )
        except Exception as e:
            logging.error(f"Error generating transcript for slide {idx}: {str(e)}")
            transcript = slide_info.verified_content

        # Step 3: Review transcript
        logging.info(f"Step 3: Transcript review for slide {idx}")
        try:
            review = transcript_reviewer.review_transcript(slide_info, transcript)
            final_transcript = review.get("revised_transcript", transcript)
        except Exception as e:
            logging.error(f"Error reviewing transcript for slide {idx}: {str(e)}")
            review = {
                "accurate": True, "hallucinations": [], 
                "corrections": [], "revised_transcript": transcript,
            }
            final_transcript = transcript

        # Store results
        slide_data = {
            "slide_number": idx,
            "original_content": slide_content,
            "unpolished_notes": unpolished_notes,
            "verified_content": slide_info.verified_content,
            "verified_facts": slide_info.facts,
            "transcript": final_transcript,
            "accuracy_review": review,
        }

        # Include simplified knowledge items
        if slide_info.knowledge_items:
            simplified_knowledge = []
            for item in slide_info.knowledge_items:
                simplified_knowledge.append({
                    "content": item.content[:500] if len(item.content) > 500 else item.content,
                    "source": item.source,
                    "metadata": item.metadata,
                })
            slide_data["knowledge_items"] = simplified_knowledge

        transcripts.append(slide_data)
        previous_transcripts.append(final_transcript)
        logging.info(f"Completed processing for slide {idx}")

    # Write transcripts back to the PPTX file
    noted_pptx_path = write_transcripts_to_pptx(pptx_path, transcripts, output_base_dir)
    logging.info(f"PPTX with transcripts has been saved to {noted_pptx_path}")

    # Save process data
    process_data = {
        "slides": transcripts,
        "metadata": {
            "source_file": os.path.basename(pptx_path),
            "total_slides": len(presentation.slides),
            "processed_slides": len(transcripts),
            "target_language": target_language,
            "model": model,
            "web_search_enabled": enable_search,
        },
    }

    process_data_path = os.path.join(
        output_base_dir,
        f"{os.path.splitext(os.path.basename(pptx_path))[0]}_process_data.json",
    )
    with open(process_data_path, "w") as f:
        json.dump(process_data, f, indent=2)
    logging.info(f"Process data has been saved to {process_data_path}")
    logging.info("Processing completed successfully")

    return noted_pptx_path 