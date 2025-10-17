"""
Transcript generation and review modules
"""

import json
import re
import logging
from functools import lru_cache
from typing import List, Optional, Any, Dict
import re

# Optional semantic similarity support — used only if available
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _SEM_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    _SEM_MODEL = None

# Module-level logger only; configuration is owned by the caller/processor
logger = logging.getLogger(__name__)


class TranscriptGenerator:
    """Generates concise, slide-faithful transcripts with strictly limited RAG use."""

    def __init__(
        self,
        llm_client,
        model: Optional[str] = None,
        target_language: Optional[str] = None,
        knowledge_base_dir: Optional[str] = None,
        knowledge_retriever: Optional[Any] = None,
        max_rag_items: int = 2,
    ):
        """
        Args:
            llm_client: object with `.generate(prompt, model)` method returning string.
            knowledge_retriever: optional pre-instantiated retriever; if None and knowledge_base_dir
                                is provided you can initialize your own retriever externally.
            max_rag_items: maximum number of knowledge snippets to include for clarification.
        """
        self.llm_client = llm_client
        self.model = model
        self.target_language = target_language or "english"
        self.max_rag_items = max_rag_items

        # Prefer explicit retriever if passed; else it's the caller's responsibility
        self.knowledge_retriever = knowledge_retriever
        if knowledge_base_dir and not self.knowledge_retriever:
            try:
                from .knowledge_base import KnowledgeRetriever  # local import if exists
                self.knowledge_retriever = KnowledgeRetriever(knowledge_base_dir=knowledge_base_dir)
                logger.info(f"Initialized KnowledgeRetriever at {knowledge_base_dir}")
            except Exception:
                logger.warning("knowledge_retriever not provided and KnowledgeRetriever import failed.")

    # ---------- Technical term extraction ----------
    @lru_cache(maxsize=256)
    def extract_technical_terms(self, slide_content: str) -> List[str]:
        """
        Identify a short list of technical terms from slide content.
        - Use simple heuristics to skip titles/short slides.
        - Fall back to LLM-based extraction when needed.
        """
        if not slide_content:
            return []

        text = slide_content.strip()
        # Heuristics: skip very short slides or obvious title/thank-you slides
        lower = text.lower()
        if len(text.split()) < 8 or any(k in lower for k in ("thank you", "q&a", "questions", "title")):
            return []

        # Quick heuristic detection: uppercase tokens or acronyms
        terms = []
        for token in text.replace("(", " ").replace(")", " ").split():
            if token.isupper() and 2 <= len(token) <= 6:
                terms.append(token)
        # Keep distinct small set
        terms = list(dict.fromkeys(terms))[:3]

        if terms:
            return terms

        # If heuristics didn't find anything, prompt LLM for short list
        prompt = f"""
        Extract up to 3 specialized technical terms or acronyms from the following slide text.
        Return a JSON list of strings only (e.g. ["Transformer", "LSTM"]).
        If none, return [].

        SLIDE:
        {slide_content}
        """
        try:
            result = self.llm_client.generate(prompt, self.model)
            parsed = json.loads(result)
            if isinstance(parsed, list):
                return parsed[:3]
        except Exception as e:
            logger.debug(f"LLM term extraction fallback failed: {e}")

        return []

    # ---------- Knowledge retrieval ----------
    def retrieve_technical_knowledge(self, slide_content: str, slide_number: int = 0) -> List[Any]:
        """Return a short list of knowledge items (max self.max_rag_items)."""
        if not self.knowledge_retriever or not slide_content:
            return []

        technical_terms = self.extract_technical_terms(slide_content)[:3]
        if not technical_terms:
            return []

        # To avoid triggering web search in the generator path, rely on KB-only retrieval
        # by passing an empty query list (web search branch requires non-empty queries).
        queries: List[str] = []
        try:
            items = self.knowledge_retriever.retrieve_for_slide(
                slide_content=slide_content,
                slide_number=slide_number,
                queries=queries,
                max_items=self.max_rag_items,
            )
            logger.info(f"RAG: retrieved {len(items)} items for slide {slide_number}")
            return items or []
        except Exception as e:
            logger.warning(f"Knowledge retrieval failed: {e}")
            return []

    # ---------- Semantic pre-check (optional) ----------
    def semantic_similarity(self, a: str, b: str) -> Optional[float]:
        """Return cosine similarity between a and b if sentence-transformers available; else None."""
        if not _SEM_MODEL:
            return None
        try:
            a_vec = _SEM_MODEL.encode(a, convert_to_tensor=True)
            b_vec = _SEM_MODEL.encode(b, convert_to_tensor=True)
            sim = st_util.cos_sim(a_vec, b_vec).item()
            return float(sim)
        except Exception as e:
            logger.debug(f"Semantic similarity failed: {e}")
            return None

    # ---------- Transcript generation ----------
    def generate_transcript(
        self,
        slide_info: Any,
        previous_transcripts: Optional[List[str]] = None,
        enforce_semantic_check: bool = False,
    ) -> str:
        """
        Generate a TTS-ready transcript anchored to the original slide content.

        enforce_semantic_check: if True and sentence-transformers is available, we will check
        that the generated transcript is semantically close to the slide before returning.
        """

        # --- Gather slide metadata ---
        slide_position = "middle"
        slide_type = "content"
        slide_number = 0
        original_slide_text = ""
        unpolished_notes = ""
        verified_facts: List[str] = []
        # slide_info is required; support both dict and dataclass
        if isinstance(slide_info, dict):
            slide_position = slide_info.get("position", slide_position)
            slide_type = slide_info.get("type", slide_type)
            slide_number = slide_info.get("slide_number", slide_number)
            original_slide_text = slide_info.get("original_content", original_slide_text)
            unpolished_notes = slide_info.get("unpolished_notes", unpolished_notes)
            verified_facts = slide_info.get("facts", []) or []
        else:
            slide_position = getattr(slide_info, "position", slide_position)
            slide_type = getattr(slide_info, "type", slide_type)
            slide_number = getattr(slide_info, "slide_number", slide_number)
            original_slide_text = getattr(slide_info, "original_content", original_slide_text)
            unpolished_notes = getattr(slide_info, "unpolished_notes", unpolished_notes)
            verified_facts = getattr(slide_info, "facts", []) or []

        # Primary text for prompting and checks
        slide_text = original_slide_text or ""

        context = "\n".join(previous_transcripts) if previous_transcripts else ""

        # --- Get minimal technical context (short snippets only) ---
        technical_knowledge = ""
        # Prefer using knowledge already attached to slide_info (from cached KB)
        rag_items = []
        attached_items = []
        try:
            attached_items = getattr(slide_info, "knowledge_items", []) or []
        except Exception:
            attached_items = []

        if attached_items:
            rag_items = attached_items
        elif self.knowledge_retriever:
            # Fallback: KB-only retrieval (web disabled in this path)
            basis = slide_text
            rag_items = self.retrieve_technical_knowledge(basis, slide_number)

        if rag_items:
            snippets = []
            for item in rag_items[: self.max_rag_items]:
                # Prefer QA answer if present
                md = getattr(item, "metadata", {}) or {}
                qa = md.get("qa") if isinstance(md, dict) else None
                if isinstance(qa, dict):
                    ans = qa.get("answer") or qa.get("a")
                    if isinstance(ans, str) and ans.strip():
                        s = " ".join(ans.splitlines())[:250].strip()
                        if s:
                            snippets.append(s)
                            continue
                content = getattr(item, "content", str(item)) or ""
                s = " ".join(content.splitlines())[:250].strip()
                if s:
                    snippets.append(s)
            technical_knowledge = " | ".join(snippets)

        # --- Build the strict prompt ---
        # Prepare a compact representation of verified facts to prevent omissions
        facts_block = "\n".join(f"- {f}" for f in verified_facts) if verified_facts else "(no discrete facts provided)"

        # Positional guidance for introduction/conclusion handling
        positional_guidance = ""
        if slide_position == "first":
            positional_guidance = (
                "OPENING: Begin with a brief, friendly welcome and a one-sentence setup of what this talk is about. "
                "Do not invent facts beyond the slide. Keep it under 30–40 words for the opening tone before covering the slide content."
            )
        elif slide_position == "last":
            positional_guidance = (
                "CLOSING: End with a concise wrap-up that ties the slide back to the overall talk, and include a brief thank-you. "
                "Do not add new facts. Keep closing under 40 words."
            )
        else:
            positional_guidance = (
                "MIDDLE: Do NOT include introductions or conclusions. Continue seamlessly from prior slides and focus only on this slide's points."
            )

        # Optional style reference from notes (trimmed)
        style_reference = ""
        if unpolished_notes and len(unpolished_notes.split()) >= 12:
            style_reference = (" ".join(unpolished_notes.splitlines())).strip()[:600]

        prompt = f"""
SYSTEM: You are a professional presenter and transcript writer.
Your goal is to create a natural, human-sounding spoken transcript based strictly on the ORIGINAL SLIDE CONTENT.
The transcript should sound smooth, conversational, and confident—like a real speaker presenting the slide—not like a reading of text.

PRIMARY SOURCE:
Use ONLY the ORIGINAL SLIDE CONTENT below as your factual foundation.
You may lightly rephrase for flow and clarity, but never add or invent new information.

SECONDARY CONTEXT (for clarification only, not for adding facts):
{technical_knowledge if technical_knowledge else "None"}

PREVIOUS SLIDES (for continuity only):
{context if context else "None"}

SLIDE DETAILS:
- Position: {slide_position}
- Type: {slide_type}
- Language: {self.target_language.upper() if self.target_language else "ENGLISH"}

POSITIONAL GUIDANCE:
{positional_guidance}

STYLE REFERENCE (tone only; do not add facts):
{style_reference if style_reference else "None"}

SPEAKING STYLE GUIDELINES:
1. Sound natural and conversational — as if explaining the slide out loud.
2. Use short sentences, pauses, and connecting words like “so,” “now,” or “let’s look at…”.
3. Lightly rephrase technical points for clarity, but do not add new examples, numbers, or facts.
4. If you need to explain a term, keep it under 15 words and stay neutral.
5. Maintain the same structure and order as the slide.
6. Keep within these word limits:
   - title <= 30 words
   - content <= 200 words
   - thank-you/Q&A <= 50 words
7. Avoid robotic or academic phrasing — this is meant for spoken delivery.
8. Do not use bullet lists or headings; output one continuous spoken paragraph only.
9. MUST include every verified fact; do not remove any verified content.

NATURAL SPEECH TIPS:
- Prefer everyday phrasing and contractions (e.g., “we’re”, “it’s”).
- Vary sentence length; use light rhetorical cues (e.g., “here’s why”, “let’s connect this”).
- Keep a steady rhythm and clear transitions without sounding like a list.

CONFERENCE/MEETING PRESENTER TONE:
- Confident but approachable; speak to an audience in a room.
- Use audience anchors like “what this means” or “the key takeaway”.
- Emphasize implications briefly without inventing new facts.

STORYTELLING FRAMEWORK (without adding facts):
- Anchor: one short clause that orients the audience to the slide’s focus.
- Progression: present points in slide order with crisp transitions (“first…”, “then…”, “importantly…”).
- Takeaway: one short closing clause that reinforces the main point of this slide.

FACT PRESENTATION RULE:
If any fact is provided in Q/A form (e.g., "Q: ... A: ..."), include only the answer content in your speech; the question text is for your reference, not to be spoken.

ORIGINAL SLIDE CONTENT:
{slide_text}

VERIFIED FACTS (must all be preserved, paraphrasing allowed but no omissions):
{facts_block}

Now, write the final transcript as if it were being spoken by a confident, engaging presenter.

BEGIN TRANSCRIPT:
"""

        # LLM generation
        raw_out = self.llm_client.generate(prompt, self.model)

        # Optional semantic check
        if enforce_semantic_check and _SEM_MODEL:
            sim = self.semantic_similarity(slide_text, raw_out)
            if sim is not None and sim < 0.55:
                logger.warning(f"Low semantic similarity ({sim:.2f}) between slide and transcript.")
                # Slight repair prompt: ask LLM to strictly reduce to slide content only
                repair_prompt = f"""
You produced a transcript that is semantically distant from the slide content (similarity={sim:.2f}).
Please re-generate a transcript that STRICTLY uses only the ORIGINAL SLIDE CONTENT below, follows the same order, and obeys previous rules.
ORIGINAL SLIDE CONTENT:
{slide_text}
BEGIN TRANSCRIPT:
"""
                raw_out = self.llm_client.generate(repair_prompt, self.model)

        final_out = raw_out.strip()

        # Coverage pass: ensure all verified facts are present (simple containment heuristic)
        if verified_facts:
            missing = []
            lowered = final_out.lower()
            for fact in verified_facts:
                # Use a light heuristic: check presence of key tokens from the fact
                # If fact contains Q/A, derive tokens from the answer part only
                answer_only = fact
                m = re.search(r"A[:：]\s*(.*)$", fact, flags=re.IGNORECASE | re.DOTALL)
                if m:
                    answer_only = m.group(1).strip()
                tokens = [t for t in answer_only.lower().replace("\n", " ").split() if len(t) > 3]
                if tokens and not all(t in lowered for t in tokens[: max(1, min(3, len(tokens))) ]):
                    missing.append(fact)

            if missing:
                logger.info(f"Coverage repair: {len(missing)} verified facts missing; requesting inclusion.")
                repair_prompt = f"""
You must produce a single-paragraph spoken transcript that preserves ALL verified facts below.
Do not add new information. Keep it natural, ordered like the original slide, and under the word limits.

ALREADY GENERATED DRAFT (improve by inserting the missing facts smoothly):
{final_out}

ORIGINAL SLIDE CONTENT (ground truth):
{slide_text}

MISSING VERIFIED FACTS (must be incorporated):
{"\n".join(f"- {m}" for m in missing)}

BEGIN REVISED TRANSCRIPT:
"""
                final_out = self.llm_client.generate(repair_prompt, self.model).strip()

        return final_out


class TranscriptReviewer:
    """Reviews transcripts for hallucinations and prepares a polished, slide-faithful revised transcript."""

    def __init__(self, llm_client, model: Optional[str] = None, strict_mode: bool = True):
        """
        strict_mode: if True, do not allow paraphrases that imply new facts.
        """
        self.llm_client = llm_client
        self.model = model
        self.strict_mode = strict_mode

    def _safe_json_parse(self, text: str) -> Dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse reviewer LLM JSON output: {e}")
            return {}

    def review_transcript(self, slide_info: Any, transcript: str) -> Dict:
        """
        Returns a JSON-like dict with fields:
          - assessment: {is_factually_accurate, follows_slide_order, is_presentation_ready}
          - presentation_quality_score: int
          - detailed_critique: {unsupported_claims, omissions, structural_issues, style_issues}
          - revised_transcript: str
        """

        # Extract ground truth and metadata (use original slide text)
        ground_content = ""
        verified_facts = []
        slide_position = "unknown"
        slide_type = "unknown"
        unpolished_notes = ""

        if slide_info:
            if isinstance(slide_info, dict):
                ground_content = slide_info.get("original_content", "")
                verified_facts = slide_info.get("facts", [])
                slide_position = slide_info.get("position", slide_position)
                slide_type = slide_info.get("type", slide_type)
                unpolished_notes = slide_info.get("unpolished_notes", "")
            else:
                ground_content = getattr(slide_info, "original_content", "")
                verified_facts = getattr(slide_info, "facts", [])
                slide_position = getattr(slide_info, "position", slide_position)
                slide_type = getattr(slide_info, "type", slide_type)
                unpolished_notes = getattr(slide_info, "unpolished_notes", "")

        facts_text = "\n".join(f"- {f}" for f in verified_facts) if verified_facts else ground_content

        # Build review prompt
        prompt = f"""
ROLE: Factuality and Consistency Auditor.

ORIGINAL SLIDE CONTENT (ground truth):
{ground_content}

VERIFIED FACTS (if any):
{facts_text}

GENERATED TRANSCRIPT:
{transcript}

SLIDE METADATA:
- position: {slide_position}
- type: {slide_type}

TASK:
1) Identify any claims in the GENERATED TRANSCRIPT that are NOT supported by the ORIGINAL SLIDE CONTENT.
   For each, provide the claim text and a short explanation why it is unsupported.
2) Identify any important points present in the ORIGINAL SLIDE CONTENT that are missing from the transcript (omissions).
3) Confirm whether the transcript follows the same order as the slide content.
4) Provide a short presentation_quality_score (1-10) and list style issues if present.
5) Produce a revised_transcript that:
   - Is presentation-ready and natural sounding for speech.
   - Is 100% grounded in the ORIGINAL SLIDE CONTENT (no added facts).
   - Includes ALL verified facts (do not omit any verified content).
   - Keeps the same sequence as the slide.
   - Obeys the word limits (title/thank-you/content).
   - Is ONE coherent paragraph with smooth transitions, contractions where natural, and varied sentence length (no bullets or headings).
   - Is ONE coherent paragraph; no bullets, numbering, or headings.

OUTPUT:
Return a single valid JSON object with keys:
- assessment: {{ is_factually_accurate, follows_slide_order, is_presentation_ready }}
- presentation_quality_score: int
- detailed_critique: {{ unsupported_claims: [...], omissions: [...], structural_issues: [...], style_issues: [...] }}
- revised_transcript: "<text>"
"""

        raw = self.llm_client.generate(prompt, self.model)

        review = self._safe_json_parse(raw)
        if not review:
            # Fallback conservative review if parsing failed
            review = {
                "assessment": {
                    "is_factually_accurate": False,
                    "follows_slide_order": False,
                    "is_presentation_ready": False,
                },
                "presentation_quality_score": 3,
                "detailed_critique": {
                    "unsupported_claims": [],
                    "omissions": [],
                    "structural_issues": [],
                    "style_issues": [{"issue": "Failed to parse LLM reviewer output; returned raw text snippet", "example": raw[:300]}],
                },
                # Best-effort revised transcript: clamp down to original slide content (shorten if needed)
                "revised_transcript": self._force_clamp_to_content(transcript, ground_content, slide_type),
            }

        # Ensure revised transcript is coherent speech, not a fact list
        revised = review.get("revised_transcript", "").strip() or transcript
        if self._looks_list_like(revised):
                polish_prompt = f"""
Rewrite the transcript into ONE coherent, natural-sounding spoken paragraph.
- No bullets, numbering, or headings.
- Use smooth transitions; avoid list-like phrasing.
- Stay strictly grounded in the ORIGINAL SLIDE CONTENT; do not add facts.
- Aim for a confident conference/meeting presenter tone.
- Use the storytelling framework: anchor → progression → takeaway.

ORIGINAL SLIDE CONTENT:
{ground_content}

DRAFTED NOTES (tone/style reference only):
{getattr(slide_info, 'unpolished_notes', '') if slide_info else ''}

DRAFT TO POLISH:
{revised}

OUTPUT: Single paragraph only.
"""
            try:
                polished = self.llm_client.generate(polish_prompt, self.model).strip()
                if polished:
                    review["revised_transcript"] = polished
            except Exception:
                # keep the existing revised transcript on failure
                pass

        # Always run a gentle style polish to enhance natural delivery while preserving content
        try:
            revised_now = review.get("revised_transcript", revised)
            style_polish_prompt = f"""
Polish the transcript into a natural, conversational paragraph suitable for speaking.
- Use contractions and smooth transitions.
- Vary sentence length; avoid list-like cadence.
- Keep facts, ordering, and meaning exactly the same.
- Do NOT add new facts beyond the ORIGINAL SLIDE CONTENT and VERIFIED FACTS.
- Aim for a confident conference/meeting presenter tone.
- Use the storytelling framework: brief anchor → ordered progression → one-sentence takeaway.

ORIGINAL SLIDE CONTENT:
{ground_content}

VERIFIED FACTS (for coverage; do not list them verbatim):
{"\n".join(f"- {f}" for f in verified_facts) if verified_facts else "(none)"}

STYLE REFERENCE (tone only):
{unpolished_notes if unpolished_notes else "(none)"}

DRAFT TO POLISH:
{revised_now}

OUTPUT: Single coherent paragraph.
"""
            polished2 = self.llm_client.generate(style_polish_prompt, self.model).strip()
            if polished2:
                review["revised_transcript"] = polished2
        except Exception:
            pass

        return review

    def _looks_list_like(self, text: str) -> bool:
        if not text:
            return False
        # Common bullet/numbered list patterns
        bullet_pat = re.compile(r"(^|\n)\s*(?:[-•\u2022\*]|\d+[.)])\s+", re.MULTILINE)
        if bullet_pat.search(text):
            return True
        # Excessive short line breaks suggest lists
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(lines) >= 3 and sum(1 for ln in lines if len(ln) <= 60) >= len(lines) - 1:
            return True
        return False

    def _force_clamp_to_content(self, transcript: str, content: str, slide_type: str) -> str:
        """
        Conservative fallback: produce a very short transcript that only reads the original content.
        This aims to avoid hallucinations in failure modes.
        """
        # If content is short enough, return it verbatim (trim to word limit)
        if not content:
            return transcript

        word_limit = 200
        if slide_type in ("title",):
            word_limit = 30
        elif slide_type in ("thank_you", "q_and_a", "q&a"):
            word_limit = 50

        words = content.split()
        if len(words) <= word_limit:
            # Make it speakable: a minimal transformation for oral delivery
            short = content.strip()
            # Replace newlines with short pauses (commas) for flow
            short = " ".join(short.splitlines())
            return short
        else:
            return " ".join(words[:word_limit]).strip() + "..."
