# prompts.py — LLM Prompts for NLP Text Analyzer
# ------------------------------------------------
# These prompts are used ONLY for tasks where Gemini adds real value
# beyond local NLP libraries: spell checking, normalization, and
# intelligent analysis that requires reasoning/world knowledge.
# All structural NLP tasks (tokenization, POS, NER, n-grams, dependency
# parsing) are handled locally by spaCy + NLTK for accuracy and speed.

SYSTEM_PROMPT = """You are an expert NLP assistant. You analyze text and return
results in clean, valid JSON only — no markdown fences, no extra commentary.
Follow each task instruction precisely. Be thorough and accurate."""


# ── Spell Check ───────────────────────────────────────────────────────────────
# Used by: /api/spellcheck
# Why LLM: Context-aware correction that understands intent, not just
#           dictionary lookup. Can handle proper nouns, technical terms,
#           and grammar errors that spell-checkers miss.

SPELL_CHECK_PROMPT = """Detect and correct ALL spelling and grammatical errors in the text.
Use context to determine the intended word (e.g., "their/there/they're").
Return ONLY a JSON object with this exact structure:
{{
  "original": "the original text",
  "corrected": "the fully corrected text",
  "errors": [
    {{
      "original": "misspeled",
      "corrected": "misspelled",
      "error_type": "spelling|grammar|punctuation|word_choice",
      "explanation": "brief reason for correction"
    }},
    ...
  ],
  "error_count": <integer>,
  "overall_quality": "poor|fair|good|excellent"
}}

Text: {text}"""


# ── Text Normalization ────────────────────────────────────────────────────────
# Used by: /api/normalize
# Why LLM: Understands domain-specific abbreviations, slang, and context
#           far better than rule-based systems.

NORMALIZE_PROMPT = """Normalize the following text by applying ALL applicable transformations:
1. Expand contractions (e.g., don't → do not, I've → I have)
2. Expand abbreviations (e.g., NLP → Natural Language Processing, ASAP → as soon as possible)
3. Normalize numbers (e.g., 1st → first, $5 → five dollars — where appropriate)
4. Fix capitalization errors
5. Remove/standardize extra whitespace and punctuation
6. Convert informal language to formal (e.g., gonna → going to, wanna → want to)

Return ONLY a JSON object with this exact structure:
{{
  "original": "the original text",
  "normalized": "the normalized text",
  "changes": [
    {{
      "type": "contraction|abbreviation|number|capitalization|whitespace|informal",
      "original": "...",
      "normalized": "...",
      "position": <approximate word index>
    }},
    ...
  ],
  "change_count": <integer>,
  "formality_score": <integer 1-10, 10 being most formal>
}}

Text: {text}"""


# ── AI-Powered Full Analysis ──────────────────────────────────────────────────
# Used by: /api/analyze (AI portion only — local NLP runs separately)
# Why LLM: Semantic understanding, pragmatic analysis, discourse-level
#           insights that go beyond what spaCy can provide structurally.

AI_ANALYSIS_PROMPT = """Perform a deep semantic analysis of the following text.
The structural NLP (tokenization, POS, NER) has already been done locally.
Focus on what requires human-level understanding:

Return ONLY a JSON object with this exact structure:
{{
  "key_phrases": ["most important phrase", ...],
  "topics": ["main topic", "secondary topic", ...],
  "sentiment": {{
    "label": "positive|negative|neutral|mixed",
    "score": <float -1.0 to 1.0>,
    "reasoning": "brief explanation"
  }},
  "discourse_type": "narrative|argumentative|expository|descriptive|conversational",
  "complexity": {{
    "level": "simple|intermediate|advanced|expert",
    "factors": ["reason1", "reason2"]
  }},
  "spelling_issues": [
    {{"original": "misspeled", "corrected": "misspelled"}}
  ],
  "normalized_text": "normalized version of the text",
  "summary": "one sentence summary of what this text is about"
}}

Text: {text}"""
