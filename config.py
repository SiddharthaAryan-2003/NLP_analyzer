# config.py — Configuration for NLP Text Analyzer
# ------------------------------------------------
# Replace GEMINI_API_KEY with your actual key from:
# https://aistudio.google.com/app/apikey  (free tier available)

import os

# ── Gemini API ────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
GEMINI_MODEL   = "gemini-1.5-flash"          # fast & free-tier friendly

# ── spaCy ─────────────────────────────────────────────────────────────────────
# en_core_web_trf → transformer-based (requires PyTorch)
# en_core_web_md  → medium model with 300d GloVe word vectors & accurate NER/syntax
# en_core_web_sm  → small, fast
SPACY_MODEL    = "en_core_web_md"

# ── Server ────────────────────────────────────────────────────────────────────
HOST  = "0.0.0.0"
PORT  = 5000
DEBUG = True                                  # set False for production

# ── NLP defaults ─────────────────────────────────────────────────────────────
DEFAULT_NGRAM_N   = 2                         # bigrams by default
MAX_INPUT_CHARS   = 3000                      # increased for heavier workloads
SMOOTHING_ALPHA   = 0.5                       # Lidstone smoothing alpha (0<α≤1)
