# config.py — Configuration for NLP Text Analyzer
# ------------------------------------------------
# Configure your GEMINI_API_KEY in the .env file.
# Get a free API key at: https://aistudio.google.com/app/apikey

import os
from pathlib import Path

# ── Load environment variables from .env ──────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Built-in lightweight fallback parser if python-dotenv is not installed
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.is_file():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v

# ── Gemini API ────────────────────────────────────────────────────────────────
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL    = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
FALLBACK_MODELS = [GEMINI_MODEL, "gemini-3.8-flash", "gemini-flash-latest"]

# ── spaCy ─────────────────────────────────────────────────────────────────────
# en_core_web_trf → transformer-based (requires PyTorch)
# en_core_web_md  → medium model with 300d GloVe word vectors & accurate NER/syntax
# en_core_web_sm  → small, fast
SPACY_MODEL    = os.getenv("SPACY_MODEL", "en_core_web_md")

# ── Server ────────────────────────────────────────────────────────────────────
HOST  = os.getenv("HOST", "0.0.0.0")
PORT  = int(os.getenv("PORT", "5000"))
DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")

# ── NLP defaults ─────────────────────────────────────────────────────────────
DEFAULT_NGRAM_N   = int(os.getenv("DEFAULT_NGRAM_N", "2"))                         # bigrams by default
MAX_INPUT_CHARS   = int(os.getenv("MAX_INPUT_CHARS", "3000"))                      # increased for heavier workloads
SMOOTHING_ALPHA   = float(os.getenv("SMOOTHING_ALPHA", "0.5"))                     # Lidstone smoothing alpha (0<α≤1)
