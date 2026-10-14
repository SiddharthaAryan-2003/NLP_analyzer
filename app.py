# app.py — NLP Text Analyzer (Flask Backend)
# -------------------------------------------
# Hybrid NLP system:
#   • spaCy (GPU via RTX 3050) → tokenization, POS, NER, dependency parsing
#   • NLTK                     → n-grams, frequency analysis, smoothing
#   • Google Gemini API        → spell check, normalization, semantic analysis
#
# Architecture:
#   Browser ──POST──▶ Flask endpoint ──▶ nlp_local.py (spaCy/NLTK)
#                                    └──▶ Gemini API (prompts.py)
#                  ◀── JSON ──────────── merged results

import json
import re
from google import genai
from google.genai import types
from flask import Flask, request, jsonify, render_template

from config import (
    GEMINI_API_KEY, GEMINI_MODEL,
    HOST, PORT, DEBUG,
    MAX_INPUT_CHARS, DEFAULT_NGRAM_N, SMOOTHING_ALPHA,
)
from prompts import (
    SYSTEM_PROMPT,
    SPELL_CHECK_PROMPT,
    NORMALIZE_PROMPT,
    AI_ANALYSIS_PROMPT,
)
from nlp_local import (
    get_device_info,
    local_tokenize,
    local_pos_tag,
    local_ner,
    local_dependency,
    local_ngrams,
)

# ── App & Gemini setup ────────────────────────────────────────────────────────
app = Flask(__name__)
client = genai.Client(api_key=GEMINI_API_KEY)


# ── Helpers ───────────────────────────────────────────────────────────────────

def call_gemini(prompt: str) -> dict:
    """Send a prompt to Gemini and return parsed JSON."""
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        raise ValueError(
            "Gemini API key is not configured. Please set GEMINI_API_KEY in config.py "
            "or set the environment variable $env:GEMINI_API_KEY. "
            "Get a free key at https://aistudio.google.com/app/apikey"
        )
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
            ),
        )
        raw = response.text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$",          "", raw)
        return json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"Gemini API call failed: {e}")


def validate_input(text: str):
    """Return (error_message, status_code) or (None, None) if valid."""
    if not text or not text.strip():
        return "Text input is required.", 400
    if len(text) > MAX_INPUT_CHARS:
        return f"Input exceeds {MAX_INPUT_CHARS} character limit.", 400
    return None, None


# ── System info ───────────────────────────────────────────────────────────────

@app.route("/api/device")
def device_info():
    """Return info about the compute device (GPU/CPU)."""
    return jsonify(get_device_info())


# ── Frontend ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── Local NLP Endpoints (spaCy / NLTK) ───────────────────────────────────────

@app.route("/api/tokenize", methods=["POST"])
def tokenize():
    """Tokenize text using spaCy (GPU-accelerated)."""
    data = request.get_json()
    text = data.get("text", "")
    err, code = validate_input(text)
    if err:
        return jsonify({"error": err}), code

    result = local_tokenize(text)
    return jsonify(result)


@app.route("/api/pos", methods=["POST"])
def pos_tag():
    """POS tagging using spaCy (Penn Treebank tagset)."""
    data = request.get_json()
    text = data.get("text", "")
    err, code = validate_input(text)
    if err:
        return jsonify({"error": err}), code

    result = local_pos_tag(text)
    return jsonify(result)


@app.route("/api/ner", methods=["POST"])
def named_entities():
    """Named Entity Recognition using spaCy."""
    data = request.get_json()
    text = data.get("text", "")
    err, code = validate_input(text)
    if err:
        return jsonify({"error": err}), code

    result = local_ner(text)
    return jsonify(result)


@app.route("/api/dependency", methods=["POST"])
def dependency():
    """Dependency parsing using spaCy. Returns relations + SVG visualization."""
    data = request.get_json()
    text = data.get("text", "")
    err, code = validate_input(text)
    if err:
        return jsonify({"error": err}), code

    result = local_dependency(text)
    return jsonify(result)


@app.route("/api/ngrams", methods=["POST"])
def ngrams():
    """N-gram analysis with Lidstone smoothing using NLTK."""
    data  = request.get_json()
    text  = data.get("text", "")
    n     = int(data.get("n", DEFAULT_NGRAM_N))
    alpha = float(data.get("alpha", SMOOTHING_ALPHA))
    err, code = validate_input(text)
    if err:
        return jsonify({"error": err}), code
    if n < 1 or n > 5:
        return jsonify({"error": "n must be between 1 and 5."}), 400
    if not (0 < alpha <= 1):
        return jsonify({"error": "alpha must be between 0 (exclusive) and 1."}), 400

    result = local_ngrams(text, n, alpha)
    return jsonify(result)


# ── Gemini-Powered Endpoints ──────────────────────────────────────────────────

@app.route("/api/spellcheck", methods=["POST"])
def spellcheck():
    """Context-aware spell & grammar check using Gemini."""
    data = request.get_json()
    text = data.get("text", "")
    err, code = validate_input(text)
    if err:
        return jsonify({"error": err}), code

    try:
        result = call_gemini(SPELL_CHECK_PROMPT.format(text=text))
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/normalize", methods=["POST"])
def normalize():
    """Text normalization (contractions, abbreviations, etc.) using Gemini."""
    data = request.get_json()
    text = data.get("text", "")
    err, code = validate_input(text)
    if err:
        return jsonify({"error": err}), code

    try:
        result = call_gemini(NORMALIZE_PROMPT.format(text=text))
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ── Full Hybrid Analysis ──────────────────────────────────────────────────────

@app.route("/api/analyze", methods=["POST"])
def full_analyze():
    """
    Hybrid full analysis:
      • spaCy  → tokenization, POS tags, NER, dependency relations
      • NLTK   → bigrams with smoothing
      • Gemini → sentiment, topics, key phrases, normalization, semantic insights
    """
    data = request.get_json()
    text = data.get("text", "")
    err, code = validate_input(text)
    if err:
        return jsonify({"error": err}), code

    # Run local processing first (fast)
    tok_result  = local_tokenize(text)
    pos_result  = local_pos_tag(text)
    ner_result  = local_ner(text)
    gram_result = local_ngrams(text, n=2)

    # Then call Gemini for semantic/reasoning layer
    try:
        ai_result = call_gemini(AI_ANALYSIS_PROMPT.format(text=text))
    except Exception as e:
        # Graceful fallback: return local results even if Gemini key is not set
        ai_result = {
            "sentiment": {"label": "neutral", "score": 0.0, "reasoning": f"Gemini API offline: {e}"},
            "topics": ["(Requires configured Gemini API key)"],
            "key_phrases": [],
            "discourse_type": "informative",
            "complexity": {"level": "standard", "factors": ["local analysis only"]},
            "spelling_issues": [],
            "normalized_text": text.lower().strip(),
            "summary": "Local analysis complete. Configure Gemini API key for AI summary & semantic insights."
        }

    # Merge into a single comprehensive response
    return jsonify({
        "summary": {
            "word_count":        tok_result["word_count"],
            "sentence_count":    tok_result["sentence_count"],
            "unique_words":      tok_result["unique_words"],
            "avg_word_length":   tok_result["avg_word_length"],
            "lexical_diversity": tok_result["lexical_diversity"],
            "stop_word_count":   tok_result["stop_word_count"],
        },
        "tokens":          tok_result["tokens"],
        "sentences":       tok_result["sentences"],
        "pos_tags":        pos_result["tagged"],
        "pos_summary":     pos_result["pos_summary"],
        "entities":        ner_result["entities"],
        "entity_types":    ner_result["entity_types"],
        "annotated":       ner_result["annotated"],
        "bigrams":         gram_result.get("top_10", []),
        "sentiment":       ai_result.get("sentiment", {}),
        "topics":          ai_result.get("topics", []),
        "key_phrases":     ai_result.get("key_phrases", []),
        "discourse_type":  ai_result.get("discourse_type", ""),
        "complexity":      ai_result.get("complexity", {}),
        "spelling_issues": ai_result.get("spelling_issues", []),
        "normalized_text": ai_result.get("normalized_text", ""),
        "summary_text":    ai_result.get("summary", ""),
        "engines": {
            "structural": tok_result["engine"],
            "semantic":   f"Google {GEMINI_MODEL}",
        },
    })


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=DEBUG)
