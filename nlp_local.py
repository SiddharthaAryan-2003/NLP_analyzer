# nlp_local.py — Local NLP processing using spaCy + NLTK
# --------------------------------------------------------
# Handles all structural NLP tasks locally on your GPU (RTX 3050).
# spaCy's transformer model (en_core_web_trf) uses CUDA automatically
# when PyTorch detects a compatible GPU.

import spacy
try:
    import torch
    _cuda_available = torch.cuda.is_available()
    _device_name    = torch.cuda.get_device_name(0) if _cuda_available else "CPU"
except ImportError:
    torch = None
    _cuda_available = False
    _device_name    = "CPU (install PyTorch with CUDA for RTX 3050 acceleration)"

import nltk
from nltk import FreqDist
from nltk.util import ngrams as nltk_ngrams
from nltk.probability import LidstoneProbDist, FreqDist as NLTKFreqDist
from collections import Counter
from config import SPACY_MODEL, SMOOTHING_ALPHA

# ── GPU detection ─────────────────────────────────────────────────────────────
if _cuda_available:
    spacy.prefer_gpu()   # tells spaCy to use CUDA

# ── Download NLTK data (one-time) ─────────────────────────────────────────────
for _pkg in ["punkt", "punkt_tab", "stopwords", "averaged_perceptron_tagger"]:
    nltk.download(_pkg, quiet=True)

# ── Load spaCy model ──────────────────────────────────────────────────────────
try:
    nlp = spacy.load(SPACY_MODEL)
except OSError:
    # Fallback: try md, lg then sm if preferred model not found
    for _fallback in ["en_core_web_md", "en_core_web_lg", "en_core_web_sm"]:
        try:
            nlp = spacy.load(_fallback)
            break
        except OSError:
            continue
    else:
        raise RuntimeError(
            "No spaCy English model found. Run: python setup.py"
        )

# ── NER label descriptions ────────────────────────────────────────────────────
NER_DESCRIPTIONS = {
    "PERSON":   "Person", "ORG": "Organization", "GPE": "Country/City/State",
    "LOC":      "Location", "DATE": "Date", "TIME": "Time",
    "MONEY":    "Monetary value", "PERCENT": "Percentage",
    "PRODUCT":  "Product", "EVENT": "Event", "WORK_OF_ART": "Work of art",
    "LAW":      "Law/regulation", "LANGUAGE": "Language",
    "NORP":     "Nationality/Religion/Group", "FAC": "Facility",
    "CARDINAL": "Number", "ORDINAL": "Ordinal number", "QUANTITY": "Quantity",
}

# ── POS full names (Penn Treebank) ────────────────────────────────────────────
POS_NAMES = {
    "CC":"Coordinating conjunction","CD":"Cardinal number","DT":"Determiner",
    "EX":"Existential there","FW":"Foreign word","IN":"Preposition/subord. conj.",
    "JJ":"Adjective","JJR":"Adjective, comparative","JJS":"Adjective, superlative",
    "LS":"List item marker","MD":"Modal","NN":"Noun, singular/mass",
    "NNS":"Noun, plural","NNP":"Proper noun, singular","NNPS":"Proper noun, plural",
    "PDT":"Predeterminer","POS":"Possessive ending","PRP":"Personal pronoun",
    "PRP$":"Possessive pronoun","RB":"Adverb","RBR":"Adverb, comparative",
    "RBS":"Adverb, superlative","RP":"Particle","SYM":"Symbol","TO":"to",
    "UH":"Interjection","VB":"Verb, base form","VBD":"Verb, past tense",
    "VBG":"Verb, gerund","VBN":"Verb, past participle",
    "VBP":"Verb, non-3rd person singular","VBZ":"Verb, 3rd person singular",
    "WDT":"Wh-determiner","WP":"Wh-pronoun","WP$":"Possessive wh-pronoun",
    "WRB":"Wh-adverb","``":"Opening quote","''":"Closing quote",
    ".":"Punctuation","," :"Comma",":":"Colon","(":"Left bracket",")":"Right bracket",
    "$":"Dollar sign","#":"Pound sign","HYPH":"Hyphen","NFP":"Superfluous punct.",
    "-LRB-":"Left bracket","-RRB-":"Right bracket","ADD":"Email/URL",
    "GW":"Filler/hesitation","XX":"Unknown","SP":"Space","_SP":"Space",
}


def get_device_info() -> dict:
    """Return info about available compute devices."""
    return {
        "cuda_available": _cuda_available,
        "device": _device_name,
        "spacy_model": nlp.meta.get("name", SPACY_MODEL),
        "spacy_version": spacy.__version__,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tokenization
# ─────────────────────────────────────────────────────────────────────────────

def local_tokenize(text: str) -> dict:
    """Full tokenization with sentence segmentation using spaCy."""
    doc = nlp(text)

    tokens    = [t.text for t in doc]
    sentences = [s.text.strip() for s in doc.sents]
    words     = [t.text for t in doc if not t.is_punct and not t.is_space]
    unique_w  = set(w.lower() for w in words)

    return {
        "tokens":           tokens,
        "sentences":        sentences,
        "word_count":       len(words),
        "sentence_count":   len(sentences),
        "unique_words":     len(unique_w),
        "avg_word_length":  round(sum(len(w) for w in words) / max(len(words), 1), 2),
        "lexical_diversity": round(len(unique_w) / max(len(words), 1), 4),
        "punct_count":      sum(1 for t in doc if t.is_punct),
        "stop_word_count":  sum(1 for t in doc if t.is_stop),
        "engine":           "spaCy " + nlp.meta.get("name", ""),
    }


# ─────────────────────────────────────────────────────────────────────────────
# POS Tagging
# ─────────────────────────────────────────────────────────────────────────────

def local_pos_tag(text: str) -> dict:
    """POS tagging using spaCy (Penn Treebank tags via .tag_)."""
    doc = nlp(text)

    tagged = []
    pos_counts = Counter()
    for token in doc:
        if token.is_space:
            continue
        tag   = token.tag_ or token.pos_
        desc  = POS_NAMES.get(tag, token.pos_)
        tagged.append({
            "token":       token.text,
            "pos":         tag,
            "description": desc,
            "lemma":       token.lemma_,
            "is_stop":     token.is_stop,
        })
        pos_counts[tag] += 1

    return {
        "tagged":     tagged,
        "pos_summary": [{"tag": k, "description": POS_NAMES.get(k, k), "count": v}
                        for k, v in pos_counts.most_common()],
        "engine":     "spaCy " + nlp.meta.get("name", ""),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Named Entity Recognition
# ─────────────────────────────────────────────────────────────────────────────

def local_ner(text: str) -> dict:
    """Named Entity Recognition using spaCy."""
    doc = nlp(text)

    entities = []
    for ent in doc.ents:
        entities.append({
            "text":        ent.text,
            "label":       ent.label_,
            "description": NER_DESCRIPTIONS.get(ent.label_, ent.label_),
            "start":       ent.start_char,
            "end":         ent.end_char,
        })

    # Annotated text: wrap entities in markers for frontend highlighting
    annotated = []
    prev = 0
    for ent in doc.ents:
        if ent.start_char > prev:
            annotated.append({"type": "text", "text": text[prev:ent.start_char]})
        annotated.append({
            "type":  "entity",
            "text":  ent.text,
            "label": ent.label_,
            "desc":  NER_DESCRIPTIONS.get(ent.label_, ent.label_),
        })
        prev = ent.end_char
    if prev < len(text):
        annotated.append({"type": "text", "text": text[prev:]})

    entity_counts = Counter(e["label"] for e in entities)

    return {
        "entities":      entities,
        "entity_count":  len(entities),
        "entity_types":  [{"label": k, "description": NER_DESCRIPTIONS.get(k, k), "count": v}
                          for k, v in entity_counts.most_common()],
        "annotated":     annotated,
        "engine":        "spaCy " + nlp.meta.get("name", ""),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dependency Parsing
# ─────────────────────────────────────────────────────────────────────────────

def local_dependency(text: str) -> dict:
    """Dependency parse using spaCy. Returns token relations and SVG."""
    from spacy import displacy

    doc = nlp(text)

    # Limit to first sentence for readability
    first_sent = list(doc.sents)[0] if list(doc.sents) else doc

    deps = []
    for token in doc:
        if token.is_space:
            continue
        deps.append({
            "token":    token.text,
            "dep":      token.dep_,
            "head":     token.head.text,
            "head_pos": token.head.tag_,
            "children": [c.text for c in token.children],
        })

    # Generate SVG for first sentence only (keeps it readable)
    svg = displacy.render(first_sent, style="dep", jupyter=False,
                          options={"compact": True, "bg": "#0f172a",
                                   "color": "#e2e8f0", "font": "Segoe UI"})

    return {
        "dependencies":    deps,
        "svg":             svg,
        "note": "Dependency tree shown for first sentence only." if len(list(doc.sents)) > 1 else "",
        "engine":          "spaCy " + nlp.meta.get("name", ""),
    }


# ─────────────────────────────────────────────────────────────────────────────
# N-Gram Analysis with Smoothing
# ─────────────────────────────────────────────────────────────────────────────

def local_ngrams(text: str, n: int, alpha: float = SMOOTHING_ALPHA) -> dict:
    """
    Real n-gram computation using NLTK.
    Includes Lidstone (add-alpha) smoothing and raw frequency counts.
    Lidstone with alpha=1 is Laplace smoothing (covered in syllabus).
    """
    from nltk.tokenize import word_tokenize

    tokens = word_tokenize(text.lower())
    # Filter punctuation for cleaner n-grams
    words  = [t for t in tokens if t.isalpha()]

    if len(words) < n:
        return {"error": f"Text too short for {n}-grams. Need at least {n} words."}

    gram_list  = list(nltk_ngrams(words, n))
    gram_strs  = [" ".join(g) for g in gram_list]
    freq_dist  = FreqDist(gram_strs)

    # Lidstone smoothed probability
    vocab_size = len(set(gram_strs))
    total      = len(gram_strs)

    top_10 = [
        {
            "ngram":       gram,
            "count":       count,
            "probability": round(count / total, 6),
            "smoothed_prob": round((count + alpha) / (total + alpha * vocab_size), 6),
        }
        for gram, count in freq_dist.most_common(10)
    ]

    return {
        "n":              n,
        "ngrams":         gram_strs,
        "total_ngrams":   total,
        "unique_ngrams":  vocab_size,
        "top_10":         top_10,
        "smoothing":      f"Lidstone (α={alpha}) — equivalent to Laplace when α=1",
        "alpha":          alpha,
        "engine":         "NLTK",
    }
