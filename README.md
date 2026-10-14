# NLP Text Analyzer

A **hybrid NLP web application** powered by:

- 🚀 **spaCy** (`en_core_web_trf`) — GPU-accelerated transformer model for structural NLP
- 📊 **NLTK** — Real n-gram computation with Lidstone smoothing
- 🧠 **Google Gemini** — Semantic reasoning, spell checking, normalization

An end-to-end Natural Language Processing (NLP) web orchestrator built with **Flask**, **spaCy**, **NLTK**, and **Google Gemini API**.

The system implements a **hybrid architecture** that divides responsibilities based on algorithmic efficiency and model capabilities:

- **Local Structural NLP Engine (spaCy & NLTK):** Handles deterministic linguistic tasks locally with high throughput—including word/sentence tokenization, Penn Treebank POS tagging, Named Entity Recognition (NER), syntactic dependency trees (visualized with SVG), and $N$-gram frequency estimation with Lidstone/Laplace smoothing ($\alpha$).
- **Generative AI Semantic Layer (Google Gemini API):** Powers context-aware spelling & grammar correction, rule-based text normalization (contraction/abbreviation expansion), sentiment polarity scoring, and discourse-level summarization.

### Key Highlights:

- ⚡ **High Efficiency:** Structural parsing executes locally with millisecond latency; LLM calls are reserved for high-order semantic reasoning.
- 🔄 **Resilient API Integration:** Built on the Google GenAI SDK with automated multi-model fallback (`gemini-3.5-flash-lite` → `gemini-3.8-flash`).
- 📊 **Curriculum Aligned:** Covers fundamental NLP syllabus topics from Finite State concepts and tokenization to smoothing algorithms and Context-Free / Dependency parsing.

## Features

| Task                         | Engine      | Syllabus Topics Covered                                          |
| ---------------------------- | ----------- | ---------------------------------------------------------------- |
| **Tokenization**             | spaCy (GPU) | Words & sentence tokenization, segmentation                      |
| **POS Tagging**              | spaCy (GPU) | Penn Treebank tagsets, English word classes                      |
| **Named Entity Recognition** | spaCy (GPU) | Part-of-Speech tagging, case study                               |
| **Dependency Parsing**       | spaCy (GPU) | Dependency grammar, constituency, parsing with CFG               |
| **N-Gram Analysis**          | NLTK        | N-grams, unsmoothed/smoothed N-grams, Lidstone/Laplace smoothing |
| **Spell Checking**           | Gemini      | Detecting & correcting spelling errors, normalizing text         |
| **Normalization**            | Gemini      | Normalizing text, ambiguity resolution                           |
| **Full Analysis**            | All three   | Everything above + sentiment, discourse type, complexity         |

## Project Structure

```
nlp-analyzer/
├── app.py            # Flask backend — REST API endpoints (hybrid architecture)
├── nlp_local.py      # Local NLP: spaCy (GPU) + NLTK processing
├── prompts.py        # LLM prompts (evaluation criterion)
├── config.py         # Configuration (API key, model, spaCy model, limits)
├── setup.py          # One-time setup: downloads spaCy model + NLTK data
├── requirements.txt  # Python dependencies
└── templates/
    └── index.html    # Single-page frontend with interactive visualizations
```

## Architecture

```
Browser
  │
  ├─POST─▶ Flask (app.py)
  │              │
  │    ┌─────────┴──────────┐
  │    ▼                    ▼
  │  nlp_local.py        Gemini API
  │  (spaCy + NLTK)      (prompts.py)
  │  RTX 3050 GPU        spell/normalize/semantic
  │    │                    │
  │    └─────────┬──────────┘
  │              │ merged JSON
  └◀─────────────┘
```

**Task routing:**

- Tokenize / POS / NER / Dependency → spaCy `en_core_web_trf` (CUDA)
- N-Grams → NLTK with Lidstone smoothing
- Spell Check / Normalize → Gemini (context-aware)
- Full Analysis → all three merged

## Setup (First Time)

### 1. Get a Gemini API Key (free)

[https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

### 2. Set your API key

Create or edit your `.env` file (copy from `.env.example`):

```bash
cp .env.example .env
```

And set your key inside `.env`:

```ini
GEMINI_API_KEY=your_actual_api_key_here
```

Or set the environment variable directly:

```powershell
$env:GEMINI_API_KEY = "your-key-here"
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> For GPU support (RTX 3050), also install:
>
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
> ```

### 4. Download spaCy model + NLTK data

```bash
python setup.py
```

This downloads `en_core_web_trf` (~400MB, transformer-based, GPU-enabled) and all NLTK corpora.

### 5. Run

```bash
python app.py
```

Open **http://localhost:5000**

## API Reference

| Method | Endpoint          | Engine | Body               |
| ------ | ----------------- | ------ | ------------------ |
| GET    | `/api/device`     | —      | Device/GPU info    |
| POST   | `/api/tokenize`   | spaCy  | `{text}`           |
| POST   | `/api/pos`        | spaCy  | `{text}`           |
| POST   | `/api/ner`        | spaCy  | `{text}`           |
| POST   | `/api/dependency` | spaCy  | `{text}`           |
| POST   | `/api/ngrams`     | NLTK   | `{text, n, alpha}` |
| POST   | `/api/spellcheck` | Gemini | `{text}`           |
| POST   | `/api/normalize`  | Gemini | `{text}`           |
| POST   | `/api/analyze`    | All    | `{text}`           |

## Configuration (`config.py`)

| Key               | Default            | Description                   |
| ----------------- | ------------------ | ----------------------------- |
| `GEMINI_API_KEY`  | env var            | Your Gemini API key           |
| `GEMINI_MODEL`    | `gemini-1.5-flash` | Gemini model                  |
| `SPACY_MODEL`     | `en_core_web_trf`  | spaCy model (trf=GPU, sm=CPU) |
| `MAX_INPUT_CHARS` | `3000`             | Input character limit         |
| `SMOOTHING_ALPHA` | `0.5`              | Lidstone α (1.0 = Laplace)    |

## Example Input (for NER)

```
Barack Obama visited Google's headquarters in Mountain View, California
on January 15th, 2024. He didn't think the company's NLP models were
progressing fast enough compared to OpenAI's GPT-4.
```

Expected entities: `Barack Obama` (PERSON), `Google` (ORG), `Mountain View` (GPE),
`California` (GPE), `January 15th, 2024` (DATE), `OpenAI` (ORG), `GPT-4` (PRODUCT)
