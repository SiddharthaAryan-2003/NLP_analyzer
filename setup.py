# -*- coding: utf-8 -*-
# setup.py -- One-time setup script for NLP Text Analyzer
# --------------------------------------------------------
# Run this ONCE before starting the app:
#   python setup.py
#
# What it does:
#   1. Downloads NLTK corpora (punkt, stopwords, etc.)
#   2. Downloads the spaCy transformer model (GPU-enabled)
#   3. Verifies CUDA availability for your RTX 3050
#   4. Runs a quick sanity check on both libraries

import sys
import subprocess


def run(cmd, desc):
    print("\n" + "-" * 50)
    print(">>  " + desc)
    print("-" * 50)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print("WARNING: '{}' exited with code {}".format(" ".join(cmd), result.returncode))
    return result.returncode == 0


def main():
    print("=" * 50)
    print("  NLP Text Analyzer -- Setup")
    print("=" * 50)

    # 1. NLTK data
    print("\n[1/3] Downloading NLTK data...")
    import nltk
    for pkg in ["punkt", "punkt_tab", "stopwords",
                "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
        nltk.download(pkg, quiet=False)

    # 2. spaCy transformer model
    print("\n[2/3] Downloading spaCy model (en_core_web_trf)...")
    print("      ~400MB transformer model -- GPU-accelerated on your RTX 3050!")
    ok = run(
        [sys.executable, "-m", "spacy", "download", "en_core_web_trf"],
        "spacy download en_core_web_trf"
    )
    if not ok:
        print("      Falling back to en_core_web_sm...")
        run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
            "spacy download en_core_web_sm")

    # 3. Sanity check
    print("\n[3/3] Running sanity check...")
    try:
        import torch
        cuda = torch.cuda.is_available()
        gpu  = torch.cuda.get_device_name(0) if cuda else "None"
        print("  [OK] PyTorch: " + torch.__version__)
        print("  [{}] CUDA: {}".format("OK" if cuda else "!!",
                                       "available" if cuda else "NOT available"))
        print("  GPU: " + gpu)
    except ImportError:
        print("  [!!] PyTorch not installed -- spaCy will use CPU only.")
        print("       For GPU support run:")
        print("       pip install torch --index-url https://download.pytorch.org/whl/cu121")

    try:
        import spacy
        spacy.prefer_gpu()
        loaded = None
        for model_name in ["en_core_web_trf", "en_core_web_sm"]:
            try:
                nlp = spacy.load(model_name)
                doc = nlp("Natural language processing is fascinating.")
                tokens = [t.text for t in doc]
                print("  [OK] spaCy model: " + model_name)
                print("  [OK] Test tokens: " + str(tokens))
                loaded = model_name
                break
            except OSError:
                continue
        if not loaded:
            print("  [!!] No spaCy model found -- check the download above.")
    except Exception as e:
        print("  [ERROR] spaCy: " + str(e))

    try:
        from nltk.util import ngrams
        from nltk.tokenize import word_tokenize
        test = list(ngrams(word_tokenize("hello world test sentence"), 2))
        print("  [OK] NLTK n-grams: " + str(test))
    except Exception as e:
        print("  [ERROR] NLTK: " + str(e))

    print("\n" + "=" * 50)
    print("  Setup complete!")
    print("  Run:  python app.py")
    print("  Open: http://localhost:5000")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
