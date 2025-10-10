# === NLP Pipeline for French Rice Health Bulletins ===
# Author: [Your Name]
# Language: English
# Purpose: Preprocess, analyze, and summarize French agricultural bulletins

# --- 1. Install dependencies (if not yet installed) ---
# !pip install clean-text spacy keybert transformers sentence-transformers langdetect

import re
from cleantext import clean
from langdetect import detect
import spacy
from keybert import KeyBERT
from transformers import pipeline

# --- 2. Load NLP models ---
# Load French spaCy model (run: python -m spacy download fr_core_news_md)
nlp = spacy.load("fr_core_news_sm")

# Keyword extractor (uses a multilingual BERT)
kw_model = KeyBERT(model='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Summarizer model (fine-tuned for French)
summarizer = pipeline("summarization", model="moussaKam/barthez-orangesum-abstract")


# --- 3. Preprocessing function ---
def preprocess_text(text: str) -> str:
    """Clean and prepare raw French text."""
    # Detect and check language
    lang = detect(text)
    if lang != 'fr':
        raise ValueError(f"Expected French text, but detected: {lang}")

    # Clean text
    text = clean(
        text,
        lower=True,
        no_urls=True,
        no_punct=False,
        no_emails=True,
        no_phone_numbers=True,
        no_currency_symbols=True
    )

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# --- 4. Thematic analysis function ---
def extract_keywords(text: str, top_n: int = 5):
    """Extract key thematic terms using KeyBERT."""
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words='french',
        top_n=top_n
    )
    return [kw for kw, _ in keywords]


# --- 5. Summarization function ---
def summarize_text(text: str, min_length: int = 50, max_length: int = 150) -> str:
    """Generate a concise summary using a French summarization model."""
    summary = summarizer(clean_text, max_new_tokens=60, do_sample=False)
    return summary[0]['summary_text']


# --- 6. Example run ---
if __name__ == "__main__":
    # Example rice health bulletin (French)
    bulletin = """
    Le riz de Camargue présente une recrudescence de la pyriculariose 
    due aux conditions climatiques humides de la semaine passée. 
    Les parcelles en stade montaison sont les plus sensibles. 
    Il est recommandé de surveiller les symptômes sur les feuilles 
    et d’éviter les excès d’azote. 
    Aucun traitement généralisé n’est conseillé pour le moment.
    """

    # Step 1: Clean text
    clean_text = preprocess_text(bulletin)

    # Step 2: Extract thematic keywords
    themes = extract_keywords(clean_text)

    # Step 3: Generate summary
    summary = summarize_text(clean_text)

    # Step 4: Display results
    print("\n=== THEMATIC KEYWORDS ===")
    print(themes)

    print("\n=== AUTOMATIC SUMMARY ===")
    print(summary)