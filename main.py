# === NLP Pipeline for French Rice Health Bulletins ===
# Author: [Your Name]
# Language: English
# Purpose: Preprocess, analyze, and summarize French agricultural bulletins

# Attention penser a : pip install emoji==1.7.0 clean-text==0.3.0

# --- 1. Install dependencies (if not yet installed) ---
# !pip install clean-text spacy keybert transformers sentence-transformers langdetect

import re
from cleantext import clean
from langdetect import detect
import spacy
from keybert import KeyBERT
from transformers import pipeline
from PyPDF2 import PdfReader
import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# --- 2. Load NLP models ---
# Load French spaCy model (run: python -m spacy download fr_core_news_md)
nlp = spacy.load("fr_core_news_sm")

# Keyword extractor (uses a multilingual BERT)
kw_model = KeyBERT(model='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Summarizer model (fine-tuned for French)
summarizer = pipeline("summarization", model="moussaKam/barthez-orangesum-abstract")


def extractTxtFrom(path):
    """take in argument the path of and return it in text chain"""
    path = path.replace("\\", "/")
    texte = ""
    try:
        with open(path, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                texte += page.extract_text() or ""
    except Exception as e:
        print(f"error during the PDF reading : {e}")
    return texte.strip()

def pdfFiles():
    """
    return all pdf files in the Input folder, placed in teh same folder as the script, ex :
        Porjet-Neuronal-Network-Rice-analyze-4A-P.Dijon/
        │
        ├── main.py
        ├── output.pdf
        └── input/
            ├── doc1.pdf
            └── doc2.pdf
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "input")
    if not os.path.exists(input_dir):
        print(f"The 'input' can't be found : {input_dir}")
        os.makedirs("input", exist_ok=True)
        print("the 'input folder has been sucessfuly created'")
        return []
    pdf_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(".pdf")
    ]

    return pdf_files

def outputMerger(text):
    """
    add at the end of output.txt the parmeter text
    """
    with open("output.txt", "a", encoding="utf-8") as f:
        f.write(text + "\n")

def outputToPdf():
    """
    teke the content of the output.txt and put it in a pdf, before deleting output.txt
    """
    dossier_script = os.path.dirname(os.path.abspath(__file__))
    # CpathFinding
    output_path = os.path.join(dossier_script, "output.txt")
    pdf_path = os.path.join(dossier_script, "rapport.pdf")

    if not os.path.exists(output_path):
        print(f"⚠️ Le fichier '{output_path}' n'existe pas.")
        return

    with open(output_path, "r", encoding="utf-8") as f:
        contenu = f.read()

    c = canvas.Canvas(pdf_path, pagesize=A4)
    largeur, hauteur = A4
    x, y = 50, hauteur - 50
    for ligne in contenu.split("\n"):
        c.drawString(x, y, ligne)
        y -= 15
        if y < 50:
            c.showPage()
            y = hauteur - 50

    c.save()
    os.remove(output_path)

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
def textProcessingComplete(bulletin):

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

def listPdfs():
    # Get the absolute path of the folder where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # List all files in that folder
    files = os.listdir(script_dir)
    
    # Filter only PDF files
    pdfs = [os.path.join(script_dir, f) for f in files if f.lower().endswith(".pdf")]
    
    return pdfs

def read_output_file():
    try:
        with open("output.txt", "r", encoding="utf-8") as file:
            content = file.read()
        return content
    except FileNotFoundError:
        return "Error: 'output.txt' not found."

def main():
    for i in pdfFiles() :
        outputMerger(extractTxtFrom(i))
    textProcessingComplete(read_output_file())
    outputToPdf()

main()# final_rice_reports_analysis.py
# Author: Augustin's Final Unified PDF Analyzer
# Purpose: Combine all PDFs into one corpus and generate a global, high-quality summary and keyword analysis.
# Language: English (code), but French processing/summarization.

import os
import re
from typing import List
from collections import Counter
from cleantext import clean
from langdetect import detect
import spacy
from keybert import KeyBERT
from transformers import pipeline
from PyPDF2 import PdfReader

# ---------------------------
# Configuration
# ---------------------------
INPUT_FOLDER = "input"
OUTPUT_FILE = "output.txt"
MAX_CHARS_PER_SEGMENT = 3000  # for summarization segmentation
TOP_KEYWORDS = 12             # more global keywords
MIN_TEXT_LEN_TO_SUMMARIZE = 80

# ---------------------------
# Model loading
# ---------------------------
print("Loading models... this might take a minute.")
try:
    nlp = spacy.load("fr_core_news_sm")
except Exception as e:
    raise RuntimeError("Please install French model: python -m spacy download fr_core_news_sm") from e

kw_model = KeyBERT(model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
summarizer = pipeline("summarization", model="moussaKam/barthez-orangesum-abstract")

# ---------------------------
# File utilities
# ---------------------------
def get_all_pdfs(folder: str = INPUT_FOLDER) -> List[str]:
    """
    Return list of all PDF paths in the input folder. Create folder if missing.
    """
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, folder)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print(f"Created folder '{path}'. Add PDFs and rerun.")
        return []
    pdfs = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(".pdf")]
    return sorted(pdfs)


def extract_pdf_text(pdf_path: str) -> str:
    """
    Extract and concatenate text from a PDF file using PyPDF2.
    """
    try:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        print(f"Failed to read {pdf_path}: {e}")
        return ""


# ---------------------------
# Text preprocessing
# ---------------------------
def preprocess_french_text(text: str) -> str:
    """
    Basic French text cleaning and normalization.
    """
    if not text.strip():
        return ""
    try:
        lang = detect(text)
        if lang != "fr":
            print(f"Warning: Detected '{lang}' language instead of 'fr'.")
    except Exception:
        pass
    cleaned = clean(
        text,
        lower=True,
        no_urls=True,
        no_emails=True,
        no_phone_numbers=True,
        no_currency_symbols=True,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# ---------------------------
# Keyword extraction
# ---------------------------
def extract_global_keywords(text: str, top_n: int = TOP_KEYWORDS) -> List[str]:
    """
    Extract representative keywords for the entire corpus using KeyBERT.
    Falls back to noun chunk frequency if needed.
    """
    if not text.strip():
        return []

    try:
        kw = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2),
                                       stop_words="french", top_n=top_n)
        result = [w for w, _ in kw if w and w.strip()]
        if result:
            return result
    except Exception as e:
        print(f"KeyBERT failed: {e}")

    # fallback using spaCy
    doc = nlp(text)
    chunks = []
    for chunk in doc.noun_chunks:
        t = chunk.text.lower().strip()
        if len(t) > 2:
            chunks.append(t)
    freq = Counter(chunks)
    return [k for k, _ in freq.most_common(top_n)]


# ---------------------------
# Summarization (multi-step)
# ---------------------------
def split_text_into_segments(text: str, max_chars: int = MAX_CHARS_PER_SEGMENT) -> List[str]:
    """
    Split large text into smaller sentence-based segments.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    segments = []
    current = ""
    for s in sentences:
        if len(current) + len(s) > max_chars:
            segments.append(current.strip())
            current = s
        else:
            current += " " + s
    if current.strip():
        segments.append(current.strip())
    return segments


def summarize_long_text(text: str, min_length: int = 60, max_length: int = 200) -> str:
    """
    Perform multi-step summarization: segment -> summarize each -> combine -> summarize again.
    """
    if len(text) < MIN_TEXT_LEN_TO_SUMMARIZE:
        return "Texte trop court pour une synthèse pertinente."

    segments = split_text_into_segments(text)
    print(f"Segmented into {len(segments)} parts for summarization...")

    partial_summaries = []
    for i, seg in enumerate(segments, 1):
        print(f" Summarizing segment {i}/{len(segments)} ...")
        try:
            part = summarizer(seg, min_length=min_length, max_length=max_length, do_sample=False)
            summary_text = part[0]["summary_text"].strip() if isinstance(part, list) else str(part)
            partial_summaries.append(summary_text)
        except Exception as e:
            print(f"Error summarizing segment {i}: {e}")

    # Combine partials and summarize again to ensure global coherence
    combined = " ".join(partial_summaries)
    try:
        final_summary = summarizer(combined, min_length=200, max_length=400, do_sample=False)
        return final_summary[0]["summary_text"].strip()
    except Exception as e:
        print(f"Final summarization failed: {e}")
        return combined


# ---------------------------
# Output writing
# ---------------------------
def write_final_output(keywords: List[str], summary: str, output_path: str = OUTPUT_FILE):
    """
    Write the global synthesis and keywords to output.txt.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=== GLOBAL SYNTHESIS OF ALL PDF DOCUMENTS ===\n\n")
        f.write("Keywords (overall themes):\n")
        f.write(", ".join(keywords) + "\n\n")
        f.write("=== SUMMARY ===\n\n")
        f.write(summary.strip() + "\n\n")
        f.write("=" * 60 + "\n")
    print(f"\nGlobal synthesis written to '{output_path}'.")


# ---------------------------
# Main function
# ---------------------------
def main():
    """
    Collect all PDFs, extract their text, produce one unified corpus,
    and generate a global synthesis + keywords into output.txt.
    """
    pdfs = get_all_pdfs()
    if not pdfs:
        print("No PDFs found in input/. Please add files and rerun.")
        return

    print(f"Found {len(pdfs)} PDF(s). Extracting text...")

    all_texts = []
    for p in pdfs:
        print(f"Reading {os.path.basename(p)} ...")
        raw = extract_pdf_text(p)
        pre = preprocess_french_text(raw)
        if pre:
            all_texts.append(pre)

    if not all_texts:
        print("No valid text found in PDFs.")
        return

    combined_text = "\n".join(all_texts)
    print(f"Total corpus length: {len(combined_text)} characters")

    keywords = extract_global_keywords(combined_text)
    summary = summarize_long_text(combined_text)
    write_final_output(keywords, summary)


if __name__ == "__main__":
    main()
