from transformers import pipeline, AutoTokenizer
import fitz
import os
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import re

# === CONFIG ===
INPUT_FOLDER = "input"
OUTPUT_TXT = "output.txt"
OUTPUT_PDF = "rapport.pdf"
MODEL_NAME = "plguillou/t5-base-fr-sum-cnndm"
MAX_TOKENS_MODEL = 512
WORDS = ["riz", "Riz"]

# Initialisation modèle
summarizer = pipeline("summarization", model=MODEL_NAME, tokenizer=MODEL_NAME, device=-1)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# === FONCTIONS ===

def get_all_pdfs(folder: str = INPUT_FOLDER) -> list[str]:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, folder)
    if not os.path.exists(input_path):
        os.makedirs(input_path, exist_ok=True)
        print(f"Dossier '{input_path}' créé. Ajoutez vos PDFs et relancez.")
        return []
    return sorted([os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith(".pdf")])

def extract_paragraphs(pdf_paths) -> list[str]:
    if isinstance(pdf_paths, str):
        pdf_paths = [pdf_paths]

    paragraphs = []

    for pdf_path in pdf_paths:
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    if block["type"] == 0:
                        text = " ".join(
                            span["text"].strip()
                            for line in block.get("lines", [])
                            for span in line.get("spans", [])
                            if span["text"].strip()
                        )
                        if text:
                            paragraphs.append(text)
        except Exception as e:
            print(f"Erreur lors de la lecture de {pdf_path}: {e}")
    return paragraphs

def filter_paragraphs(paragraphs: list[str], words: list[str], min_tokens: int = 20) -> list[str]:
    filtered = []
    for para in paragraphs:
        para_lower = para.lower()
        if any(word.lower() in para_lower for word in words):
            if len(tokenizer.encode(para, add_special_tokens=False)) >= min_tokens:
                filtered.append(para)
    return filtered

def chunk_paragraphs_intelligente(paragraphs: list[str], max_tokens: int = MAX_TOKENS_MODEL) -> list[str]:
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        test_chunk = (current_chunk + " " + para).strip()
        token_count = len(tokenizer.encode(test_chunk, add_special_tokens=False))

        if token_count > max_tokens:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            para_tokens = len(tokenizer.encode(para, add_special_tokens=False))
            if para_tokens > max_tokens:
                encoded = tokenizer.encode(para, max_length=max_tokens, truncation=True, add_special_tokens=False)
                truncated_para = tokenizer.decode(encoded, skip_special_tokens=True)
                chunks.append(truncated_para.strip())
                current_chunk = ""
            else:
                current_chunk = para
        else:
            current_chunk = test_chunk

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

def neuronal_net_resum(chunk: str) -> str:
    if not chunk.strip():
        return ""

    token_count = len(tokenizer.encode(chunk, add_special_tokens=False))
    max_len = int(min(180, max(60, int(token_count * 0.35))))
    min_len = int(max(40, int(max_len * 0.5)))

    try:
        out = summarizer(
            chunk,
            max_length=max_len,
            min_length=min_len,
            max_new_tokens=None,
            do_sample=False
        )
        summary = out[0]["summary_text"].strip()
        if summary and summary[-1] not in ".!?":
            summary += "."
        return summary
    except Exception as e:
        print(f"Erreur lors du résumé: {e}")
        return chunk[:400].strip()

def write_output(summary: str):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    txt_path = os.path.join(base_dir, OUTPUT_TXT)
    pdf_path = os.path.join(base_dir, OUTPUT_PDF)

    # TXT
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== SYNTHÈSE GLOBALE ===\n\n")
        f.write(summary.strip() + "\n\n")
        f.write("=" * 60 + "\n")
    print(f"✔ Fichier texte écrit : {txt_path}")

    # PDF
    doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                            leftMargin=50, rightMargin=50, topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()
    body_style = styles["Normal"]
    body_style.fontName = "Helvetica"
    body_style.fontSize = 11
    body_style.leading = 14

    flowables = [Paragraph("SYNTHÈSE GLOBALE", styles["Heading2"]), Spacer(1, 6)]
    for para in summary.strip().split("\n\n"):
        cleaned = para.strip().replace("\n", " ")
        if cleaned:
            flowables.append(Paragraph(cleaned, body_style))
            flowables.append(Spacer(1, 6))

    doc.build(flowables)
    print(f"✔ Fichier PDF généré : {pdf_path}")

# === MAIN ===
def main():
    pdfs = get_all_pdfs()
    if not pdfs:
        print("Aucun PDF trouvé.")
        return

    print(f"PDFs détectés : {pdfs}")
    all_paragraphs = extract_paragraphs(pdfs)

    if not all_paragraphs:
        print("Aucun texte extrait.")
        return

    filtered_paragraphs = filter_paragraphs(all_paragraphs, WORDS)
    if not filtered_paragraphs:
        print("Aucun paragraphe contenant les mots recherchés.")
        return

    chunks = chunk_paragraphs_intelligente(filtered_paragraphs, MAX_TOKENS_MODEL)
    print(f"{len(chunks)} chunks créés à partir des paragraphes filtrés.")

    final_resume = ""
    for chunk in chunks:
        summary = neuronal_net_resum(chunk)
        if summary:
            final_resume += summary + "\n\n"

    if final_resume.strip():
        write_output(final_resume)
    else:
        print("Aucun résumé généré.")

if __name__ == "__main__":
    main()
