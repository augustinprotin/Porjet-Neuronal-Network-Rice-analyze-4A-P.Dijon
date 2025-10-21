from transformers import pipeline, AutoTokenizer
from PyPDF2 import PdfReader
import re
import os
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

INPUT_FOLDER = "input"
OUTPUT_TXT = "output.txt"
OUTPUT_PDF = "rapport.pdf"
MODEL_NAME = "facebook/bart-large-cnn"
summarizer = pipeline("summarization", model=MODEL_NAME, truncation=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def get_all_pdfs(folder: str = INPUT_FOLDER) -> list[str]:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, folder)
    if not os.path.exists(input_path):
        os.makedirs(input_path, exist_ok=True)
        print(f"Dossier '{input_path}' créé. Ajoutez vos PDFs et relancez.")
        return []
    return sorted([os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith(".pdf")])

#def extract_pdf_text(pdf_path: str) -> str:
    '''
    try:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            full_text = ""

            # Extraction du texte de chaque page
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"

        # Expression régulière pour détecter les titres de section (majuscules accentuées incluses)
        pattern = r"\b([A-ZÉÈÀÙÂÊÎÔÛÇ]+(?:\s+[A-ZÉÈÀÙÂÊÎÔÛÇ]+)*)\b"

        # Recherche des titres
        titles = re.findall(pattern, full_text)

        # Liste d'exclusions pour éviter les faux positifs
        exclusions = {"REPRODUCTION", "BULLETIN", "PAGE", "AOUT", "RIZ", "VARIETES", "DES"}

        # Ajout des balises dans le texte
        for title in titles:
            if title not in exclusions and len(title) > 3:
                full_text = re.sub(rf"\b{re.escape(title)}\b", f"\n=== {title} ===\n", full_text)

        # Sauvegarde du texte annoté
        output_path = pdf_path.replace(".pdf", "_annotated.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        print(f"✅ Extraction terminée : {output_path} généré.")
        return full_text

    except Exception as e:
        print(f"Erreur lors de la lecture de {pdf_path}: {e}")
        return ""
    
    '''
    
    
def extract_pdf_text(pdf_path: str) -> str:
    try:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
            
    except Exception as e:
        print(f"Erreur lors de la lecture de {pdf_path}: {e}")
        return ""
    

def chunk_text_by_tokens(text: str, max_tokens: int = None) -> list[str]:
    if not max_tokens:
        # garder une marge de sécurité par rapport à la limite du modèle
        max_tokens = max(256, min(800, tokenizer.model_max_length - 50))
    ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(ids), max_tokens):
        slice_ids = ids[i:i + max_tokens]
        chunk_text = tokenizer.decode(slice_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        chunks.append(chunk_text.strip())
    return [c for c in chunks if c]

def neuronal_net_resum(text: str):
    if not text.strip():
        return "Aucun texte trouvé."
    chunks = chunk_text_by_tokens(text)
    summaries = []
    print(f"{len(chunks)} blocs à résumer.")
    # paramètres raisonnables de résumé (en tokens)
    DEFAULT_MAX_SUMMARY = 180
    DEFAULT_MIN_SUMMARY = 60
    for i, chunk in enumerate(chunks, start=1):
        # adapter si chunk très court
        try:
            token_count = len(tokenizer.encode(chunk, add_special_tokens=False))
        except Exception:
            token_count = 0
        max_len = min(DEFAULT_MAX_SUMMARY, max(30, int(token_count * 0.25)))
        min_len = max(DEFAULT_MIN_SUMMARY // 3, int(max_len * 0.4))
        print(f"Bloc {i}/{len(chunks)} : ~{token_count} tokens → résumé {min_len}-{max_len} tokens.")
        try:
            out = summarizer(
                chunk,
                max_length=max_len,
                min_length=min_len,
                do_sample=False,
                truncation=True
            )
            summary = out[0]["summary_text"].strip()
            if summary:
                summaries.append(summary)
            else:
                print(f"Attention: résumé vide pour le bloc {i}, j'ajoute un fallback (premières phrases).")
                summaries.append(chunk[:min(400, len(chunk))].strip())
        except Exception as e:
            print(f"Erreur sur le bloc {i}: {e}. Ajout d'un fallback.")
            summaries.append(chunk[:min(400, len(chunk))].strip())

    # séparer par double nouvelle ligne pour que chaque chunk devienne un paragraphe distinct
    full_summary = "\n\n".join(summaries)
    print("Résumé final généré.")
    return full_summary

def write_output(summary: str):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    txt_path = os.path.join(base_dir, OUTPUT_TXT)
    pdf_path = os.path.join(base_dir, OUTPUT_PDF)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== SYNTHÈSE GLOBALE ===\n\n")
        f.write(summary.strip() + "\n\n")
        f.write("=" * 60 + "\n")
    print(f"✔ Fichier texte écrit : {txt_path}")

    doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                            leftMargin=50, rightMargin=50, topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()
    body_style = styles["Normal"]
    body_style.fontName = "Helvetica"
    body_style.fontSize = 11
    body_style.leading = 14

    flowables = []
    flowables.append(Paragraph("SYNTHÈSE GLOBALE", styles["Heading2"]))
    flowables.append(Spacer(1, 6))

    for para in summary.strip().split("\n\n"):
        cleaned = para.strip().replace("\n", " ")
        if cleaned:
            flowables.append(Paragraph(cleaned, body_style))
            flowables.append(Spacer(1, 6))

    doc.build(flowables)
    print(f"✔ Fichier PDF généré : {pdf_path}")

def main():
    text = ""
    pdfs = get_all_pdfs()
    if not pdfs:
        print("Aucun fichier PDF trouvé dans le dossier 'input'.")
        return
    print(f"PDFs détectés : {pdfs}")
    for p in pdfs:
        text_part = extract_pdf_text(p)
        if text_part:
            # ajouter un séparateur pour éviter la fusion brutale de mots/phrases entre PDF
            text += "\n\n" + text_part
            print(f"✔ Fichier texte écrit : {text}")

    if not text.strip():
        print("Aucun texte extrait des PDFs.")
        return
    resume = neuronal_net_resum(text)
    write_output(resume)

if __name__ == "__main__":
    main()
