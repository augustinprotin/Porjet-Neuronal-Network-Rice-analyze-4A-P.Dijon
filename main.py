from transformers import pipeline, AutoTokenizer
import fitz
#import frontend
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

def extract_pdf_text(pdf_path: str) -> str:
    try:
        doc = fitz.open(pdf_path)
        full_text = ""

        for page_num, page in enumerate(doc, start=1):
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span["text"].strip()
                        color = span.get("color", (0, 0, 0))
                        if isinstance(color, int):
                            # convertir l'entier 0xRRGGBB en tuple (r,g,b) normalisé entre 0 et 1
                            r = ((color >> 16) & 255) / 255
                            g = ((color >> 8) & 255) / 255
                            b = (color & 255) / 255
                            color = (r, g, b)
                        size = span["size"]

                        # === Détection du texte vert (titres de section) ===
                        if color[1] > 0.6 and color[0] < 0.4 and color[2] < 0.4:
                            full_text += f"\n=== {text.upper()} ===\n"
                        else:
                            full_text += " " + text

            full_text += "\n\n"

        output_path = pdf_path.replace(".pdf", "_annotated.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        print(f"✅ Extraction terminée (avec styles) : {output_path}")
        return full_text

    except Exception as e:
        print(f"Erreur lors de la lecture de {pdf_path}: {e}")
        return ""

def extract_paragraphs(pdf_path: str) -> list[str]:
    """
    Extrait proprement les paragraphes d’un PDF en respectant
    la structure naturelle des blocs textuels du document.
    """
    try:
        doc = fitz.open(pdf_path)
        paragraphs = []

        for page in doc:
            blocks = page.get_text("dict")["blocks"]

            for block in blocks:
                # type 0 = bloc texte
                if block["type"] != 0:
                    continue

                block_text = ""
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        t = span.get("text", "").strip()
                        if t:
                            block_text += " " + t

                #le block au dessus fait toutes la magie, il permet de récuépéré les différents "block"/paragraphe, de récupérér la police ou les spé avec "span"
                #et après récupère le texte dans ces blocks. Enfin, strip enlève juste les espaces inutiles qui se sont glissés là.

                clean = block_text.strip()
                if clean:
                    paragraphs.append(clean)

        return paragraphs

    except Exception as e:
        print(f"Erreur lors de la lecture de {pdf_path}: {e}")
        return []


def chunk_paragraphs(paragraphs: list[str], max_tokens: int = 600) -> list[str]:
    """
    Construit des chunks de texte sans jamais couper un paragraphe.
    Chaque chunk reste sous la limite de tokens du modèle.
    """
    chunks = []
    current = ""

    for para in paragraphs:
        test_text = (current + " " + para).strip()
        token_count = len(tokenizer.encode(test_text, add_special_tokens=False))

        if token_count > max_tokens:
            # On "ferme" le chunk courant
            if current.strip():
                chunks.append(current.strip())
            current = para  # On commence un nouveau chunk
        else:
            current = test_text

    if current.strip():
        chunks.append(current.strip())

    return chunks



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
    all_paragraphs = []
    pdfs = get_all_pdfs()

    if not pdfs:
        print("Aucun fichier PDF trouvé dans le dossier 'input'.")
        return

    print(f"PDFs détectés : {pdfs}")

    for p in pdfs:
        paras = extract_paragraphs(p)
        if paras:
            all_paragraphs.extend(paras)
        else:
            print(f"Aucun paragraphe extrait pour : {p}")

    if not all_paragraphs:
        print("Aucun texte extrait des PDFs.")
        return

    # Chunking propre basé sur les paragraphes
    chunks = chunk_paragraphs(all_paragraphs)
    print(f"{len(chunks)} chunks créés à partir des paragraphes.")

    # Résumé
    full_text = "\n\n".join(chunks)
    resume = neuronal_net_resum(full_text)

    write_output(resume)



if __name__ == "__main__":
    main()
