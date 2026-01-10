import subprocess
import sys

def install_dependencies():
    packages = [
        "transformers[sentencepiece]",
        "accelerate",
        "pymupdf",          # cela correspond a fitz
        "reportlab"
    ]

    print("Installation des dépendances...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--upgrade"
    ] + packages)

# Installation auto au lancement
install_dependencies()


from transformers import pipeline, AutoTokenizer
import fitz
import os
import sys
import time
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import warnings

warnings.filterwarnings(
    "ignore",
    message="Both `max_new_tokens`"
)

# ---------- CONFIG ----------
INPUT_FOLDER = "input"
OUTPUT_TXT = "output.txt"
OUTPUT_PDF = "rapport.pdf"
MODEL_NAME = "plguillou/t5-base-fr-sum-cnndm"
# mots-clés à rechercher (case-insensitive)
words = ["riz", "Riz"]  # tu peux modifier / ajouter

# réglage verbose pour plus/moins de logs
VERBOSE = True

# ---------- INITIALISATIONS ----------
def safe_base_dir():
    # si __file__ n'existe pas (ex: execution interactive), fallback sur cwd
    return os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

BASE_DIR = safe_base_dir()

# Charge tokenizer et modèle, avec gestion d'erreur claire
try:
    print("Initialisation du tokenizer... ", end="", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("OK")
except Exception as e:
    print(f"\nErreur lors du chargement du tokenizer pour '{MODEL_NAME}': {e}")
    print("Vérifie que le nom du modèle est correct et que tu as accès au modèle.")
    raise

try:
    print("Initialisation du modèle de summarization (pipeline)... ", end="", flush=True)
    # device=-1 = CPU, mettre device=0 si GPU CUDA disponible
    summarizer = pipeline("summarization", model=MODEL_NAME, tokenizer=tokenizer, device=-1)
    print("OK")
except Exception as e:
    print(f"\nErreur lors de l'initialisation du pipeline summarization: {e}")
    raise

# ---------- FONCTIONS UTILITAIRES ----------

def cprint(msg: str, color: str = ""):
    """Print avec couleurs ANSI simples (color vide = pas de couleur)."""
    colors = {
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "blue": "\033[94m",
        "reset": "\033[0m"
    }
    prefix = colors.get(color, "")
    reset = colors["reset"] if prefix else ""
    print(f"{prefix}{msg}{reset}")

def get_all_pdfs(folder: str = INPUT_FOLDER) -> list:
    """Renvoie la liste des fichiers PDF complets (chemin absolu) du dossier input."""
    input_path = os.path.join(BASE_DIR, folder)

    if not os.path.exists(input_path):
        os.makedirs(input_path, exist_ok=True)
        cprint(f"Dossier '{input_path}' créé. Ajoute tes PDFs dans ce dossier puis relance le script.", "yellow")
        return []

    pdfs = sorted([
        os.path.join(input_path, f)
        for f in os.listdir(input_path)
        if f.lower().endswith(".pdf")
    ])

    if pdfs:
        cprint(f"{len(pdfs)} fichier(s) PDF trouvé(s) dans '{input_path}':", "green")
        for p in pdfs:
            print("   -", p)
    else:
        cprint(f"Aucun PDF trouvé dans '{input_path}'.", "yellow")

    return pdfs

def extract_paragraphs(pdf_path: str) -> list:
    paragraphs = []
    source_name = os.path.basename(pdf_path)

    try:
        if VERBOSE:
            cprint(f"Ouverture : {pdf_path}", "blue")
        doc = fitz.open(pdf_path)
        buffer = ""
        page_count = doc.page_count if hasattr(doc, "page_count") else len(doc)

        for i, page in enumerate(doc, start=1):
            blocks = page.get_text("dict").get("blocks", [])
            for block in blocks:
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    line_text = ""
                    for span in line.get("spans", []):
                        t = span.get("text", "").strip()
                        if t:
                            line_text += " " + t if line_text else t

                    line_text = line_text.strip()
                    if not line_text:
                        continue

                    buffer += (" " + line_text).strip()

                    if line_text.endswith((".", "!", "?")):
                        clean = buffer.strip()
                        if clean:
                            paragraphs.append((clean, source_name))
                        buffer = ""

        if buffer.strip():
            paragraphs.append((buffer.strip(), source_name))

        return paragraphs

    except Exception as e:
        cprint(f"Erreur lecture '{pdf_path}': {e}", "red")
        return []


def chunk_paragraphs(paragraphs: list, max_tokens: int = 600) -> list:
    """Regroupe les paragraphes en chunks sans dépasser max_tokens (tokenizer utilisé)."""
    if not paragraphs:
        return []
    chunks = []
    current = ""
    for para in paragraphs:
        test_text = (current + " " + para).strip() if current else para
        # utiliser tokenizer pour estimer le nombre de tokens
        try:
            token_count = len(tokenizer.encode(test_text, add_special_tokens=False))
        except Exception:
            # fallback: approximation par mots
            token_count = len(test_text.split())
        if token_count > max_tokens:
            if current.strip():
                chunks.append(current.strip())
            # si un paragraphe seul dépasse la limite, on l'ajoute lui-même (évite boucle infinie)
            single_count = len(tokenizer.encode(para, add_special_tokens=False)) if para else 0
            if single_count > max_tokens:
                # on coupe proprement le paragraphe en morceaux approximatifs par mots
                words = para.split()
                piece = []
                for w in words:
                    piece.append(w)
                    test_piece = " ".join(piece)
                    if len(tokenizer.encode(test_piece, add_special_tokens=False)) > max_tokens:
                        # retirer dernier mot pour rester en dessous
                        piece.pop()
                        if piece:
                            chunks.append(" ".join(piece))
                        piece = [w]
                if piece:
                    chunks.append(" ".join(piece))
                current = ""
            else:
                current = para
        else:
            current = test_text
    if current.strip():
        chunks.append(current.strip())
    return chunks

def neuronal_net_resum(text: str) -> str:
    """Résumé basé sur le pipeline summarizer avec garde-fous (longueur min/max raisonnable)."""
    if not text or not text.strip():
        return ""

    # estimation simple du nombre de tokens / mots
    token_count = len(text.split())

    # choix de max_length raisonnable pour le modèle (éviter max trop grand)
    max_len = min(512, max(60, int(token_count * 0.30)))  # entre 60 et 512
    min_len = max(20, int(max_len * 0.30))

    try:
        if VERBOSE:
            print(f"    Résumé (approx tokens={token_count}) -> min_len={min_len}, max_len={max_len} ...", end="", flush=True)
        out = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
        summary = out[0].get("summary_text", "").strip()
        if summary and summary[-1] not in ".!?":
            summary += "."
        if VERBOSE:
            print(" OK")
        return summary
    except Exception as e:
        cprint(f"\n    Erreur lors du résumé: {e}", "red")
        # fallback : tronquer
        fallback = (text[:400].strip() + "...") if len(text) > 400 else text.strip()
        return fallback

def write_output(summary: str):
    """Écrit le résumé global en TXT et PDF."""
    txt_path = os.path.join(BASE_DIR, OUTPUT_TXT)
    pdf_path = os.path.join(BASE_DIR, OUTPUT_PDF)

    try:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("=== SYNTHÈSE GLOBALE ===\n\n")
            f.write(summary.strip() + "\n\n")
            f.write("=" * 60 + "\n")
        cprint(f"✔ Fichier texte écrit : {txt_path}", "green")
    except Exception as e:
        cprint(f"Erreur écriture TXT: {e}", "red")

    try:
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
        cprint(f"✔ Fichier PDF généré : {pdf_path}", "green")
    except Exception as e:
        cprint(f"Erreur génération PDF: {e}", "red")

# ---------- MAIN ----------
def main():
    start_time = time.time()
    cprint("=== Début du traitement des PDFs ===", "blue")

    pdfs = get_all_pdfs()

    if not pdfs:
        cprint("Fin (aucun PDF à traiter).", "yellow")
        return

    all_paragraphs = []
    # extraction par fichier
    for idx, p in enumerate(pdfs, start=1):
        cprint(f"[{idx}/{len(pdfs)}] Extraction depuis : {os.path.basename(p)}", "blue")
        paras = extract_paragraphs(p)
        if paras:
            all_paragraphs.extend(paras)
        else:
            cprint(f"  Aucun paragraphe extrait pour : {p}", "yellow")

    if not all_paragraphs:
        cprint("Aucun texte extrait des PDFs. Fin.", "red")
        return

    cprint(f"Total paragraphes extraits : {len(all_paragraphs)}", "green")

    # Filtrage : ne garder que les paragraphes contenant au moins un mot-clé (case-insensitive)
    filtered_paragraphs = []
    cprint("Filtrage des paragraphes selon les mots-clés...", "blue")
    for i, para in enumerate(all_paragraphs, start=1):
        # checks case-insensitively
        lower_para = para.lower()
        matched = False
        for w in words:
            if w.lower() in lower_para:
                filtered_paragraphs.append(para)
                matched = True
                break
        if VERBOSE:
            status = "✓" if matched else "·"
            print(f"  [{i}/{len(all_paragraphs)}] {status} (len={len(para.split())} mots)")

    cprint(f"Paragraphes après filtrage : {len(filtered_paragraphs)}", "green")

    if not filtered_paragraphs:
        cprint("Aucun paragraphe contenant les mots recherchés. Fin.", "yellow")
        return

    # Chunking
    cprint("Création des chunks (groupes de paragraphes)...", "blue")
    chunks = chunk_paragraphs(filtered_paragraphs, max_tokens=600)
    cprint(f"{len(chunks)} chunk(s) créés à partir des paragraphes filtrés.", "green")

    if not chunks:
        cprint("Aucun chunk créé. Fin.", "red")
        return

    # Résumer chaque chunk
    cprint("Résumé des chunks...", "blue")
    summaries = []
    for i, chunk in enumerate(chunks, start=1):
        cprint(f"  Résumé chunk {i}/{len(chunks)} (approx {len(chunk.split())} mots)...", "blue")
        s = neuronal_net_resum(chunk)
        if s:
            summaries.append(s)
        else:
            cprint(f"    Aucun résumé généré pour le chunk {i}", "yellow")

    final_resume = "\n\n".join(summaries).strip()
    if not final_resume:
        cprint("Aucun résumé final généré. Fin.", "red")
        return

    # Ecriture des résultats
    cprint("Écriture des fichiers de sortie...", "blue")
    write_output(final_resume)

    elapsed = time.time() - start_time
    cprint(f"=== Traitement terminé en {elapsed:.1f}s ===", "green")



if __name__ == "__main__":
    main()

# ATTENTION, EXECUTER "pip install --upgrade transformers[sentencepiece] accelerate"