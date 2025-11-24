from transformers import pipeline, AutoTokenizer
import fitz
import os
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Configuration
INPUT_FOLDER = "input"
OUTPUT_TXT = "output.txt"
OUTPUT_PDF = "rapport.pdf"
MODEL_NAME = "facebook/bart-large-cnn"
words = [""]

texte_de_test = ["La transition énergétique repose sur une mutation profonde des systèmes de production, de transport et de consommation d’énergie. Elle vise à réduire la dépendance aux énergies fossiles et à limiter les émissions de gaz à effet de serre. Cette transformation implique des évolutions techniques, économiques et réglementaires importantes.",
                 "Le développement des énergies renouvelables constitue un pilier central de cette transition. L’éolien, le solaire, l’hydraulique et la biomasse occupent une place croissante dans le mix énergétique mondial. Leur intégration massive nécessite toutefois une adaptation des réseaux électriques, notamment pour gérer l’intermittence et garantir la stabilité du système.",
                 "L’efficacité énergétique représente un autre levier essentiel. Elle consiste à diminuer la consommation d’énergie pour un même service rendu. Cela implique l’amélioration de l’isolation des bâtiments, l’optimisation des procédés industriels et l’adoption d’appareils plus performants. Les gains obtenus permettent de réduire la demande globale sans sacrifier le confort ou la productivité.",
                 "Enfin, la réussite de la transition énergétique dépend de la mobilisation des acteurs publics et privés. Les politiques gouvernementales, les investissements des entreprises et les changements de comportement des citoyens sont indispensables pour soutenir cette transformation. Sans coordination et engagement collectif, les objectifs climatiques ne pourront pas être atteints."]

# Initialisation du tokenizer et du modèle de résumé
summarizer = pipeline("summarization", model=MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# Récupère la liste des fichiers PDF dans le dossier INPUT_FOLDER
def get_all_pdfs(folder: str = INPUT_FOLDER) -> list[str]:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, folder)
    if not os.path.exists(input_path):
        # Si le dossier n'existe pas, le créer et informer l'utilisateur
        os.makedirs(input_path, exist_ok=True)
        print(f"Dossier '{input_path}' créé. Ajoutez vos PDFs et relancez.")
        return []
    # Retourne la liste triée des chemins absolus vers les fichiers .pdf
    return sorted([os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith(".pdf")])


# Extrait les paragraphes textuels d'un PDF en respectant les blocs textuels
def extract_paragraphs(pdf_path: str) -> list[str]:
    try:
        doc = fitz.open(pdf_path)
        paragraphs = []
        buffer = ""

        for page in doc:
            blocks = page.get_text("dict")["blocks"]

            for block in blocks:
                if block.get("type") != 0:
                    continue

                for line in block.get("lines", []):
                    line_text = ""
                    for span in line.get("spans", []):
                        t = span.get("text", "").strip()
                        if t:
                            line_text += " " + t

                    line_text = line_text.strip()
                    if not line_text:
                        continue

                    buffer += " " + line_text

                    # Si la ligne se termine par un signe de fin de phrase, on "ferme" le paragraphe
                    if line_text.endswith((".", "!", "?")):
                        clean = buffer.strip()
                        if clean:
                            paragraphs.append(clean)
                        buffer = ""

        # Dernier paragraphe si pas terminé proprement
        if buffer.strip():
            paragraphs.append(buffer.strip())

        return paragraphs

    except Exception as e:
        print(f"Erreur lors de la lecture de {pdf_path}: {e}")
        return []



# Regroupe des paragraphes sans couper un paragraphe pour rester sous la limite en tokens
def chunk_paragraphs(paragraphs: list[str], max_tokens: int = 600) -> list[str]:
    chunks = []
    current = ""

    for para in paragraphs:
        test_text = (current + " " + para).strip()
        token_count = len(tokenizer.encode(test_text, add_special_tokens=False))

        if token_count > max_tokens:
            # On ferme le chunk courant si non vide, puis on démarre un nouveau chunk
            if current.strip():
                chunks.append(current.strip())
            current = para
        else:
            current = test_text

    if current.strip():
        chunks.append(current.strip())

    return chunks


# Résume un texte avec le modèle de summarization, prend des garde-fous en fallback
def neuronal_net_resum(text: str) -> str:
    if not text.strip():
        return ""

    token_count = len(text.split())

    # Calcule une taille de résumé plus large pour éviter les coupures
    max_len = token_count#min(250, max(60, int(token_count * 0.30)))
    min_len = int(max_len * 0.35)

    try:
        out = summarizer(
            text,
            max_length=max_len,
            min_length=min_len,
            do_sample=False
        )
        summary = out[0]["summary_text"].strip()

        # Ajoute un point si la phrase ne se termine pas correctement
        if summary and summary[-1] not in ".!?":
            summary += "."

        return summary

    except Exception as e:
        print(f"Erreur lors du résumé: {e}")
        fallback = text[:400].strip()
        return fallback


# Écrit la synthèse finale en TXT et PDF
def write_output(summary: str):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    txt_path = os.path.join(base_dir, OUTPUT_TXT)
    pdf_path = os.path.join(base_dir, OUTPUT_PDF)

    # Écriture du fichier texte
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== SYNTHÈSE GLOBALE ===\n\n")
        f.write(summary.strip() + "\n\n")
        f.write("=" * 60 + "\n")
    print(f"✔ Fichier texte écrit : {txt_path}")

    # Génération du PDF via reportlab
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
    # Etape 1 : On récupère les pdfs
    pdfs = get_all_pdfs()
    all_paragraphs = extract_paragraphs(pdfs)
    final_resume = ""
    if not pdfs:
        print("Aucun fichier PDF trouvé dans le dossier 'input'.")
        return
    print(f"PDFs détectés : {pdfs}")

    # Etape 2 : On les transforme en paragraphes structurés
    for p in pdfs:

        paras = extract_paragraphs(p)
        if paras:
            all_paragraphs.extend(paras)
        else:
            print(f"Aucun paragraphe extrait pour : {p}")
    if not all_paragraphs:
        print("Aucun texte extrait des PDFs.")
        return

    # Etape 3 : Filtrage des paragraphes selon la présence d'au moins un mot clé
    filtered_paragraphs = []

    for para in all_paragraphs:
        for w in words:
            print ("paargraphe traité")
            print (para)
            if w in para:
                filtered_paragraphs.append(para)
                print ("mot présent, paragraphe validé")
                break  # on évite d'ajouter deux fois le même paragraphe
            else :
                print ("mot non-présent, paragraphe non-validé")


        if not filtered_paragraphs:
            print("Aucun paragraphe contenant les mots recherchés.")
            return

    # Etape 4 : On transforme les paragraphes filtrés en chunks
    chunks = chunk_paragraphs(filtered_paragraphs)
    print(f"{len(chunks)} chunks créés à partir des paragraphes filtrés.")

    # Etape 5 : On résume chaque chunk et on concatène les résumés
    for chunk in chunks:
        summary = neuronal_net_resum(chunk)
        if summary:
            final_resume += summary + "\n\n"

    if not final_resume.strip():
        print("Aucun résumé généré.")
        return

    # Etape 6 : On écrit le résultat final en TXT et PDF
    write_output(final_resume)


if __name__ == "__main__":
    main()