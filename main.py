from PyPDF2 import PdfReader
import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

def main():
    print ('hello caca')


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

#outputMerger(extractTxtFrom(r"C:\Users\augus\Downloads\CV Augustin PROTIN English.pdf"))
#pdfFiles()
#outputToPdf()