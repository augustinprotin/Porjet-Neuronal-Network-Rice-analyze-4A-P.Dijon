import re
from cleantext import clean
from keybert import KeyBERT
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import spacy

# === Load NLP models ===
nlp = spacy.load("fr_core_news_sm")
kw_model = KeyBERT(model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# === Load summarization model safely ===
model_name = "moussaKam/barthez-orangesum-abstract"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

# === Preprocessing ===
def preprocess_text(text: str) -> str:
    """Nettoyage doux du texte français."""
    text = clean(
        text,
        lower=False,
        no_urls=True,
        no_emails=True,
        no_phone_numbers=True,
        no_currency_symbols=True
    )
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# === Keyword extraction ===
def extract_keywords(text: str, top_n: int = 8):
    """Extraction robuste avec KeyBERT + fallback spaCy."""
    try:
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 1),
            top_n=top_n
        )
        keyphrases = [kw for kw, score in keywords if len(kw.split()) > 0]

        # Fallback si KeyBERT ne trouve rien
        if not keyphrases:
            doc = nlp(text)
            keyphrases = list({ent.text for ent in doc.ents if len(ent.text) > 2})
        return keyphrases
    except Exception as e:
        print("⚠️ Erreur KeyBERT :", e)
        return []

# === Summarization ===
def summarize_text(text: str, min_length: int = 40, max_length: int = 150):
    """Résumé français stable avec tronquage au niveau des tokens."""
    try:
        # Encode et tronque proprement
        inputs = tokenizer(
            text,
            max_length=1024,     # limite du modèle
            truncation=True,
            return_tensors="pt"
        )

        summary_ids = model.generate(
            **inputs,
            min_length=min_length,
            max_length=max_length,
            no_repeat_ngram_size=2,
            num_beams=4
        )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary.strip()

    except Exception as e:
        print("⚠️ Erreur résumé :", e)
        return "(résumé non disponible)"

# === Example run ===
if __name__ == "__main__":
    bulletin = """Les parcelles de référence, sur lesquelles sont localisés les essais variétaux,
permettent un suivi des stades des différentes variétés en fonction des
dates de semis.Malgré les températures globalement supérieures aux normales depuis la
mise en place des cultures, les cycles semis-épiaison apparaissent proches
des cycles standards pour les variétés très précoces à demi-précoces ayant
épié à ce jour. Après une période avec un débit très limité mi-juillet (environ 500
m3/seconde à Tarascon), le Rhône est légèrement remonté depuis, entre
700 et 800 m3/seconde sur la période récente (source : http://www.rdbrmc.com/hydroreel2/station.php?codestation=81).
Il convient de rester vigilant et de contrôler la salinité de l’eau au
niveau des stations de pompage, plus particulièrement en cas
d’épisodes de Mistral. Le réseau de piégeage (pièges à phéromones) a été mis en place début juin sur
la zone rizicole camarguaise (20 pièges à phéromones répartis sur 9 sites), alors
que le pic de vol de la première génération était passé. Après une période au cours de laquelle les captures de papillons ont été
quasiment nulles, les relevés réalisés depuis une dizaine de jours ont montré un
redémarrage des captures (voir courbe ci-dessous). La durée de vie des papillons est courte, 4 à 8 jours pendant lesquels les
femelles pondent leurs œufs. A cette génération, compte tenu de températures élevées, l’incubation des œufs
dure 5 à 7 jours. Dès éclosion, les jeunes larves migrent vers l’intérieur des tiges
de riz dans lesquelles elles se développent.La sensibilité des différentes variétés (voir tableau ci-après) à la pyrale du riz et
la situation parcellaire (zone sensible et/ou parcelle abritée) sont les deux
éléments principaux à prendre en compte dans l’analyse du risque. Des stries longitudinales caractéristiques, en particulier à l’extrémité des
dernières feuilles étalées, sont à nouveau observées depuis une dizaine de jours. Les charançons adultes, responsables de ces symptômes sont plus difficiles à
observer qu’au mois de mai-juin compte tenu d’une végétation plus développée
dans laquelle ils s’abritent aux heures chaudes. Cette génération d’adultes, issue des larves observées durant le mois de juillet
au niveau des racines, est celle qui passera l’hiver en diapause (vie ralentie) dans les zones adjacentes aux rizières.
A ce jour, quelques symptômes sur feuilles et cous ont été observés sur une
variété sensible présente dans le réseau variétal. La durée d’humectation du
feuillage (favorisée par des conditions de vent réduit ou nul et de temps couvert)
constitue un facteur clé du développement de la maladie. Il convient de rester vigilant, et de réaliser des observations fréquentes, en
particulier dans les situations à risques : variétés les plus sensibles, parcelles ou parties de parcelles abritées (haies, ségonnaux), disponibilité élevée en azote
(précédent luzerne, …). LES OBSERVATIONS CONTENUES DANS CE BULLETIN ONT ETE REALISEES PAR LES PARTENAIRES SUIVANTS QUI CONSTITUENT LE COMITE DE REDACTION DE CE BULLETIN : ARNAUD BOISNARD, Sonia ER-RAHMOUNI, Gérard
FEOUGIER, Cyrille THOMAS (Centre Français du Riz),


"""

    clean_text = preprocess_text(bulletin)
    keywords = extract_keywords(clean_text)
    summary = summarize_text(clean_text)

    print("\n=== MOTS-CLÉS ===")
    print(keywords if keywords else "(aucun mot-clé trouvé)")

    print("\n=== RÉSUMÉ ===")
    print(summary)
