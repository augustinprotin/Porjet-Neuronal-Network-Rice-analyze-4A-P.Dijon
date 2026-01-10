# Projet Neuronal Network – Rice Analyze

## Présentation

Ce programme permet d’analyser automatiquement un ensemble de documents PDF, d’en extraire le texte, de filtrer les paragraphes contenant certains mots-clés, puis de générer une synthèse automatique à l’aide d’un modèle de résumé neuronal basé sur Transformers.

Chaque résumé indique précisément de quel document il est issu.

Le programme génère :
- un fichier texte de synthèse
- un fichier PDF de synthèse

---

## Fonctionnalités

- Lecture automatique de tous les PDF d’un dossier
- Extraction intelligente des paragraphes
- Filtrage par mots-clés
- Découpage en blocs optimisés pour le modèle de résumé
- Résumé neuronal en français
- Traçabilité des sources (chaque résumé indique son PDF d’origine)
- Génération d’un rapport TXT et PDF

---

## Prérequis

- Python 3.9 ou plus  
- Connexion internet (au premier lancement pour télécharger le modèle)

Les dépendances sont installées automatiquement.

---

## Installation

Aucune installation manuelle requise.  
Au premier lancement, le script installe automatiquement :

- transformers + sentencepiece  
- accelerate  
- PyMuPDF (fitz)  
- reportlab  

---

## Utilisation

### 1. Ajouter les fichiers PDF

Créer un dossier `input` à la racine du projet et y placer les fichiers PDF à analyser :

Projet-Neuronal-Network-Rice-analyze/
├── main.py
├── input/
│ ├── document1.pdf
│ ├── document2.pdf


---

### 2. Configurer les mots-clés

Dans `main.py`, modifier la variable :

```python
words = ["riz", "Riz"]

Ajouter autant de mots-clés que nécessaire.

3. Lancer le programme
python main.py

Résultats

Après exécution, deux fichiers sont générés :

output.txt
rapport.pdf


Chaque résumé contient la mention :

Source(s) : document1.pdf, document2.pdf