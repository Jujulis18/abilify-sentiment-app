
# 🧠 Analyse des avis patients sur l’Abilify

Ce projet explore les retours d'expérience de patients concernant le médicament Abilify (aripiprazole), en analysant leurs avis textuels à l'aide de traitements NLP classiques et modernes. L’objectif est double :
- Identifier automatiquement les sentiments exprimés (positifs / négatifs).
- Extraire les problèmes récurrents, effets secondaires ou suggestions via topic modeling (LDA & BERTopic).

L’application est disponible via une interface interactive sous Streamlit, incluant des filtres par âge, genre et condition médicale : [Interface Streamlit](https://bvx9kwtgop7okgxpkwv624.streamlit.app/)

## 🧰 Data
Dataset récupéré sur Kaggle: [Abilify-oral-reviews-dataset](https://www.kaggle.com/datasets/joyshil0599/abilify-oral-reviews-dataset?resource=download)
incluant description-text, age, gender, condition, sentiment, etc.

## 🧰 Tech Stack

- [Python](https://www.python.org/)
- [Streamlit](https://streamlit.io/) pour le dashboard interactif
- [scikit-learn](https://scikit-learn.org/) pour LDA et preprocessing
- [BERTopic](https://maartengr.github.io/BERTopic/) pour la détection de thématiques avancées
- [Sentence-Transformers](https://www.sbert.net/) pour l’encodage sémantique
- [Seaborn / Matplotlib](https://seaborn.pydata.org/) pour les visualisations

---

## 📊 Analyse de sentiment

Une première étape a consisté à **classifier les avis en “Positif” ou “Négatif”**.  
Cette classification a été réalisée en analysant les colonnes de **notation globale (`overall-rating`)** :

- Avis **positif** : note ≥ 7/10
- Avis **négatif** : note ≤ 4/10
- Les notes intermédiaires sont considérés comme des avis **neutre**

Cela a permis de visualiser les tendances par tranche d’âge, genre, ou condition, et d’observer des différences notables dans la perception du médicament.

---

## 🧵 Extraction de thématiques

Deux approches ont été testées pour identifier les thèmes récurrents dans les textes sachant que l'on est sur des méthodes non-supervisés : 

### 1. LDA (Latent Dirichlet Allocation)

- Approche probabiliste classique
- Textes nettoyés et vectorisés avec `CountVectorizer`
- Il est possible de tester l'influence du nombre de topics à extraire et de mots-clés dominants par topic
- Des exemples d’avis représentatifs sont présentés pour chaque sujet

#### Fonctionnement du modèle :
LDA analyse la cooccurrence des mots dans les documents pour découvrir des groupes de termes qui reviennent souvent ensemble. Ces groupes représentent des thèmes latents.

🔍 **Limite** : 
- LDA ne comprend pas le sens des mots (pas de contexte), il ne fait qu’analyser des fréquences. Il fonctionne donc mieux sur des textes longs et bien structurés.
- les topics restent parfois trop généraux, mélangeant symptômes, effets indésirables

### 2. BERTopic (avec Sentence-BERT)

- Utilisation du modèle `all-MiniLM-L12-v2` pour encoder les phrases (possibilité de choisir: MPNet ou paraphrase)

#### Fonctionnement du modèle :
BERTopic est un modèle de clustering de textes basé sur l'encodage sémantique (via BERT ou SentenceTransformer).
Il convertit chaque avis en un vecteur qui capture son sens global (grâce au contexte), puis groupe ces vecteurs pour identifier des topics sémantiques.

📌 **Amélioration proposée** :
Pour une analyse plus fine, il est envisageable de :

- **Séparer les avis positifs et négatifs** avant de lancer BERTopic, pour éviter que des sentiments opposés apparaissent dans un même topic.
- **Ajouter une couche de détection d’opinion phrase-par-phrase** (avec un modèle comme RoBERTa finetuné sur des phrases d’avis médicaux) afin de segmenter finement les retours.

---

## ✅ Résultats

- Des thématiques ont émergé mais ne sont pas encore bien définis : troubles du sommeil, prise de poids, dosage trop important
- Il n'y a pas d'avis tranchés sur l'efficacité de ce médicament. Beaucoup d'effets secondaires mais il fonctionne sur certaines personnes sans être capable de définir la raison.

---

## 🚧 Pistes d'amélioration

### 📉 Limites actuelles

- LDA ne parvient pas toujours à produire des topics distincts
- BERTopic peut mélanger des opinions opposées dans un même cluster
- La classification positive/négative repose sur une simple règle de seuil de note

### 🚀 Améliorations envisagées

- Utiliser un **modèle de sentiment pré-entraîné** (RoBERTa, DistilBERT) pour classer chaque phrase individuellement
- Appliquer **BERTopic uniquement sur les avis négatifs** pour mieux identifier les problèmes majeurs
- Étiqueter manuellement un sous-ensemble pour un **finetuning supervisé**
- Explorer **Top2Vec**, **KeyBERT**, ou des méthodes hybrides pour améliorer la cohérence des thématiques
- Extraction de motif: Utiliser des modèles sequence-to-sequence (T5, BART) pour résumer ou extraire les problèmes mentionnés.
- Recherche des modèles pertinent à intégrer sur Hugging Face

---

## 💻 Démarrage rapide

```bash
git clone https://github.com/votre-utilisateur/abilify-nlp-analysis.git
cd abilify-nlp-analysis
pip install -r requirements.txt
streamlit run app.py
```

---


---

## 📬 Contact

Projet réalisé dans le cadre d’un portfolio de data science appliqué au domaine médical.  
Pour toute remarque ou suggestion, n'hésitez pas à me contacter !
