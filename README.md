# Analyse des avis patients sur l’Abilify

## Contexte
Projet de data science en NLP visant à analyser les retours d’expérience des patients sur le médicament Abilify, à partir d’un dataset public.
L’objectif est d’identifier automatiquement les sentiments exprimés dans les avis et de faire émerger les effets secondaires ou problèmes récurrents à l’aide de méthodes classiques et modernes de traitement du langage naturel.

## Objectifs
- **Objectifs principaux** :
  - Classifier les sentiments exprimés (positifs / négatifs).
  - Identifier les problèmes récurrents, effets secondaires ou suggestions via topic modeling (LDA & BERTopic).
- **Objectifs secondaires** :
  - Développer une interface interactive sous Streamlit pour visualiser les résultats.
  - Comparer les performances des méthodes classiques et modernes de NLP.

## Méthodologie
- **Outils et technologies utilisés** :
  - Python, Panda, scikit-learn
  - BERTopic pour la détection de thématiques avancées
  - Sentence-Transformers pour l’encodage sémantique
  - Seaborn / Matplotlib pour les visualisations
  - Streamlit pour le dashboard interactif [Streamlit Dashboard](https://abilify-sentiment-app-gxcwthycjesnveq6mosldv.streamlit.app/)
 
- **Processus** :
  1. Récupération du dataset sur Kaggle : Abilify-oral-reviews-dataset incluant description-text, age, gender, condition, sentiment, etc.
  2. Classification des avis en “Positif” ou “Négatif” en analysant les colonnes de notation globale (overall-rating).
  3. Visualisation des tendances par tranche d’âge, genre, ou condition.
  4. Extraction de topics en utilisant LDA et BERTopic.
  5. Analyse des résultats et identification des thèmes récurrents.
 
<img width="500" height="825" alt="image" src="https://github.com/user-attachments/assets/8573e3f2-a44b-4cfb-9f51-b5bac8ac8f77" />

<img width="500" height="741" alt="image" src="https://github.com/user-attachments/assets/8cb2f406-61b3-4534-ae00-abf566e56303" />


## Analyse et Résultats
- **Analyse des données** :
  - Classification des avis en positifs, négatifs et neutres basée sur les notes globales.
  - Visualisation des tendances par tranche d’âge, genre, ou condition.
- **Résultats obtenus** :
  - Identification de thématiques récurrentes telles que les troubles du sommeil, la prise de poids, et le dosage trop important.
  - Observation de différences notables dans la perception du médicament selon les groupes démographiques.
  - **Limites des modèles** :
    - **LDA (Latent Dirichlet Allocation)** : LDA est une approche probabiliste classique qui analyse la répétabilité des mots dans les documents pour découvrir des groupes de termes qui reviennent souvent ensemble. Il ne comprend pas le sens des mots (pas de contexte), il ne fait qu’analyser des fréquences. Il fonctionne donc mieux sur des textes longs et bien structurés. Les topics restent parfois trop généraux, mélangeant symptômes et effets indésirables.
    - **BERTopic** : BERTopic est un modèle de clustering de textes basé sur l'encodage sémantique (via BERT ou SentenceTransformer). Il convertit chaque avis en un vecteur qui capture son sens global, puis groupe ces vecteurs pour identifier des topics récurrents. Il peut mélanger des opinions opposées dans un même cluster. La classification positive/négative repose sur une simple règle de seuil de note, ce qui peut limiter la précision des résultats.

## Impact Business
- **Valeur ajoutée** :
  - Meilleure compréhension des retours patients sur l'Abilify.
  - Identification des problèmes récurrents et des effets secondaires pour améliorer le médicament.
- **Recommandations** :
  - Utiliser un modèle de sentiment pré-entraîné pour classer chaque phrase individuellement.
  - Appliquer BERTopic uniquement sur les avis négatifs pour mieux identifier les problèmes majeurs.
  - Étiqueter manuellement un sous-ensemble pour un finetuning supervisé.

## Conclusion
Le projet a permis d'identifier des thématiques récurrentes et des différences notables dans la perception du médicament selon les groupes démographiques.

Les méthodes de NLP classiques et modernes ont été comparées pour extraire des informations pertinentes des avis patients.
- **Leçons apprises** :
  - L'importance de combiner différentes méthodes de NLP pour obtenir des résultats plus précis.
  - La nécessité de continuer à améliorer les modèles pour mieux comprendre les retours patients.

## 🚀 Comment démarrer

1. Clonez le dépôt
2. Installez les dépendances :
```bash
pip install -r requirements.txt
```
3. Lancez le dashboard :
```bash
streamlit run app.py
```

## Références et Liens
- **Sources de données** :
  - [Dataset sur Kaggle: Abilify-oral-reviews-dataset](https://www.kaggle.com/datasets/joyshil0599/abilify-oral-reviews-dataset?resource=download)

