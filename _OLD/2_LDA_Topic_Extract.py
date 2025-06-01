from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation

# Modèle LDA
  n_topics = 8
  df = df.dropna(subset=["clean_review"]).reset_index(drop=True)
  texts = df["clean_review"].tolist()

  vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words="english")
  X = vectorizer.fit_transform(texts)
  
  lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
  lda.fit(X)

  # --- Extraction des mots-clés par topic ---
  def get_lda_topics(model, feature_names, n_words=10):
      topics = []
      for topic_idx, topic in enumerate(model.components_):
          keywords = [feature_names[i] for i in topic.argsort()[:-n_words - 1:-1]]
          topics.append(keywords)
      return topics
  
  topics_keywords = get_lda_topics(lda, vectorizer.get_feature_names_out())

  # debug
  print("Nombre de lignes dans df :", len(df))
  print("Nombre de textes :", len(texts))

  print(df[df["clean_review"].isna()])
  print(df.index.duplicated().sum())

  
  # --- Affectation des topics aux documents ---
  doc_topics = lda.transform(X)
  df["topic"] = doc_topics.argmax(axis=1)
  
  st.title("Problèmes identifiés (LDA)")
  st.markdown("Explorez les sujets récurrents dans les avis des patients.")
  
  for topic_num, keywords in enumerate(topics_keywords):
      st.markdown(f"### 🔹 Topic {topic_num}")
  
      st.write(f"**Mots-clés** : {', '.join(keywords)}")
  
      # Avis représentatifs
      examples = df[df["topic"] == topic_num]["description-text"].dropna().sample(min(3, len(df[df["topic"] == topic_num])), random_state=42)
      for example in examples:
          st.markdown(f"> {example.strip()}")
      
      st.markdown("---")
      
