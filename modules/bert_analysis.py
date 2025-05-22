from bertopic import BERTopic
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def run_bert(texts, n_topics="auto"):
  df["clean_review"] = df["clean_review"].apply(clean_text)

  embedding_model = SentenceTransformer("all-MiniLM-L12-v2")
  
  # topic_model = BERTopic(embedding_model=embedding_model, nr_topics=n_topics)
  topic_model = BERTopic(embedding_model=embedding_model, nr_topics=n_topics, verbose=True)
  topics, probs = topic_model.fit_transform(texts)

  return topics
  
