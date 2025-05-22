import streamlit as st
from bertopic import BERTopic
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


df = load_data("data/review_cleaned.csv")
texts = df["description-text"].dropna().tolist()

df["clean_review"] = df["clean_review"].apply(clean_text)

embedding_model = SentenceTransformer("all-MiniLM-L12-v2")

# topic_model = BERTopic(embedding_model=embedding_model, nr_topics=n_topics)
topic_model = BERTopic(embedding_model=embedding_model, nr_topics="auto", verbose=True)
topics, probs = topic_model.fit_transform(texts)

df["bertopic_topic"] = topics

st.write("#### Topics BERTopic:")
st.dataframe(topic_model.get_topic_info())

topics_info = topic_model.get_topic_info()  # dataframe avec tous les topics

# Récupère les mots-clés par topic (sauf topic -1)
topic_keywords = {
    topic_id: topic_model.get_topic(topic_id)
    for topic_id in topics_info["Topic"].values
    if topic_id != -1
}

st.title("Insights BERTopic")

def generate_bertopic_label(keywords):
    return ", ".join([word for word, _ in keywords[:3]])

topic_labels = {
    topic_id: generate_bertopic_label(keywords)
    for topic_id, keywords in topic_keywords.items()
}

for topic_id, keywords in topic_keywords.items():
    st.subheader(f"Topic {topic_id} : {topic_labels[topic_id]}")

    
    # Liste des mots-clés
    keyword_list = [word for word, _ in keywords]
    st.write("**Mots-clés :**", ", ".join(keyword_list))
    
    # Exemples d'avis du topic
    topic_docs = df[df["bertopic_topic"] == topic_id]["description-text"].sample(min(3, len(df[df["bertopic_topic"] == topic_id])))
    st.write("**Exemples d'avis :**")
    for text in topic_docs:
        st.write(f"- {text}")
    
    st.markdown("---")

