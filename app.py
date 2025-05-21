import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sentiment Analyse - Abilify", layout="wide")

# 📥 Chargement des données
df = pd.read_csv("data/reviews_cleaned.csv")

st.title("Analyse des avis patients sur l'Abilify")
st.markdown("Explorez les avis patients selon l’âge, le genre, et les conditions médicales.")

tab1, tab2 = st.tabs(["Analyse Sentiment", "Extraction Problèmes / Suggestions"])

# Analyse Sentiment
with tab1:
    # Filtres utilisateur
    age_range = st.slider("Filtrer par âge", int(df['Age_numeric'].min()), int(df['Age_numeric'].max()), (20, 60))
    gender_filter = st.multiselect("Genre", options=df['Gender'].unique(), default=df['Gender'].unique())
    condition_filter = st.multiselect("Condition médicale", options=df['Condition'].dropna().unique(), default=df['Condition'].dropna().unique()[:5])
    
    # Filtrage dynamique
    filtered_df = df[
        (df['Age_numeric'] >= age_range[0]) &
        (df['Age_numeric'] <= age_range[1]) &
        (df['Gender'].isin(gender_filter)) &
        (df['Condition'].isin(condition_filter))
    ]
    
    col1, col2 = st.columns(2)
    
    # Visualisation des sentiments
    with col1:
        st.subheader("Répartition des sentiments")
        fig1 = plt.figure(figsize=(5, 3))
        sns.countplot(data=filtered_df, x='sentiment', palette='pastel')
        st.pyplot(fig1)
    
    # Distribution d'âge selon le sentiment
    with col2:
        st.subheader("Âge selon sentiment")
        fig2 = plt.figure(figsize=(5, 3))
        sns.boxplot(data=filtered_df, x='sentiment', y='Age_numeric', palette='coolwarm')
        st.pyplot(fig2)
    
    # Exemples d'avis
    st.subheader("Exemples d'avis")
    sentiment_choisi = st.selectbox("Choisissez un type d'avis :", ['Positif', 'Négatif'])
    
    avis_filtrés = df[df['sentiment'] == sentiment_choisi]
    
    st.subheader(f"Exemples d'avis {sentiment_choisi.lower()}s")
    for i, row in avis_filtrés[['description-text', 'sentiment']].sample(3, random_state=1).iterrows():
        st.markdown(f"**Sentiment**: {row['sentiment']}")
        st.write(row['description-text'])
        st.markdown("---")
    
    
    
# Extraction Problèmes / Suggestions (BERT)
with tab2:
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer
    import pandas as pd
    from sklearn.decomposition import LatentDirichletAllocation
    from sentence_transformers import SentenceTransformer
    
    # Nettoyage
    df_cleaned = df[df["description-text"].notnull()]
    texts = df_cleaned["description-text"].tolist()
    
    # Création du modèle
    topic_model = BERTopic(language="english")
    topics, probs = topic_model.fit_transform(texts)
    
    # Visualisation des top topics
    topic_info = topic_model.get_topic_info()
    st.write("📌 Thèmes détectés :")
    st.dataframe(topic_info.head(10))
    
    # Affichage des termes fréquents pour un thème donné
    selected_topic = st.selectbox("Choisir un thème", topic_info["Topic"].values)
    if selected_topic != -1:
        st.write(topic_model.get_topic(selected_topic))

   

    # --- LDA ---
    st.write("### LDA :")
    
    # Vectorisation avec CountVectorizer
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    X = vectorizer.fit_transform(texts)
    
    # Modèle LDA
    n_topics = 8
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    
    # Fonction d'affichage des mots clés par topic LDA
    def display_topics(model, feature_names, no_top_words):
        for topic_idx, topic in enumerate(model.components_):
            st.write(f"**Topic {topic_idx}:**")
            st.write(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
    
    st.write("#### Topics LDA:")
    display_topics(lda, vectorizer.get_feature_names_out(), 10)
    
    # --- BERTopic ---
    st.write("### BERTopic :")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    topic_model = BERTopic(embedding_model=embedding_model, nr_topics=n_topics)
    topics, probs = topic_model.fit_transform(texts)
    
    st.write("#### Topics BERTopic:")
    st.dataframe(topic_model.get_topic_info())
    
    st.write("#### Mots clés du premier topic BERTopic:")
    st.write(topic_model.get_topic(0))

