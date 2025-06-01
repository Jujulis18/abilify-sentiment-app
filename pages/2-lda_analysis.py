import streamlit as st
import pandas as pd

from modules.preprocessing.data_loader import load_data

from modules.preprocessing.lda_analyzer import ( 
    prepare_lda_data, 
    run_lda_analysis, 
    extract_lda_topics, 
    assign_topics_to_documents, 
    get_topic_examples
)
from utils import handle_empty_dataframe

st.set_page_config(page_title="Analyse LDA - Topics", layout="wide")

@st.cache_data
def load_and_cache_data():
    return load_data("data/reviews_cleaned.csv")

@st.cache_data
def run_cached_lda_analysis(texts_tuple, n_topics):
    """Cache l'analyse LDA car elle est coûteuse en calcul"""
    texts = list(texts_tuple)  # Reconvertir le tuple en liste pour LDA
    return run_lda_analysis(texts, n_topics=n_topics)

def main():
    st.title("Problèmes identifiés (LDA)")
    st.markdown("Explorez les sujets récurrents dans les avis des patients.")
    
    df = load_and_cache_data()
    
    # Paramètres de l'analyse LDA
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Paramètres LDA")
        n_topics = st.slider("Nombre de topics", min_value=3, max_value=15, value=8)
        n_words = st.slider("Mots par topic", min_value=5, max_value=20, value=10)
        n_examples = st.slider("Exemples par topic", min_value=1, max_value=5, value=3)
    
    with col1:
        df_cleaned, texts = prepare_lda_data(df, text_column='clean_review')
        
        if not handle_empty_dataframe(df_cleaned, "Aucun texte nettoyé disponible pour l'analyse LDA."):
            return
        
        # Informations sur les données
        st.info(f"📊 Analyse sur {len(df_cleaned)} avis avec des textes nettoyés")
        
        # Analyse LDA 
        with st.spinner("Exécution de l'analyse LDA..."):
            # Convertir en tuple pour le cache (les listes ne sont pas hashables)
            texts_tuple = tuple(texts)
            lda_model, vectorizer, doc_topics = run_cached_lda_analysis(texts_tuple, n_topics)
        
        # Extraction des topics
        topics_keywords = extract_lda_topics(lda_model, vectorizer, n_words=n_words)
        
        # Assignation des topics aux documents
        df_with_topics = assign_topics_to_documents(df_cleaned, doc_topics)
        
        # Affichage des résultats
        display_lda_results(df_with_topics, topics_keywords, n_examples)

def display_lda_results(df_with_topics, topics_keywords, n_examples):
    """
    Affiche les résultats de l'analyse LDA de manière organisée
    
    Args:
        df_with_topics: DataFrame avec les topics assignés
        topics_keywords: liste des mots-clés par topic
        n_examples: nombre d'exemples à afficher par topic
    """
    
    # Statistiques générales
    st.subheader("📈 Répartition des topics")
    topic_counts = df_with_topics['topic'].value_counts().sort_index()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.bar_chart(topic_counts)
    
    with col2:
        st.write("**Nombre d'avis par topic:**")
        for topic_num, count in topic_counts.items():
            st.write(f"Topic {topic_num}: {count} avis")
    
    st.markdown("---")
    
    # Détail des topics
    st.subheader("🔍 Analyse détaillée des topics")
    
    for topic_num, keywords in enumerate(topics_keywords):
        with st.expander(f"🔹 Topic {topic_num} - {topic_counts.get(topic_num, 0)} avis", expanded=False):
            
            # Mots-clés
            st.write(f"**Mots-clés principaux:** {', '.join(keywords)}")
            
            # Exemples d'avis
            st.write("**Exemples d'avis représentatifs:**")
            examples = get_topic_examples(df_with_topics, topic_num, n_examples=n_examples)
            
            if len(examples) > 0:
                for i, example in enumerate(examples, 1):
                    st.markdown(f"{i}. *{example.strip()}*")
            else:
                st.warning("Aucun exemple disponible pour ce topic.")
            
            st.markdown("---")

if __name__ == "__main__":
    main()