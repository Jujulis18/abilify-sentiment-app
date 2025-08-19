import streamlit as st
import pandas as pd
import pickle

from modules.preprocessing.data_loader import load_data

from modules.preprocessing.lda_analyzer import ( 
    prepare_lda_data, 
    run_lda_analysis, 
    extract_lda_topics, 
    assign_topics_to_documents, 
    get_topic_examples
)
from modules.utils import handle_empty_dataframe

st.set_page_config(page_title="Analyse LDA - Topics", layout="wide")


@st.cache_data
def run_cached_lda_analysis(texts_tuple, n_topics):
    """Cache l'analyse LDA car elle est coûteuse en calcul"""
    texts = list(texts_tuple)  # Reconvertir le tuple en liste pour LDA
    return run_lda_analysis(texts, n_topics=n_topics)

def main():
    st.title("Problèmes identifiés (LDA)")
    st.markdown("Explorez les sujets récurrents dans les avis des patients.")
    
    df_negative = load_data("data/df_with_topics_negative.csv")
    df_positive = load_data("data/df_with_topics_positive.csv")
    with open("data/topics_keywords.pkl", "rb") as f:
        topics_keywords = pickle.load(f)
    

    

    st.subheader("Paramètres LDA")
    #n_topics = st.slider("Nombre de topics", min_value=3, max_value=15, value=8)
    #n_words = st.slider("Mots par topic", min_value=5, max_value=20, value=10)
    n_examples = st.slider("Exemples par topic", min_value=1, max_value=5, value=3)
    st.warning("**Recommendation: prendre en compte les topics avec plus de 100 avis**")
    st.warning("Vous pourrez très prochainement jouer avec les parametres d'entrainement du model")

    col1, col2 = st.columns([2, 2])
        
    with col1:
        #df_cleaned, texts = prepare_lda_data(df, text_column='clean_review')
        
        # Informations sur les données
        st.info(f"📊 Analyse sur {len(df_negative)} avis négatifs nettoyés")
        
        # Analyse LDA 
        #with st.spinner("Exécution de l'analyse LDA..."):
            #texts_tuple = tuple(texts)
            #lda_model, vectorizer, doc_topics = run_cached_lda_analysis(texts_tuple, n_topics)
        
        # Extraction des topics
        #topics_keywords = extract_lda_topics(lda_model, vectorizer, n_words=n_words)
        
        # Assignation des topics aux documents
        #df_with_topics = assign_topics_to_documents(df_negative, doc_topics)
        
        # Affichage des résultats
        display_lda_results(df_negative, topics_keywords["negative"], n_examples)

    with col2:
        st.info(f"📊 Analyse sur {len(df_positive)} avis positive nettoyés")
        display_lda_results(df_positive, topics_keywords["positive"], n_examples)

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
        st.write(f" 🔹 Topic {topic_num}: {topic_counts.get(topic_num, 0)} avis", unsafe_allow_html=True)

        with st.expander(f" Mots-clés principaux : {', '.join(keywords)}", expanded=False):
                    
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