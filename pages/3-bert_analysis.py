import streamlit as st
import pandas as pd

from modules.preprocessing.data_loader import load_data

from modules.preprocessing.bert_analyzer import (
    prepare_bert_data,
    run_bert_analysis,
    extract_bert_topics_info,
    generate_bert_topic_labels,
    assign_bert_topics_to_documents,
    get_bert_topic_examples
)
from modules.utils import handle_empty_dataframe, clean_text

st.set_page_config(page_title="Analyse BERTopic", layout="wide")

@st.cache_data
def load_and_cache_data():
    return load_data("data/reviews_cleaned.csv")

@st.cache_data
def run_cached_bert_analysis(texts_tuple, n_topics, embedding_model_name):
    """Cache l'analyse BERTopic car elle est très coûteuse en calcul"""
    texts = list(texts_tuple)  # Reconvertir le tuple en liste
    return run_bert_analysis(texts, n_topics=n_topics, embedding_model_name=embedding_model_name, verbose=False)

def main():
    st.title("Insights BERTopic")
    st.markdown("Découvrez les sujets émergents avec l'analyse BERTopic basée sur les embeddings.")
    
    df = load_and_cache_data()
    
    # Paramètres de l'analyse BERTopic
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Paramètres BERTopic")
        
        # Option pour le nombre de topics
        auto_topics = st.checkbox("Détection automatique des topics", value=True)
        if auto_topics:
            n_topics = "auto"
        else:
            n_topics = st.slider("Nombre de topics", min_value=5, max_value=20, value=10)
        
        # Modèle d'embedding
        embedding_options = {
            "all-MiniLM-L12-v2": "MiniLM (rapide)",
            "all-mpnet-base-v2": "MPNet (meilleur)",
            "paraphrase-MiniLM-L6-v2": "Paraphrase (léger)"
        }
        
        selected_model = st.selectbox(
            "Modèle d'embedding",
            options=list(embedding_options.keys()),
            format_func=lambda x: embedding_options[x],
            index=0
        )
        
        n_examples = st.slider("Exemples par topic", min_value=1, max_value=5, value=3)
        n_label_words = st.slider("Mots pour les labels", min_value=2, max_value=5, value=3)
    
    with col1:
        # Préparation des données
        df_cleaned, texts = prepare_bert_data(df, text_column='description-text')
        
        if not handle_empty_dataframe(df_cleaned, "Aucun texte disponible pour l'analyse BERTopic."):
            return
        
        # Nettoyage optionnel des textes 
        if 'clean_review' not in df_cleaned.columns:
            with st.spinner("Nettoyage des textes..."):
                df_cleaned['clean_review'] = df_cleaned['description-text'].apply(clean_text)
        
        # Informations sur les données
        st.info(f"📊 Analyse BERTopic sur {len(df_cleaned)} avis")
        
        # Analyse BERTopic
        with st.spinner("Exécution de l'analyse BERTopic... (cela peut prendre du temps)"):
            # Convertir en tuple pour le cache
            texts_tuple = tuple(texts)
            topic_model, topics, probs = run_cached_bert_analysis(texts_tuple, n_topics, selected_model)
        
        # Extraction des informations des topics
        topics_info, topic_keywords = extract_bert_topics_info(topic_model)
        
        # Génération des labels
        topic_labels = generate_bert_topic_labels(topic_keywords, n_words=n_label_words)
        
        # Assignation des topics aux documents
        df_with_topics = assign_bert_topics_to_documents(df_cleaned, topics)
        
        # Affichage des résultats
        display_bert_results(df_with_topics, topics_info, topic_keywords, topic_labels, n_examples)

def display_bert_results(df_with_topics, topics_info, topic_keywords, topic_labels, n_examples):
    """
    Affiche les résultats de l'analyse BERTopic de manière organisée
    
    Args:
        df_with_topics: DataFrame avec les topics assignés
        topics_info: DataFrame des informations des topics
        topic_keywords: dictionnaire des mots-clés par topic
        topic_labels: dictionnaire des labels par topic
        n_examples: nombre d'exemples à afficher par topic
    """
    
    # Vue d'ensemble des topics
    st.subheader("📊 Vue d'ensemble des topics")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(topics_info, use_container_width=True)
    
    with col2:
        # Statistiques
        total_topics = len(topic_keywords)
        outliers = len(df_with_topics[df_with_topics['bertopic_topic'] == -1])
        
        st.metric("Topics détectés", total_topics)
        st.metric("Documents outliers", outliers)
        st.metric("% couverture", f"{((len(df_with_topics) - outliers) / len(df_with_topics) * 100):.1f}%")
    
    # Graphique de répartition
    if len(topic_keywords) > 0:
        topic_counts = df_with_topics['bertopic_topic'].value_counts().sort_index()
        # Exclure les outliers (-1) du graphique principal
        topic_counts_filtered = topic_counts[topic_counts.index != -1]
        
        if len(topic_counts_filtered) > 0:
            st.subheader("📈 Répartition des topics")
            st.bar_chart(topic_counts_filtered)
    
    st.markdown("---")
    
    # Détail des topics
    st.subheader("🔍 Analyse détaillée des topics")
    
    if len(topic_keywords) == 0:
        st.warning("Aucun topic significatif détecté. Essayez avec des paramètres différents.")
        return
    
    for topic_id in sorted(topic_keywords.keys()):
        keywords = topic_keywords[topic_id]
        label = topic_labels.get(topic_id, f"Topic {topic_id}")
        topic_count = len(df_with_topics[df_with_topics['bertopic_topic'] == topic_id])
        
        with st.expander(f"🔹 Topic {topic_id}: {label} ({topic_count} documents)", expanded=False):
            
            # Mots-clés avec scores
            st.write("**Mots-clés principaux:**")
            keywords_formatted = []
            for word, score in keywords[:10]:  # Top 10 mots
                keywords_formatted.append(f"{word} ({score:.3f})")
            st.write(", ".join(keywords_formatted))
            
            # Exemples d'avis
            st.write("**Exemples d'avis représentatifs:**")
            examples = get_bert_topic_examples(df_with_topics, topic_id, n_examples=n_examples)
            
            if len(examples) > 0:
                for i, example in enumerate(examples, 1):
                    st.markdown(f"{i}. *{example.strip()}*")
            else:
                st.warning("Aucun exemple disponible pour ce topic.")
            
            st.markdown("---")
    
    # Informations sur les outliers
    if -1 in df_with_topics['bertopic_topic'].values:
        outlier_count = len(df_with_topics[df_with_topics['bertopic_topic'] == -1])
        st.info(f"ℹ️ **Outliers**: {outlier_count} documents n'ont pas pu être assignés à un topic spécifique.")

if __name__ == "__main__":
    main()