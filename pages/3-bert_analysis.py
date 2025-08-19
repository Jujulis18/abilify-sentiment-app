import streamlit as st
import pandas as pd
from bertopic import BERTopic

from modules.preprocessing.data_loader import load_data

#from modules.utils import handle_empty_dataframe, clean_text

st.set_page_config(page_title="Analyse BERTopic", layout="wide")


@st.cache_data
def run_cached_bert_analysis(texts_tuple, n_topics, embedding_model_name):
    """Cache l'analyse BERTopic car elle est très coûteuse en calcul"""
    #texts = list(texts_tuple)  # Reconvertir le tuple en liste
    #return run_bert_analysis(texts, n_topics=n_topics, embedding_model_name=embedding_model_name, verbose=False)

def main():
    st.title("Insights BERTopic")
    st.markdown("Découvrez les sujets émergents avec l'analyse BERTopic basée sur les embeddings.")
    
    df_negative = load_data("data/topic_info_neg.csv")
    df_positive = load_data("data/topic_info_pos.csv")

# negatif
    topic_model = BERTopic.load("data/bert_model_neg")
    topics_info_neg = topic_model.get_topic_info()


    # Construire dictionnaire mots-clés par topic
    topic_keywords_neg = {
        row["Topic"]: row["Representation"].split(", ") 
        for _, row in df_negative.iterrows() if row["Topic"] != -1
    }

    # Construire dictionnaire labels (nom simplifié du topic)
    topic_labels_neg = {
        row["Topic"]: row["Name"] if "Name" in df_negative.columns else row["Representation"] 
        for _, row in df_negative.iterrows() if row["Topic"] != -1
    }

# Positif

    topic_model = BERTopic.load("data/bert_model_pos")
    topics_info_pos = topic_model.get_topic_info()

    # Construire dictionnaire mots-clés par topic
    topic_keywords_pos = {
        row["Topic"]: row["Representation"].split(", ") 
        for _, row in df_positive.iterrows() if row["Topic"] != -1
    }

    # Construire dictionnaire labels (nom simplifié du topic)
    topic_labels_pos = {
        row["Topic"]: row["Name"] if "Name" in df_positive.columns else row["Representation"] 
        for _, row in df_positive.iterrows() if row["Topic"] != -1
    }


    st.subheader("Paramètres BERTopic")
        
    # Option pour le nombre de topics
    #auto_topics = st.checkbox("Détection automatique des topics", value=True)
    #if auto_topics:
    #    n_topics = "auto"
    #else:
    n_topics = st.slider("Nombre de topics", min_value=5, max_value=20, value=10)
    
    # Modèle d'embedding
    #embedding_options = {
    #    "all-MiniLM-L12-v2": "MiniLM (rapide)",
    #    "all-mpnet-base-v2": "MPNet (meilleur)",
    #    "paraphrase-MiniLM-L6-v2": "Paraphrase (léger)"
    #}
    
    #selected_model = st.selectbox(
    #    "Modèle d'embedding",
    #    options=list(embedding_options.keys()),
    #    format_func=lambda x: embedding_options[x],
    #    index=0
    #)
    
    n_examples = st.slider("Exemples par topic", min_value=1, max_value=5, value=3)
    #n_label_words = st.slider("Mots pour les labels", min_value=2, max_value=5, value=3)

    st.warning("Le model a été entrainé sur les avis bruts. Une prochaine version montrera la différence avec un pretraitement qui retirera les endwords et les mots les plus récurrents")
    st.warning("Vous pourrez très prochainement jouer avec les parametres d'entrainement du model")
   
    
    # Paramètres de l'analyse BERTopic
    col1, col2 = st.columns([2, 2])
    
    
    with col1:
        # Préparation des données
        #df_cleaned, texts = prepare_bert_data(df, text_column='description-text')
                
        # Informations sur les données
        st.info(f"📊 Analyse BERTopic sur {df_negative['Count'].sum()} avis négatifs")
        
        # Analyse BERTopic
        #with st.spinner("Exécution de l'analyse BERTopic... (cela peut prendre du temps)"):
            # Convertir en tuple pour le cache
        #    texts_tuple = tuple(texts)
        #    topic_model, topics, probs = run_cached_bert_analysis(texts_tuple, n_topics, selected_model)
        
        # Extraction des informations des topics
        #topics_info, topic_keywords = extract_bert_topics_info(topic_model)
        
        # Génération des labels
        #topic_labels = generate_bert_topic_labels(topic_keywords, n_words=n_label_words)
        
        # Assignation des topics aux documents
        #df_with_topics = assign_bert_topics_to_documents(df_cleaned, topics)
        
        # Affichage des résultats
        display_bert_results(df_negative, topics_info_neg, topic_keywords_neg, topic_labels_neg, n_examples)

    with col2:
        st.info(f"📊 Analyse BERTopic sur {df_positive['Count'].sum()} avis positifs")
        display_bert_results(df_positive, topics_info_pos, topic_keywords_pos, topic_labels_pos, n_examples)



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
        outliers = len(df_with_topics[df_with_topics['Topic'] == -1])
        
        st.metric("Topics détectés", total_topics)
        st.metric("Documents outliers", outliers)

        coverage = (df_with_topics['Count'].sum() - df_with_topics.iloc[0]['Count']) / df_with_topics['Count'].sum() * 100
        st.metric("% couverture", f"{coverage:.1f}%")
    
    # Graphique de répartition
    topic_counts_filtered = topics_info[topics_info["Topic"] != -1].set_index("Topic")["Count"]

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
        topic_count = len(df_with_topics[df_with_topics['Topic'] == topic_id])
        
        with st.expander(f"🔹 Topic {topic_id}: {label} ({topic_count} documents)", expanded=False):
            
            # Mots-clés avec scores
            st.write("**Mots-clés principaux:**")
            st.write(df_with_topics["Representation"][topic_id+1])

            
            # Exemples d'avis
            st.write("**Exemples d'avis représentatifs:**")

            rep_docs = df_with_topics["Representative_Docs"]
            #rep_docs = ast.literal_eval(rep_docs)  # convertit la string en liste python
               

            # Afficher seulement les N premiers exemples (par ex.)
            for i, doc in enumerate(rep_docs[:n_examples], 1):
                st.markdown(f"_{i}. {doc}_")
            
            
            st.markdown("---")
    
    # Informations sur les outliers
    if -1 in df_with_topics['Topic'].values:
        outlier_count = len(df_with_topics[df_with_topics['Topic'] == -1])
        st.info(f"ℹ️ **Outliers**: {outlier_count} documents n'ont pas pu être assignés à un topic spécifique.")




if __name__ == "__main__":
    main()