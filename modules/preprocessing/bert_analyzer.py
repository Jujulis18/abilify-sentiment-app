import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

def prepare_bert_data(df, text_column='description-text'):
    """
    Prépare les données pour l'analyse BERTopic
    
    Args:
        df: DataFrame contenant les textes
        text_column: nom de la colonne contenant les textes originaux
    
    Returns:
        tuple: (df_cleaned, texts_list)
    """
    df_cleaned = df.dropna(subset=[text_column]).reset_index(drop=True)
    texts = df_cleaned[text_column].tolist()
    return df_cleaned, texts

def run_bert_analysis(texts, n_topics="auto", embedding_model_name="all-MiniLM-L12-v2", verbose=True):
    """
    Exécute l'analyse BERTopic sur une liste de textes
    
    Args:
        texts: liste des textes à analyser
        n_topics: nombre de topics à extraire ("auto" ou nombre)
        embedding_model_name: nom du modèle d'embedding à utiliser
        verbose: afficher les informations de progression
    
    Returns:
        tuple: (topic_model, topics, probabilities)
    """
    embedding_model = SentenceTransformer(embedding_model_name)
    
    topic_model = BERTopic(
        embedding_model=embedding_model, 
        nr_topics=n_topics, 
        verbose=verbose
    )
    
    topics, probs = topic_model.fit_transform(texts)
    
    return topic_model, topics, probs

def extract_bert_topics_info(topic_model):
    """
    Extrait les informations des topics BERTopic
    
    Args:
        topic_model: modèle BERTopic entraîné
    
    Returns:
        tuple: (topics_info_df, topic_keywords_dict)
    """
    topics_info = topic_model.get_topic_info()
    
    # Récupère les mots-clés par topic (sauf topic -1 qui représente les outliers)
    topic_keywords = {
        topic_id: topic_model.get_topic(topic_id)
        for topic_id in topics_info["Topic"].values
        if topic_id != -1
    }
    
    return topics_info, topic_keywords

def generate_bert_topic_labels(topic_keywords, n_words=3):
    """
    Génère des labels pour les topics BERTopic basés sur les mots-clés principaux
    
    Args:
        topic_keywords: dictionnaire des mots-clés par topic
        n_words: nombre de mots à utiliser pour le label
    
    Returns:
        dict: labels par topic_id
    """
    topic_labels = {}
    for topic_id, keywords in topic_keywords.items():
        words = [word for word, _ in keywords[:n_words]]
        topic_labels[topic_id] = ", ".join(words)
    
    return topic_labels

def assign_bert_topics_to_documents(df, topics):
    """
    Assigne les topics BERTopic aux documents
    
    Args:
        df: DataFrame des documents
        topics: liste des topics assignés par BERTopic
    
    Returns:
        DataFrame avec une colonne 'bertopic_topic' ajoutée
    """
    df_copy = df.copy()
    df_copy['bertopic_topic'] = topics
    return df_copy

def get_bert_topic_examples(df, topic_id, text_column='description-text', n_examples=3, random_state=42):
    """
    Récupère des exemples d'avis pour un topic BERTopic donné
    
    Args:
        df: DataFrame avec les topics assignés
        topic_id: ID du topic
        text_column: colonne contenant le texte original
        n_examples: nombre d'exemples à retourner
        random_state: graine pour la reproductibilité
    
    Returns:
        Series: exemples d'avis pour le topic
    """
    topic_docs = df[df['bertopic_topic'] == topic_id][text_column].dropna()
    n_samples = min(n_examples, len(topic_docs))
    
    if n_samples == 0:
        return pd.Series(dtype=str)
    
    return topic_docs.sample(n_samples, random_state=random_state)