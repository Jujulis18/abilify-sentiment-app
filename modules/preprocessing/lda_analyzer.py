import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def prepare_lda_data(df, text_column='clean_review'):
    """
    Prépare les données pour l'analyse LDA
    
    Args:
        df: DataFrame contenant les textes
        text_column: nom de la colonne contenant les textes nettoyés
    
    Returns:
        tuple: (df_cleaned, texts_list)
    """
    df_cleaned = df.dropna(subset=[text_column]).reset_index(drop=True)
    texts = df_cleaned[text_column].tolist()
    return df_cleaned, texts

def run_lda_analysis(texts, n_topics=8, max_df=0.95, min_df=2, random_state=42):
    """
    Exécute l'analyse LDA sur une liste de textes
    
    Args:
        texts: liste des textes à analyser
        n_topics: nombre de topics à extraire
        max_df: fréquence maximale des termes (pour CountVectorizer)
        min_df: fréquence minimale des termes (pour CountVectorizer)
        random_state: graine pour la reproductibilité
    
    Returns:
        tuple: (lda_model, vectorizer, doc_topic_matrix)
    """
    vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, stop_words='english')
    X = vectorizer.fit_transform(texts)
    
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=random_state)
    lda.fit(X)
    
    doc_topics = lda.transform(X)
    
    return lda, vectorizer, doc_topics

def extract_lda_topics(lda_model, vectorizer, n_words=10):
    """
    Extrait les mots-clés pour chaque topic du modèle LDA
    
    Args:
        lda_model: modèle LDA entraîné
        vectorizer: vectorizer utilisé pour l'entraînement
        n_words: nombre de mots-clés par topic
    
    Returns:
        list: liste des mots-clés par topic
    """
    topics = []
    feature_names = vectorizer.get_feature_names_out()
    
    for topic_idx, topic in enumerate(lda_model.components_):
        keywords = [feature_names[i] for i in topic.argsort()[:-n_words - 1:-1]]
        topics.append(keywords)
    
    return topics

def assign_topics_to_documents(df, doc_topics):
    """
    Assigne le topic principal à chaque document
    
    Args:
        df: DataFrame des documents
        doc_topics: matrice des probabilités topic-document
    
    Returns:
        DataFrame avec une colonne 'topic' ajoutée
    """
    df_copy = df.copy()
    df_copy['topic'] = doc_topics.argmax(axis=1)
    return df_copy

def get_topic_examples(df, topic_num, text_column='description-text', n_examples=3, random_state=42):
    """
    Récupère des exemples d'avis pour un topic donné
    
    Args:
        df: DataFrame avec les topics assignés
        topic_num: numéro du topic
        text_column: colonne contenant le texte original
        n_examples: nombre d'exemples à retourner
        random_state: graine pour la reproductibilité
    
    Returns:
        Series: exemples d'avis pour le topic
    """
    topic_docs = df[df['topic'] == topic_num][text_column].dropna()
    n_samples = min(n_examples, len(topic_docs))
    
    if n_samples == 0:
        return pd.Series(dtype=str)
    
    return topic_docs.sample(n_samples, random_state=random_state)
