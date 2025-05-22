from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def run_lda(texts, n_topics=8, top_words=10):
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    X = vectorizer.fit_transform(texts)
    
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        keywords = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-top_words - 1:-1]]
        topics.append((topic_idx, keywords))
    return topics
