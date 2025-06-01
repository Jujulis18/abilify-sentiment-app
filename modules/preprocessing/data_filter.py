def filter_data(df, age_range, gender_filter, condition_filter):
    """
    Filtre les données selon l'âge, le genre et la condition médicale
    
    Args:
        df: DataFrame à filtrer
        age_range: tuple (min_age, max_age)
        gender_filter: liste des genres à inclure
        condition_filter: liste des conditions médicales à inclure
    
    Returns:
        DataFrame filtré
    """
    filtered_df = df[
        (df['Age_numeric'] >= age_range[0]) &
        (df['Age_numeric'] <= age_range[1]) &
        (df['Gender'].isin(gender_filter)) &
        (df['Condition'].isin(condition_filter))
    ]
    return filtered_df

def get_sample_reviews(df, sentiment_type, n_samples=3, random_state=1):
    """
    Récupère un échantillon d'avis pour un type de sentiment donné
    
    Args:
        df: DataFrame contenant les avis
        sentiment_type: 'Positif' ou 'Négatif'
        n_samples: nombre d'échantillons à retourner
        random_state: graine pour la reproductibilité
    
    Returns:
        DataFrame avec les échantillons d'avis
    """
    avis_filtrés = df[df['sentiment'] == sentiment_type]
    return avis_filtrés[['description-text', 'sentiment']].sample(n_samples, random_state=random_state)
