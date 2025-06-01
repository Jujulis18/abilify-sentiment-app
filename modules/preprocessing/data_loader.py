import pandas as pd

def load_data(path):
    """Charge les données depuis un fichier CSV"""
    return pd.read_csv(path)

def clean_age(df):
    """Nettoie la colonne Age en supprimant les espaces et les valeurs nulles"""
    df['Age'] = df['Age'].str.strip()
    df = df[df['Age'].notna()]
    return df

def get_data_ranges(df):
    """
    Récupère les plages de valeurs pour les filtres
    
    Args:
        df: DataFrame
    
    Returns:
        dict contenant les ranges et options pour les filtres
    """
    return {
        'age_min': int(df['Age_numeric'].min()),
        'age_max': int(df['Age_numeric'].max()),
        'genders': df['Gender'].unique(),
        'conditions': df['Condition'].dropna().unique()
    }

