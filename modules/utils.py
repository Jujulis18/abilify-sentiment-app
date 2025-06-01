import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def safe_sample(df, n_samples=3, random_state=1):
    """
    Échantillonne un DataFrame de manière sécurisée
    
    Args:
        df: DataFrame à échantillonner
        n_samples: nombre d'échantillons souhaités
        random_state: graine pour la reproductibilité
    
    Returns:
        DataFrame échantillonné ou DataFrame complet si pas assez de lignes
    """
    if len(df) >= n_samples:
        return df.sample(n_samples, random_state=random_state)
    else:
        return df

def display_dataframe_info(df, title="Informations sur les données"):
    """
    Affiche des informations générales sur un DataFrame
    
    Args:
        df: DataFrame à analyser
        title: titre de la section
    """
    st.subheader(title)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Nombre de lignes", len(df))
    with col2:
        st.metric("Nombre de colonnes", len(df.columns))
    with col3:
        st.metric("Valeurs manquantes", df.isnull().sum().sum())

def get_unique_values(df, column):
    """
    Récupère les valeurs uniques d'une colonne en gérant les NaN
    
    Args:
        df: DataFrame
        column: nom de la colonne
    
    Returns:
        array des valeurs uniques (sans NaN)
    """
    return df[column].dropna().unique()

def filter_dataframe_by_column(df, column, values):
    """
    Filtre un DataFrame par une colonne avec une liste de valeurs
    
    Args:
        df: DataFrame à filtrer
        column: nom de la colonne
        values: liste des valeurs à garder
    
    Returns:
        DataFrame filtré
    """
    if not values:  # Si la liste est vide, retourner DataFrame vide
        return df.iloc[0:0]
    return df[df[column].isin(values)]

def handle_empty_dataframe(df, message="Aucune donnée ne correspond aux critères sélectionnés."):
    """
    Vérifie si un DataFrame est vide et affiche un message si c'est le cas
    
    Args:
        df: DataFrame à vérifier
        message: message à afficher si vide
    
    Returns:
        bool: True si le DataFrame n'est pas vide, False sinon
    """
    if df.empty:
        st.warning(message)
        return False
    return True

def create_download_button(data, filename, button_text="Télécharger les données"):
    """
    Crée un bouton de téléchargement pour un DataFrame
    
    Args:
        data: DataFrame à télécharger
        filename: nom du fichier (avec extension)
        button_text: texte du bouton
    """
    if filename.endswith('.csv'):
        file_data = data.to_csv(index=False)
        mime_type = 'text/csv'
    elif filename.endswith('.json'):
        file_data = data.to_json(orient='records', indent=2)
        mime_type = 'application/json'
    else:
        raise ValueError("Format de fichier non supporté. Utilisez .csv ou .json")
    
    st.download_button(
        label=button_text,
        data=file_data,
        file_name=filename,
        mime=mime_type
    )

def clean_text(text):
    """
    Nettoie un texte en supprimant les mots vides et les mots courts
    
    Args:
        text: texte à nettoyer
    
    Returns:
        str: texte nettoyé
    """
    if pd.isna(text):
        return ""
    words = text.lower().split()
    return " ".join([w for w in words if w not in ENGLISH_STOP_WORDS and len(w) > 2])