import streamlit as st
import pandas as pd

from modules.preprocessing.data_loader import load_data, get_data_ranges
from modules.preprocessing.data_filter import filter_data, get_sample_reviews

from modules.visualization import create_sentiment_countplot, create_age_sentiment_boxplot
from modules.utils import handle_empty_dataframe

st.set_page_config(page_title="Sentiment Analyse - Abilify", layout="wide")

@st.cache_data
def load_and_cache_data():
    return load_data("data/reviews_cleaned.csv")

df = load_and_cache_data()

st.title("Analyse des avis patients sur l'Abilify")
st.markdown("Explorez les avis patients selon l'âge, le genre, et les conditions médicales.")

# Récupération des plages de données pour les filtres
data_ranges = get_data_ranges(df)

# Filtres utilisateur
age_range = st.slider(
    "Filtrer par âge", 
    data_ranges['age_min'], 
    data_ranges['age_max'], 
    (20, 60)
)

gender_filter = st.multiselect(
    "Genre", 
    options=data_ranges['genders'], 
    default=data_ranges['genders']
)

condition_filter = st.multiselect(
    "Condition médicale", 
    options=data_ranges['conditions'], 
    default=list(data_ranges['conditions'][:5])
)

# Filtrage dynamique des données
filtered_df = filter_data(df, age_range, gender_filter, condition_filter)

col1, col2 = st.columns(2)

# Visualisation des sentiments
with col1:
    st.subheader("Répartition des sentiments")
    if handle_empty_dataframe(filtered_df):
        fig1 = create_sentiment_countplot(filtered_df)
        st.pyplot(fig1)

# Distribution d'âge selon le sentiment
with col2:
    st.subheader("Âge selon sentiment")
    if handle_empty_dataframe(filtered_df):
        fig2 = create_age_sentiment_boxplot(filtered_df)
        st.pyplot(fig2)

# Exemples d'avis
st.subheader("Exemples d'avis")
sentiment_choisi = st.selectbox("Choisissez un type d'avis :", ['Positif', 'Négatif'])

try:
    sample_reviews = get_sample_reviews(df, sentiment_choisi)
    
    st.subheader(f"Exemples d'avis {sentiment_choisi.lower()}s")
    
    for i, row in sample_reviews.iterrows():
        st.markdown(f"**Sentiment**: {row['sentiment']}")
        st.write(row['description-text'])
        st.markdown("---")
        
except ValueError as e:
    st.error(f"Pas assez d'avis {sentiment_choisi.lower()}s disponibles pour générer des exemples.")