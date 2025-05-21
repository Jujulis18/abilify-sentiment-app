import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sentiment Analyse - Abilify", layout="wide")

# 📥 Chargement des données
df = pd.read_csv("data/reviews_cleaned.csv")

st.title("Analyse des avis patients sur l'Abilify")
st.markdown("Explorez les avis patients selon l’âge, le genre, et les conditions médicales.")

# Filtres utilisateur
age_range = st.slider("Filtrer par âge", int(df['Age_numeric'].min()), int(df['Age_numeric'].max()), (20, 60))
gender_filter = st.multiselect("Genre", options=df['Gender'].unique(), default=df['Gender'].unique())
condition_filter = st.multiselect("Condition médicale", options=df['Condition'].dropna().unique(), default=df['Condition'].dropna().unique()[:5])

# Filtrage dynamique
filtered_df = df[
    (df['Age_numeric'] >= age_range[0]) &
    (df['Age_numeric'] <= age_range[1]) &
    (df['Gender'].isin(gender_filter)) &
    (df['Condition'].isin(condition_filter))
]

# Visualisation des sentiments
st.subheader("Répartition des sentiments")
fig1 = plt.figure()
sns.countplot(data=filtered_df, x='sentiment', palette='pastel')
st.pyplot(fig1)

# Distribution d'âge selon le sentiment
st.subheader("Âge selon sentiment")
fig2 = plt.figure()
sns.boxplot(data=filtered_df, x='sentiment', y='Age', palette='coolwarm')
st.pyplot(fig2)

# Exemples d'avis
st.subheader("Exemples d'avis")
for i, row in filtered_df[['description-text', 'sentiment']].sample(3).iterrows():
    st.markdown(f"**Sentiment**: {row['sentiment']}")
    st.write(row['description-text'])
    st.markdown("---")
