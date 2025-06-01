import streamlit as st
import pandas as pd

from ui.styles import load_custom_css, get_page_config
from ui.home_components import (
    display_main_header,
    display_navigation_info,
    display_data_overview,
    display_data_filters,
    display_sample_data,
    display_analysis_instructions
)

from modules.preprocessing.data_loader import load_data, clean_age, get_data_ranges

config = get_page_config()
st.set_page_config(**config)

load_custom_css()

@st.cache_data
def load_and_process_data():
    try:
        df = load_data('data/reviews_cleaned.csv')
        df = clean_age(df)
        
        # Calcul des métriques globales
        ranges = get_data_ranges(df)
        
        return df, ranges
        
    except FileNotFoundError:
        st.error("❌ Fichier de données introuvable. Vérifiez le chemin dans `load_data()`")
        return None, None
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des données : {e}")
        return None, None

def main():

    display_main_header()
    display_navigation_info()
    
    # Chargement des données
    with st.spinner("🔄 Chargement des données..."):
        df, ranges = load_and_process_data()
    
    if df is None or ranges is None:
        st.stop()  
    
    display_data_overview(df, ranges)
    st.divider()
    filtered_df = display_data_filters(df, ranges)
    
    # Stockage des données dans la session pour les autres pages
    st.session_state['filtered_data'] = filtered_df
    st.session_state['original_data'] = df
    st.session_state['data_ranges'] = ranges
    
    st.divider()
    display_sample_data(filtered_df)
    st.divider()
    display_analysis_instructions(filtered_df)

if __name__ == "__main__":
    main()