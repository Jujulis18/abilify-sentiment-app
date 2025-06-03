import streamlit as st

def load_custom_css():
    st.markdown("""
    <style>
        /* Titre principal */
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            margin-bottom: 2rem;
            color: #1f77b4;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Cartes métriques */
        .metric-card {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Boîtes d'information */
        .info-box {
            background-color: #e8f4fd;
            border-left: 5px solid #1f77b4;
            border-radius: 5px;
            padding: 1rem;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .info-box h3 {
            margin-top: 0;
            color: #1f77b4;
        }
        
        /* Boîte de succès */
        .success-box {
            background-color: #d4edda;
            border-left: 5px solid #28a745;
            border-radius: 5px;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        /* Boîte d'avertissement */
        .warning-box {
            background-color: #fff3cd;
            border-left: 5px solid #ffc107;
            border-radius: 5px;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        /* Amélioration des expanders */
        .streamlit-expanderHeader {
            font-weight: bold;
            font-size: 1.1rem;
        }
        
        /* Styles pour les dataframes */
        .stDataFrame {
            border-radius: 10px;
            overflow: hidden;
        }
        
        /* Sidebar personnalisée */
        .css-1d391kg {
            background-color: #f8f9fa;
        }
        
        /* Boutons personnalisés */
        .stButton > button {
            border-radius: 20px;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        /* Amélioration du selectbox et multiselect */
        .stSelectbox > div > div {
            border-radius: 10px;
        }
        
        .stMultiSelect > div > div {
            border-radius: 10px;
        }
        
        /* Animation pour les métriques */
        .metric-container {
            transition: transform 0.2s ease;
        }
        
        .metric-container:hover {
            transform: scale(1.02);
        }
    </style>
    """, unsafe_allow_html=True)

def get_page_config():
    return {
        "page_title": "Dashboard - Analyse des Avis Médicaux",
        "page_icon": "🏥",
        "layout": "wide",
        "initial_sidebar_state": "expanded"
    }