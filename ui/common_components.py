import streamlit as st
import pandas as pd

def create_metric_card(title, value, help_text=None, delta=None):
    """
    Composant métrique réutilisable
    
    Args:
        title: Titre de la métrique
        value: Valeur à afficher
        help_text: Texte d'aide (optionnel)
        delta: Variation (optionnel)
    """
    st.metric(
        label=title,
        value=value,
        delta=delta,
        help=help_text
    )

def create_info_box(title, content, box_type="info", icon=None):
    """
    Boîte d'information réutilisable
    
    Args:
        title: Titre de la boîte
        content: Contenu (peut être HTML)
        box_type: Type de boîte ("info", "success", "warning")
        icon: Icône à afficher (optionnel)
    """
    # Définition des icônes par défaut
    default_icons = {
        "info": "ℹ️",
        "success": "✅", 
        "warning": "⚠️"
    }
    
    # Définition des classes CSS
    css_classes = {
        "info": "info-box",
        "success": "success-box", 
        "warning": "warning-box"
    }
    
    display_icon = icon or default_icons.get(box_type, "ℹ️")
    css_class = css_classes.get(box_type, "info-box")
    
    st.markdown(f"""
    <div class="{css_class}">
        <h3>{display_icon} {title}</h3>
        {content}
    </div>
    """, unsafe_allow_html=True)

def create_section_header(title, description=None):
    """
    Crée un en-tête de section standardisé
    
    Args:
        title: Titre de la section
        description: Description optionnelle
    """
    st.header(title)
    if description:
        st.markdown(f"*{description}*")

def display_dataframe_with_info(df, title="Données", show_shape=True, **kwargs):
    """
    Affiche un DataFrame avec des informations contextuelles
    
    Args:
        df: DataFrame à afficher
        title: Titre pour les données
        show_shape: Afficher les dimensions
        **kwargs: Arguments pour st.dataframe
    """
    if show_shape:
        st.caption(f"**{title}** - {df.shape[0]} lignes × {df.shape[1]} colonnes")
    
    st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)

def create_loading_spinner(text="Chargement en cours..."):
    """Spinner de chargement standardisé"""
    return st.spinner(text)

def create_columns_layout(ratios):
    """
    Crée une disposition en colonnes avec des ratios personnalisés
    
    Args:
        ratios: Liste des ratios pour les colonnes (ex: [1, 2, 1])
    
    Returns:
        Tuple des colonnes créées
    """
    return st.columns(ratios)

def show_success_message(message, icon="🎉"):
    st.success(f"{icon} {message}")

def show_warning_message(message, icon="⚠️"):
    st.warning(f"{icon} {message}")

def show_error_message(message, icon="❌"):
    st.error(f"{icon} {message}")

def create_download_button(data, filename, label="Télécharger", file_format="csv"):
    """
    Crée un bouton de téléchargement pour différents formats
    
    Args:
        data: Données à télécharger (DataFrame, dict, etc.)
        filename: Nom du fichier
        label: Texte du bouton
        file_format: Format du fichier ("csv", "json")
    """
    if file_format == "csv" and isinstance(data, pd.DataFrame):
        csv = data.to_csv(index=False)
        st.download_button(
            label=label,
            data=csv,
            file_name=f"{filename}.csv",
            mime="text/csv"
        )
    elif file_format == "json":
        import json
        if isinstance(data, pd.DataFrame):
            json_data = data.to_json(orient="records", indent=2)
        else:
            json_data = json.dumps(data, indent=2)
        
        st.download_button(
            label=label,
            data=json_data,
            file_name=f"{filename}.json",
            mime="application/json"
        )

def create_sidebar_filters():
    with st.sidebar:
        st.markdown("### 🔍 Filtres")
        with st.expander("Options de filtrage", expanded=True):
            return st.container()

def display_empty_state(message="Aucune donnée à afficher", icon="📭"):
    st.markdown(f"""
    <div style="text-align: center; padding: 2rem; color: #666;">
        <h2>{icon}</h2>
        <p>{message}</p>
    </div>
    """, unsafe_allow_html=True)