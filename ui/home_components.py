import streamlit as st
import pandas as pd
from preprocessing.data_filter import filter_data
from .common_components import (
    create_info_box, 
    create_metric_card, 
    create_section_header,
    display_dataframe_with_info,
    show_warning_message,
    show_success_message,
    create_columns_layout
)

def display_main_header():
    st.markdown(
        '<h1 class="main-header">🏥 Dashboard - Analyse des Avis Médicaux</h1>', 
        unsafe_allow_html=True
    )

def display_navigation_info():
    create_info_box(
        title="🧭 Navigation",
        content="""
        <p>Utilisez la barre latérale pour accéder aux différentes analyses :</p>
        <ul>
            <li><strong>Sentiment Analysis</strong> : Analyse des sentiments des avis patients</li>
            <li><strong>Topic Modeling</strong> : Découverte des thèmes principaux (LDA & BERTopic)</li>
        </ul>
        <p><em>Les données filtrées sur cette page seront utilisées dans toutes les analyses.</em></p>
        """,
        box_type="info"
    )

def display_data_overview(df, ranges):
    """
    Affiche un aperçu des données avec métriques principales
    
    Args:
        df: DataFrame des données
        ranges: Dictionnaire des plages de valeurs
    """
    create_section_header("📊 Aperçu des Données", "Statistiques générales du dataset")
    
    # Métriques principales en 4 colonnes
    col1, col2, col3, col4 = create_columns_layout([1, 1, 1, 1])
    
    with col1:
        create_metric_card(
            title="Total des Avis",
            value=f"{len(df):,}",
            help_text="Nombre total d'avis dans le dataset"
        )
    
    with col2:
        create_metric_card(
            title="Genres",
            value=len(ranges['genders']),
            help_text="Nombre de genres différents"
        )
    
    with col3:
        create_metric_card(
            title="Conditions Médicales",
            value=len(ranges['conditions']),
            help_text="Nombre de conditions médicales différentes"
        )
    
    with col4:
        create_metric_card(
            title="Tranche d'Âge",
            value=f"{ranges['age_min']}-{ranges['age_max']} ans",
            help_text="Étendue des âges dans le dataset"
        )
    
    # Informations complémentaires
    if 'sentiment' in df.columns:
        st.markdown("### Distribution des Sentiments")
        col1, col2, col3 = create_columns_layout([1, 1, 2])
        
        sentiment_counts = df['sentiment'].value_counts()
        
        with col1:
            if 'Positif' in sentiment_counts:
                create_metric_card(
                    title="Avis Positifs",
                    value=f"{sentiment_counts.get('Positif', 0):,}",
                    help_text="Nombre d'avis avec sentiment positif"
                )
        
        with col2:
            if 'Négatif' in sentiment_counts:
                create_metric_card(
                    title="Avis Négatifs", 
                    value=f"{sentiment_counts.get('Négatif', 0):,}",
                    help_text="Nombre d'avis avec sentiment négatif"
                )

def display_data_filters(df, ranges):
    """
    Affiche les filtres de données et retourne les données filtrées
    
    Args:
        df: DataFrame des données originales
        ranges: Dictionnaire des plages de valeurs
    
    Returns:
        DataFrame filtré selon les critères sélectionnés
    """
    create_section_header("🔍 Filtres de Données", "Personnalisez votre analyse")
    
    with st.expander("⚙️ Paramètres de Filtrage", expanded=True):
        col1, col2 = create_columns_layout([1, 1])
        
        with col1:
            st.markdown("**Critères démographiques**")
            
            # Filtre d'âge
            age_range = st.slider(
                "Tranche d'âge",
                min_value=ranges['age_min'],
                max_value=ranges['age_max'],
                value=(ranges['age_min'], ranges['age_max']),
                help="Sélectionnez la tranche d'âge à analyser"
            )
            
            # Filtre de genre
            selected_genders = st.multiselect(
                "Genres",
                options=list(ranges['genders']),
                default=list(ranges['genders']),
                help="Sélectionnez les genres à inclure dans l'analyse"
            )
        
        with col2:
            st.markdown("**Critères médicaux**")
            
            # Filtre de conditions (limité aux 15 plus fréquentes pour l'interface)
            condition_counts = df['Condition'].value_counts()
            top_conditions = condition_counts.head(15).index.tolist()
            
            selected_conditions = st.multiselect(
                "Conditions Médicales (Top 15)",
                options=top_conditions,
                default=top_conditions[:5],  # Sélection par défaut des 5 premières
                help="Sélectionnez les conditions médicales à analyser"
            )
            
            # Option pour inclure toutes les conditions
            include_all_conditions = st.checkbox(
                "Inclure toutes les conditions",
                value=False,
                help="Cochez pour inclure toutes les conditions médicales disponibles"
            )
            
            # Affichage du nombre total de conditions
            st.caption(f"*{len(ranges['conditions'])} conditions disponibles au total*")
    
    # Validation des sélections
    if not selected_genders:
        show_warning_message("Veuillez sélectionner au moins un genre")
        return df.head(0)
    
    if not selected_conditions and not include_all_conditions:
        show_warning_message("Veuillez sélectionner au moins une condition médicale")
        return df.head(0)
    
    # Application des filtres
    conditions_to_use = list(ranges['conditions']) if include_all_conditions else selected_conditions
    
    try:
        filtered_df = filter_data(
            df, 
            age_range=age_range,
            gender_filter=selected_genders,
            condition_filter=conditions_to_use
        )
        
        # Affichage du résultat du filtrage
        if len(filtered_df) > 0:
            show_success_message(
                f"**{len(filtered_df):,}** avis correspondent aux critères sélectionnés "
                f"({len(filtered_df)/len(df)*100:.1f}% du dataset)"
            )
        else:
            show_warning_message("Aucun avis ne correspond aux critères sélectionnés")
        
        return filtered_df
        
    except Exception as e:
        st.error(f"Erreur lors du filtrage : {e}")
        return df.head(0)

def display_sample_data(df):
    """
    Affiche un échantillon des données filtrées
    
    Args:
        df: DataFrame filtré à afficher
    """
    if df.empty:
        create_info_box(
            title="Aucune donnée",
            content="Aucune donnée ne correspond aux critères sélectionnés.",
            box_type="warning"
        )
        return
    
    create_section_header("📋 Échantillon des Données", "Aperçu des données sélectionnées")
    
    # Colonnes à afficher en priorité
    priority_columns = ['Age', 'Gender', 'Condition', 'description-text']
    available_columns = [col for col in priority_columns if col in df.columns]
    
    if not available_columns:
        show_warning_message("Colonnes d'affichage standard non trouvées")
        available_columns = df.columns.tolist()[:4]  # Prendre les 4 premières colonnes
    
    # Affichage de l'échantillon
    sample_size = min(10, len(df))
    sample_df = df[available_columns].head(sample_size)
    
    # Troncature du texte pour l'affichage
    if 'description-text' in sample_df.columns:
        sample_df = sample_df.copy()
        sample_df['description-text'] = sample_df['description-text'].apply(
            lambda x: x[:100] + "..." if isinstance(x, str) and len(x) > 100 else x
        )
    
    display_dataframe_with_info(
        sample_df, 
        title=f"Échantillon ({sample_size} premiers avis)",
        show_shape=True
    )
    
    # Statistiques rapides
    st.markdown("### 📈 Statistiques Rapides")
    col1, col2, col3 = create_columns_layout([1, 1, 1])
    
    with col1:
        if 'Age' in df.columns:
            create_metric_card(
                title="Âge Moyen",
                value=f"{df['Age_numeric'].mean():.1f} ans" if 'Age_numeric' in df.columns else "N/A"
            )
    
    with col2:
        if 'Gender' in df.columns:
            most_common_gender = df['Gender'].value_counts().index[0]
            create_metric_card(
                title="Genre Principal",
                value=most_common_gender
            )
    
    with col3:
        if 'Condition' in df.columns:
            create_metric_card(
                title="Conditions Uniques",
                value=df['Condition'].nunique()
            )

def display_analysis_instructions(filtered_df):
    """
    Affiche les instructions pour procéder aux analyses
    
    Args:
        filtered_df: DataFrame des données filtrées
    """
    if not filtered_df.empty:
        create_info_box(
            title="🚀 Prêt pour l'Analyse",
            content="""
            <p><strong>Vos données sont prêtes !</strong> Utilisez la barre latérale pour accéder aux analyses :</p>
            <ul>
                <li>📊 <strong>Sentiment Analysis</strong> : Découvrez la répartition des sentiments</li>
                <li>🔍 <strong>Topic Modeling</strong> : Identifiez les thèmes principaux avec LDA et BERTopic</li>
            </ul>
            <p><em>Les filtres appliqués seront automatiquement utilisés dans toutes les analyses.</em></p>
            """,
            box_type="success"
        )
    else:
        create_info_box(
            title="Ajustez vos Filtres",
            content="""
            <p>Aucune donnée ne correspond aux critères actuels.</p>
            <p>Essayez d'élargir vos critères de filtrage pour obtenir des résultats.</p>
            """,
            box_type="warning"
        )