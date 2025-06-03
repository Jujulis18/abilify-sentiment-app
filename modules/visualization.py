import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

def create_countplot(data, x_column, title=None, palette='pastel', figsize=(8, 6)):
    """
    Crée un graphique en barres générique
    
    Args:
        data: DataFrame
        x_column: colonne pour l'axe x
        title: titre du graphique
        palette: palette de couleurs
        figsize: taille de la figure
    
    Returns:
        Figure matplotlib
    """
    fig = plt.figure(figsize=figsize)
    sns.countplot(data=data, x=x_column, palette=palette)
    if title:
        plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def create_boxplot(data, x_column, y_column, title=None, palette='coolwarm', figsize=(8, 6)):
    """
    Crée un boxplot générique
    
    Args:
        data: DataFrame
        x_column: colonne pour l'axe x (catégorielle)
        y_column: colonne pour l'axe y (numérique)
        title: titre du graphique
        palette: palette de couleurs
        figsize: taille de la figure
    
    Returns:
        Figure matplotlib
    """
    fig = plt.figure(figsize=figsize)
    sns.boxplot(data=data, x=x_column, y=y_column, palette=palette)
    if title:
        plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def create_histogram(data, column, title=None, bins=30, figsize=(8, 6)):
    """
    Crée un histogramme
    
    Args:
        data: DataFrame
        column: colonne à représenter
        title: titre du graphique
        bins: nombre de bins
        figsize: taille de la figure
    
    Returns:
        Figure matplotlib
    """
    fig = plt.figure(figsize=figsize)
    plt.hist(data[column].dropna(), bins=bins, alpha=0.7, edgecolor='black')
    if title:
        plt.title(title)
    plt.xlabel(column)
    plt.ylabel('Fréquence')
    plt.tight_layout()
    return fig

def create_interactive_countplot(data, x_column, title=None):
    """
    Crée un graphique en barres interactif avec Plotly
    
    Args:
        data: DataFrame
        x_column: colonne pour l'axe x
        title: titre du graphique
    
    Returns:
        Figure Plotly
    """
    counts = data[x_column].value_counts()
    fig = px.bar(
        x=counts.index, 
        y=counts.values,
        title=title or f"Distribution de {x_column}",
        labels={'x': x_column, 'y': 'Nombre'}
    )
    return fig


def create_sentiment_countplot(data, figsize=(5, 3)):
    """Fonction spécifique pour les sentiments (wrapper de la fonction générique)"""
    return create_countplot(
        data=data, 
        x_column='sentiment', 
        title='Répartition des sentiments',
        figsize=figsize
    )

def create_age_sentiment_boxplot(data, figsize=(5, 3)):
    """Fonction spécifique pour âge vs sentiment (wrapper de la fonction générique)"""
    return create_boxplot(
        data=data,
        x_column='sentiment',
        y_column='Age_numeric',
        title='Distribution de l\'âge selon le sentiment',
        figsize=figsize
    )