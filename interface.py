import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import secrets
import streamlit as st
import warnings

from matplotlib.patches import Patch
from numpy.linalg import norm
from sklearn.impute import KNNImputer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# Supprime les messages d'informations (pas des erreurs, juste avertissement)
warnings.filterwarnings("ignore")

# Variables streamlit
st.session_state.setdefault('user', None)
st.session_state.setdefault('imputer', None)
st.session_state.setdefault('merged_df', None)
st.session_state.setdefault('sorted_merged_df', None)
st.session_state.setdefault('filter_user', None)
st.session_state.setdefault('style_the_more_see', None)
st.session_state.setdefault('df_matrix', None)
st.session_state.setdefault('df_matrix_imputed', None)
st.session_state.setdefault('movie_similarity', None)
st.session_state.setdefault('movie_similarity_df', None)
st.session_state.setdefault('B',None)
st.session_state.setdefault('item_based_prediction', None)
st.session_state.setdefault('user_df', None)
st.session_state.setdefault('movies', None)
st.session_state.setdefault('edges', None)


# Fonction modifiée pour recommander des films avec un pourcentage de recommandation
def recommend_movies_with_percentage(user_id, data, top_n=10):
    # Récupérer les films déjà vus par cet utilisateur
    user_data = data[data['userId'] == user_id]
    watched_movies = set(user_data['movieId'])

    # Récupérer les genres préférés de l'utilisateur avec leur fréquence
    top_genres = get_top_genres(user_id, data)
    total_genre_count = top_genres.sum()  # Nombre total de genres vus par l'utilisateur

    # Filtrer les films non vus
    unwatched_movies = data[~data['movieId'].isin(watched_movies)].copy()
    unwatched_movies['genres'] = unwatched_movies['genres'].str.split('|')

    # Fonction pour calculer le pourcentage de recommandation en fonction des genres préférés
    def calculate_percentage(genres):
        genre_match_count = sum([top_genres.get(g, 0) for g in genres])  # Somme des fréquences des genres
        return (genre_match_count / total_genre_count) * 100 if total_genre_count > 0 else 0

    # Ajouter une colonne 'recommendation_percentage' avec le pourcentage pour chaque film
    unwatched_movies['recommendation_percentage'] = unwatched_movies['genres'].apply(calculate_percentage)

    # Filtrer les films qui ont un pourcentage supérieur à zéro
    recommended = unwatched_movies[unwatched_movies['recommendation_percentage'] > 0]

    # Supprimer les doublons et trier les films par pourcentage décroissant
    recommended = recommended.drop_duplicates(subset='movieId').sort_values(by='recommendation_percentage', ascending=False)

    # Retourner les titres des films, genres et pourcentage de recommandation
    return recommended[['title', 'genres', 'recommendation_percentage']].head(top_n)


# Fonction pour calculer les genres les plus regardés par un utilisateur
def get_top_genres(user_id, data):
    # Filtrer les films vus par cet utilisateur
    user_data = data[data['userId'] == user_id]

    # Extraire et compter les genres
    genres = user_data['genres'].str.split('|').explode()
    top_genres = genres.value_counts()

    return top_genres


# Afficher les films recommandés avec un format personnalisé
def display_recommendations(recommendations):
    list_movie = []
    for i, row in enumerate(recommendations.itertuples(), start=1):
        list_movie.append(f"{i}) *{row.title}*\n\n ({', '.join(row.genres)}) \n**{row.recommendation_percentage:.2f}%**")
    return list_movie


df1 = 'asset/movies.csv'
df2 = 'asset/ratings.csv'

# 2) Ouverture des csv avec pandas
df1 = pd.read_csv(df1)
df2 = pd.read_csv(df2)

# Suppression de la colonne timestamp qui ne nous sert pas
df2 = df2.drop('timestamp', axis=1)

# Création d'un dataset rassemblabnt les informations nécessaires
st.session_state.merged_df = df2.merge(df1)
st.session_state.merged_df.to_csv('merged.csv')
# ordre ascending
st.session_state.sorted_merged_df = st.session_state.merged_df.sort_values(by='userId', ascending=True)

st.session_state.filter_user = st.session_state.sorted_merged_df[(st.session_state.sorted_merged_df['userId'] == st.session_state.user)]

st.session_state.style_the_more_see = dict(st.session_state.filter_user['genres'].str.get_dummies(sep='|').sum())

# Matrice utilisateur-film
st.session_state.df_matrix = st.session_state.merged_df.pivot_table(index='userId', columns='title', values='rating')

# Sidebar pour saisir l'utilisateur
st.session_state.user = st.sidebar.number_input(
    'Id of the user :arrow_heading_down:: ',
    format="%0f",
    min_value=float(min(st.session_state.sorted_merged_df['userId']))-1,
    max_value=float(max(st.session_state.sorted_merged_df['userId']))
)

# Vérifier si l'utilisateur est défini et calculer les recommandations
if st.session_state.user:

    # Header personnalisé
    st.header(body="Bonjour user " + str(int(st.session_state.user)), divider=True)

    # Filtrage des données de l'utilisateur
    st.session_state.filter_user = st.session_state.sorted_merged_df[
        st.session_state.sorted_merged_df['userId'] == st.session_state.user
    ]

    # Calcul et affichage des statistiques
    st.session_state.style_the_more_see = dict(st.session_state.filter_user['genres'].str.get_dummies(sep='|').sum())
    st.write(f"Nombres de films vus jusqu'à maintenant : {st.session_state.filter_user['title'].count()}")
    st.bar_chart(data=st.session_state.style_the_more_see)

    # Appliquer KNN pour compléter les notes manquantes et générer des recommandations
    st.session_state.imputer = KNNImputer(n_neighbors=20)
    st.session_state.df_matrix_imputed = st.session_state.imputer.fit_transform(st.session_state.df_matrix)

    st.session_state.df_matrix_imputed = pd.DataFrame(
        st.session_state.df_matrix_imputed, 
        index=st.session_state.df_matrix.index, 
        columns=st.session_state.df_matrix.columns
    )

    # Calcul de la similarité des films
    st.session_state.movie_similarity = cosine_similarity(st.session_state.df_matrix_imputed.T)
    st.session_state.movie_similarity_df = pd.DataFrame(
        st.session_state.movie_similarity,
        index=st.session_state.df_matrix.columns,
        columns=st.session_state.df_matrix.columns
    )

    # Calcul des genres préférés et des recommandations basées sur les genres
    top_genres_user1 = get_top_genres(st.session_state.user, st.session_state.merged_df)
    recommended_movies_user_with_percentage = recommend_movies_with_percentage(st.session_state.user, st.session_state.merged_df)

    # Affichage des recommandations
    dico_1 = display_recommendations(recommended_movies_user_with_percentage)
    gauche, droite = st.columns(2)
    with gauche:
        for i in range(0, len(dico_1), 2):
            st.write(dico_1[i])

    with droite:
        for i in range(1, len(dico_1), 2):
            st.write(dico_1[i])
