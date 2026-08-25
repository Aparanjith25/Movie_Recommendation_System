import streamlit as st
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load files
df = pickle.load(open("df.pkl", "rb"))
indices = pickle.load(open("indices.pkl", "rb"))
tfidf_matrix = pickle.load(open("tfidf_matrix.pkl", "rb"))


def recommend(title, n=10):

    if title not in indices:
        return []

    idx = indices[title]

    similarity_scores = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    similar_indices = np.argsort(similarity_scores)[::-1]

    similar_indices = [
        i for i in similar_indices
        if i != idx
    ]

    top_indices = similar_indices[:n]

    return df["title"].iloc[top_indices].tolist()


# UI
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Recommendation System")

st.write(
    "Find movies similar to your favourite movie "
    "using TF-IDF and Cosine Similarity."
)

movie_titles = sorted(
    df["title"].dropna().unique()
)

movie = st.selectbox(
    "🎥 Select a movie",
    movie_titles
)

n = st.slider(
    "Number of recommendations",
    5,
    20,
    10
)

if st.button("🔍 Recommend"):

    recommendations = recommend(movie, n)

    st.subheader(
        f"Recommendations for {movie}"
    )

    for i, title in enumerate(recommendations, 1):
        st.write(f"**{i}. {title}**")