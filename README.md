# Movie Recommendation System

A content-based movie recommender that combines **TF-IDF text similarity** with **genre-based discovery**, served through a FastAPI backend and a Streamlit frontend, deployed as two independent cloud services.

**Live demo:** https://movierecommendationsystem-lfejrgervemxssedxxxnr2.streamlit.app

---

## What it does

- Search any movie by keyword, get autocomplete suggestions and a poster grid (via TMDB)
- Open a movie's detail page — overview, genres, backdrop, release info
- Get two kinds of recommendations side by side:
  - **TF-IDF similarity** — recommends movies with similar overviews, taglines, and genres, based on a locally trained TF-IDF vectorizer + cosine similarity
  - **Genre-based** — pulls popular movies from TMDB in the same primary genre

## Tech stack

Python · FastAPI · Streamlit · scikit-learn (TF-IDF, cosine similarity) · pandas · TMDB API · Render · Streamlit Cloud
