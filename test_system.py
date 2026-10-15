import os
import sys
import pandas as pd
import app
import streamlit as st
from unittest.mock import patch, MagicMock

print("=== Test 1: Python Syntax & Imports ===")
print("app.py successfully loaded.")

print("=== Test 2: Dataset Loading & Structure ===")
df = app.dataset
assert len(df) > 4000, f"Expected >4000 movies, got {len(df)}"
assert "title" in df.columns
assert "cleaned_text" in df.columns
print(f"Passed: Dataset loaded with {len(df)} movies.")

print("=== Test 3: Movie Titles Population ===")
titles = df["title"].tolist()
assert "Avatar" in titles
assert "The Dark Knight Rises" in titles
assert "Toy Story" in titles
print("Passed: Popular movie titles are present and populated.")

print("=== Test 4: Recommendation Engine (Multiple Movies) ===")
test_movies = ["Avatar", "The Dark Knight Rises", "Titanic"]
for movie in test_movies:
    recs = app.recommend_similar_movies(movie, top_n=5, api_key="")
    assert len(recs) == 5, f"Expected 5 recs for {movie}, got {len(recs)}"
    assert movie.lower() not in recs["title"].str.lower().tolist(), f"{movie} should not recommend itself"
    print(f"Recommendations for {movie}: {recs['title'].tolist()}")
print("Passed: Multiple movie recommendations calculated successfully.")

print("=== Test 5: Verify Similarity Scores are Sorted & Positive ===")
recs_avatar = app.recommend_similar_movies("Avatar", top_n=5, api_key="")
scores = recs_avatar["similarity_score"].tolist()
assert all(s >= 0 for s in scores), "Scores should be non-negative"
assert scores == sorted(scores, reverse=True), "Scores should be in descending order"
print("Passed: Similarity scores properly computed and sorted.")

print("=== Test 6: Missing API Key & Placeholder Poster Fallback ===")
poster_url = app.fetch_movie_poster(19995, "")
assert "placehold.co" in poster_url
print("Passed: Graceful fallback when API key is missing.")

print("=== Test 7: Simulated TMDB Failure & Timeout ===")
with patch("requests.get", side_effect=Exception("Timeout")):
    fallback = app.fetch_movie_poster(19995, "dummy_key")
    assert "placehold.co" in fallback
print("Passed: Request timeouts gracefully fall back to placeholder without crashing.")

print("=== Test 8: Git & Secrets Safety Check ===")
with open(".gitignore", "r") as f:
    gi_content = f.read()
assert ".env" in gi_content
assert "secrets.toml" in gi_content
print("Passed: .gitignore safely protects credentials.")

print("\n>>> ALL 8 COMPREHENSIVE TESTS PASSED SUCCESSFULLY! <<<")

