"""
Movie Recommendation System Using NLP and Python
------------------------------------------------
Developed by:
  - Rugved Kulkarni
  - Siddhant Gaikwad

A content-based movie recommender built with Python, Streamlit,
Natural Language Processing (NLP), TF-IDF vectorization, and Cosine Similarity.
"""

import os
import re
import requests
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer

# Load local environment variables from .env if available
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Movie Recommendation System | NLP",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark cinematic styling
CUSTOM_CSS = """
<style>
/* Main app background and typography */
.main {
    background-color: #0b0f19;
    color: #e2e8f0;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Hero Section */
.hero-container {
    background: linear-gradient(135deg, #111827 0%, #1e1e38 50%, #0f172a 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 32px 28px;
    margin-bottom: 24px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    text-align: center;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #f8fafc 0%, #38bdf8 50%, #f43f5e 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    max-width: 720px;
    margin: 0 auto 16px auto;
    line-height: 1.6;
}
.tech-pill-container {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 8px;
    margin-top: 12px;
}
.tech-pill {
    background: rgba(56, 189, 248, 0.1);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.25);
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
}

/* Movie Card Styles */
.movie-card {
    background: #151c2c;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    overflow: hidden;
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    margin-bottom: 20px;
    display: flex;
    flex-direction: column;
    height: 100%;
}
.movie-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 16px 30px rgba(0, 0, 0, 0.6);
    border-color: rgba(56, 189, 248, 0.4);
}
.card-content {
    padding: 14px 16px 16px 16px;
    display: flex;
    flex-direction: column;
    flex-grow: 1;
}
.movie-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 6px;
    line-height: 1.35;
    height: 2.7em;
    overflow: hidden;
    text-overflow: ellipsis;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
}
.meta-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.8rem;
    color: #94a3b8;
    margin-bottom: 8px;
}
.score-badge {
    background: linear-gradient(135deg, #059669 0%, #10b981 100%);
    color: #ffffff;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 700;
}
.genre-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-bottom: 8px;
    min-height: 24px;
}
.genre-badge {
    background: #1e293b;
    color: #cbd5e1;
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 4px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}
.overview-text {
    font-size: 0.8rem;
    color: #94a3b8;
    line-height: 1.5;
    overflow: hidden;
    text-overflow: ellipsis;
    display: -webkit-box;
    -webkit-line-clamp: 4;
    -webkit-box-orient: vertical;
    margin-top: auto;
}

/* Footer */
.app-footer {
    border-top: 1px solid rgba(255, 255, 255, 0.08);
    margin-top: 48px;
    padding: 28px 16px;
    text-align: center;
    color: #64748b;
    font-size: 0.85rem;
}
.footer-devs {
    color: #e2e8f0;
    font-weight: 600;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Download necessary NLTK resources quietly and safely
@st.cache_resource(show_spinner=False)
def ensure_nltk_corpora():
    for pkg in ('punkt', 'punkt_tab', 'stopwords'):
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

ensure_nltk_corpora()


def get_tmdb_api_key() -> str:
    """
    Retrieve the TMDB API key safely:
    1. Read from system / Windows environment variable 'TMDB_API_KEY' (highest priority).
    2. Fall back to Streamlit Secrets ('st.secrets[\"TMDB_API_KEY\"]') for cloud deployment.
    3. Return empty string if neither is set.
    """
    env_key = os.getenv("TMDB_API_KEY", "").strip()
    if env_key and env_key != "Your Api Key here":
        return env_key

    try:
        if hasattr(st, "secrets") and "TMDB_API_KEY" in st.secrets:
            secret_key = str(st.secrets["TMDB_API_KEY"]).strip()
            if secret_key and secret_key != "Your Api Key here":
                return secret_key
    except Exception:
        pass

    return ""


TMDB_API_KEY = get_tmdb_api_key()


# Preprocessing setup
stop_words = set(stopwords.words('english'))
tokenizer = TreebankWordTokenizer()


def preprocess_text(text: str) -> str:
    """Clean, lowercase, tokenize, and remove English stopwords."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    filtered = [word for word in tokens if word not in stop_words]
    return " ".join(filtered)


@st.cache_data(show_spinner="Loading and preprocessing movie metadata...")
def load_and_preprocess_dataset():
    """
    Load movies.csv and extract relevant textual features for content-based matching.
    Cached to ensure near-instantaneous subsequent runs.
    """
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "movies.csv")
    if not os.path.exists(data_path):
        data_path = "movies.csv"

    df = pd.read_csv(data_path)
    
    # Filter for necessary columns and drop missing text records
    target_cols = ['id', 'title', 'genres', 'keywords', 'overview', 'vote_average', 'release_date']
    available_cols = [c for c in target_cols if c in df.columns]
    df = df[available_cols].copy()

    # Fill metadata defaults
    if 'vote_average' in df.columns:
        df['vote_average'] = df['vote_average'].fillna(0.0)
    else:
        df['vote_average'] = 0.0

    if 'release_date' in df.columns:
        df['release_date'] = df['release_date'].fillna("").astype(str).str.slice(0, 4)
    else:
        df['release_date'] = ""

    # Text features required for similarity
    for text_col in ['genres', 'keywords', 'overview']:
        if text_col in df.columns:
            df[text_col] = df[text_col].fillna("")
        else:
            df[text_col] = ""

    df = df[df['title'].notna() & (df['title'].str.strip() != "")].reset_index(drop=True)

    # Combine text features into single content block
    df['combined'] = df['genres'] + " " + df['keywords'] + " " + df['overview']
    df['cleaned_text'] = df['combined'].apply(preprocess_text)
    return df


@st.cache_resource(show_spinner="Fitting TF-IDF vectorizer model...")
def fit_tfidf_model(cleaned_texts):
    """
    Fit a TF-IDF Vectorizer with max_features=5000 and return the sparse feature matrix.
    Keeping the matrix sparse avoids allocating ~150MB+ for a dense similarity matrix,
    drastically reducing memory overhead for Streamlit Community Cloud.
    """
    vectorizer = TfidfVectorizer(max_features=5000)
    matrix = vectorizer.fit_transform(cleaned_texts)
    return vectorizer, matrix


# Load dataset and TF-IDF representations
dataset = load_and_preprocess_dataset()
tfidf_model, tfidf_matrix = fit_tfidf_model(dataset['cleaned_text'])


@st.cache_data(show_spinner=False)
def fetch_movie_poster(movie_id: int, api_key: str) -> str:
    """
    Dynamically fetch movie poster URL from TMDb API for a given movie ID.
    Gracefully falls back to a dark placeholder image if API key is absent,
    or if request times out or fails.
    """
    placeholder = "https://placehold.co/400x600/181f2c/ffffff?text=No+Poster"
    if not api_key:
        return placeholder

    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}"
        response = requests.get(url, timeout=3.5)
        if response.status_code == 200:
            data = response.json()
            poster_path = data.get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception:
        pass

    return placeholder


def recommend_similar_movies(movie_title: str, top_n: int = 5, api_key: str = "") -> pd.DataFrame:
    """
    Computes cosine similarity on-the-fly for the selected movie against the sparse
    TF-IDF matrix. This avoids storing a massive dense matrix in memory while taking
    only ~10ms to calculate.
    """
    matches = dataset[dataset['title'].str.lower() == movie_title.strip().lower()]
    if matches.empty:
        return pd.DataFrame()

    idx = matches.index[0]
    # Efficient on-the-fly cosine similarity for just the selected movie row
    target_vector = tfidf_matrix[idx]
    similarities = cosine_similarity(target_vector, tfidf_matrix).flatten()

    # Sort descending and exclude the movie itself (rank 0)
    ranked_indices = similarities.argsort()[::-1][1:top_n + 1]

    recommendations = dataset.iloc[ranked_indices].copy()
    recommendations['similarity_score'] = [round(float(similarities[i]) * 100, 1) for i in ranked_indices]
    recommendations['poster_url'] = [fetch_movie_poster(int(m_id), api_key) for m_id in recommendations['id']]
    return recommendations


# ==========================================
# Sidebar: Project Info, Status & Settings
# ==========================================
with st.sidebar:
    st.markdown("### ⚙️ System Status")
    if TMDB_API_KEY:
        st.success("✅ **TMDb API Connected**\n\nHigh-res movie posters enabled.")
    else:
        st.info(
            "ℹ️ **TMDb Posters Disabled**\n\n"
            "Recommendations remain 100% active using local metadata.\n\n"
            "**To enable movie posters:**\n"
            "• **Locally:** Set `TMDB_API_KEY` in Windows environment variables.\n"
            "• **Streamlit Cloud:** Add `TMDB_API_KEY = \"...\"` in App Secrets."
        )

    st.markdown("---")
    st.markdown("### 📊 Dataset Overview")
    st.write(f"• **Indexed Movies:** `{len(dataset):,}`")
    st.write("• **Algorithm:** Content-Based (TF-IDF + Cosine Similarity)")
    st.write("• **Vocabulary Size:** `5,000` features")

    st.markdown("---")
    st.markdown("### 👥 Development Team")
    st.markdown(
        """
        - **Rugved Kulkarni**
        - **Siddhant Gaikwad**
        """
    )


# ==========================================
# Main Content
# ==========================================

# Hero Banner
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">🎬 Movie Recommendation System</div>
        <div class="hero-subtitle">
            Discover your next favorite movie using Natural Language Processing.
            Our content-based engine analyzes plot overviews, genres, and keywords
            to match films with high contextual similarity.
        </div>
        <div class="tech-pill-container">
            <span class="tech-pill">Python</span>
            <span class="tech-pill">Streamlit</span>
            <span class="tech-pill">NLP</span>
            <span class="tech-pill">TF-IDF Vectorization</span>
            <span class="tech-pill">Cosine Similarity</span>
            <span class="tech-pill">TMDb API</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Navigation Tabs: Recommendation Engine vs About Project
tab_recommend, tab_about = st.tabs(["🍿 Get Recommendations", "📖 About Project"])

with tab_recommend:
    all_movie_titles = sorted(dataset['title'].unique().tolist())
    default_selection_index = all_movie_titles.index("Avatar") if "Avatar" in all_movie_titles else 0

    col_select, col_count, col_btn = st.columns([4, 1.5, 1.5], gap="medium")
    
    with col_select:
        selected_title = st.selectbox(
            "Select or search for a movie title:",
            options=all_movie_titles,
            index=default_selection_index,
            help="Type to search among 4,300+ movies in the dataset."
        )

    with col_count:
        num_recommendations = st.selectbox(
            "Results count:",
            options=[5, 6, 8, 10],
            index=0,
            help="Select how many recommendations to display."
        )

    with col_btn:
        st.write("")
        st.write("")
        recommend_clicked = st.button("✨ Recommend", width="stretch")

    # Selected movie summary snippet
    selected_row = dataset[dataset['title'] == selected_title]
    if not selected_row.empty:
        curr_movie = selected_row.iloc[0]
        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.4); border-left: 4px solid #38bdf8; border-radius: 8px; padding: 12px 18px; margin: 16px 0 24px 0;">
                <span style="color: #94a3b8; font-size: 0.85rem;">SELECTED FILM</span><br/>
                <strong style="font-size: 1.15rem; color: #f8fafc;">{curr_movie['title']}</strong>
                <span style="color: #64748b; font-size: 0.9rem;">({curr_movie['release_date'] if curr_movie['release_date'] else 'N/A'})</span>
                {" • ⭐ " + str(curr_movie['vote_average']) if curr_movie['vote_average'] > 0 else ""}
                <p style="color: #cbd5e1; font-size: 0.85rem; margin-top: 6px; margin-bottom: 0;">{curr_movie['overview'][:220] + '...' if len(curr_movie['overview']) > 220 else curr_movie['overview']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Trigger recommendation on button click or default load
    if recommend_clicked or selected_title:
        with st.spinner(f"Computing content similarity for '{selected_title}'..."):
            recs = recommend_similar_movies(selected_title, top_n=num_recommendations, api_key=TMDB_API_KEY)

        if recs.empty:
            st.warning(f"No recommendations could be computed for '{selected_title}'. Please try selecting another title.")
        else:
            st.subheader(f"Top {len(recs)} Recommendations for: {selected_title}")

            # Responsive grid display (up to 5 columns per row)
            cols_per_row = 5 if len(recs) >= 5 else len(recs)
            for row_start in range(0, len(recs), cols_per_row):
                row_slice = recs.iloc[row_start:row_start + cols_per_row]
                card_cols = st.columns(cols_per_row)

                for col_idx, (_, movie) in enumerate(row_slice.iterrows()):
                    with card_cols[col_idx]:
                        # Genre tags formatting
                        genre_list = [g.strip() for g in movie['genres'].split() if g.strip()][:3]
                        genre_html = "".join([f'<span class="genre-badge">{g}</span>' for g in genre_list])

                        # Card rendering
                        st.image(movie['poster_url'], width="stretch")
                        st.markdown(
                            f"""
                            <div class="movie-card">
                                <div class="card-content">
                                    <div class="meta-row">
                                        <span class="score-badge">{movie['similarity_score']}% Match</span>
                                        <span>{movie['release_date'] if movie['release_date'] else ''}</span>
                                    </div>
                                    <div class="movie-title" title="{movie['title']}">{movie['title']}</div>
                                    <div class="genre-tags">{genre_html}</div>
                                    <div class="overview-text">{movie['overview'] if movie['overview'] else 'No synopsis available.'}</div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

with tab_about:
    st.markdown("### 📽️ Movie Recommendation System Using NLP and Python")
    st.markdown(
        """
        #### Project Description
        This project implements a content-based movie recommendation system using Natural Language Processing.
        It processes movie information, transforms textual features using TF-IDF vectorization, and calculates
        cosine similarity to identify and recommend movies with similar content.

        The system provides an interactive web interface through which users can select movies and explore recommendations.
        """
    )

    col_tech, col_team = st.columns(2, gap="large")

    with col_tech:
        st.markdown("#### 🛠️ Technologies Used")
        st.markdown(
            """
            - **Python** (Core runtime)
            - **Streamlit** (Interactive web framework)
            - **Pandas & NumPy** (Data wrangling & numerical matrix operations)
            - **Scikit-learn** (TF-IDF vectorizer & Cosine similarity computation)
            - **Natural Language Processing (NLP)** (Tokenization & stopwords removal via NLTK)
            - **TMDB API** (Dynamic movie poster fetching)
            """
        )

    with col_team:
        st.markdown("#### 👨‍💻 Developed By")
        st.markdown(
            """
            - **Rugved Kulkarni**
            - **Siddhant Gaikwad**
            """
        )

    st.markdown("---")
    st.markdown("#### ⚙️ How the Recommendation Algorithm Works")
    st.markdown(
        """
        1. **Feature Extraction:** For every movie in the dataset, the `genres`, `keywords`, and `overview` text fields are merged into a unified metadata string.
        2. **NLP Preprocessing:** Text is stripped of punctuation, lowercased, and tokenized using NLTK's `TreebankWordTokenizer`. Standard English stopwords are removed.
        3. **TF-IDF Vectorization:** The corpus is converted into a high-dimensional feature matrix with $5,000$ top n-grams using term frequency-inverse document frequency weighting.
        4. **Sparse Cosine Similarity:** When a user queries a film, the vector of that film is compared against the full sparse corpus using cosine distance:
           $$\\text{Cosine Similarity}(u, v) = \\frac{u \\cdot v}{\\|u\\| \\|v\\|}$$
        5. **Ranked Output:** The films with the highest similarity scores (excluding the query movie itself) are ranked and displayed alongside their metadata.
        """
    )


# ==========================================
# Footer
# ==========================================
st.markdown(
    """
    <div class="app-footer">
        Movie Recommendation System Using NLP and Python<br/>
        Developed with ❤️ by <span class="footer-devs">Rugved Kulkarni</span> & <span class="footer-devs">Siddhant Gaikwad</span>
    </div>
    """,
    unsafe_allow_html=True
)
