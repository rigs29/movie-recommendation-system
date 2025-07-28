import requests
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Your TMDb API key here
TMDB_API_KEY = "Your Api Key here"

# Load the dataset
df = pd.read_csv("movies.csv")

# Select required columns and remove rows with missing data
df = df[['id', 'genres', 'keywords', 'overview', 'title']].dropna().reset_index(drop=True)

# Combine text features into a single string
df['combined'] = df['genres'] + ' ' + df['keywords'] + ' ' + df['overview']

# Define stop words
stop_words = set(stopwords.words('english'))

# Text preprocessing function
from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()

def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)       # Remove special characters/numbers
    text = text.lower()                           # Convert to lowercase
    tokens = tokenizer.tokenize(text)                  # Tokenize text
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return " ".join(tokens)

# Apply preprocessing
df['cleaned_text'] = df['combined'].apply(preprocess_text)

# Vectorize using TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get poster path from TMDb API for a single movie id
@st.cache_data(show_spinner=False)
def get_poster_path(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get('poster_path')
    else:
        return None

# Add a new column for poster urls, cache to avoid repeated API calls
@st.cache_data(show_spinner=True)
def add_poster_urls(df):
    poster_urls = []
    for movie_id in df['id']:
        poster_path = get_poster_path(movie_id)
        if poster_path:
            poster_urls.append(f"https://image.tmdb.org/t/p/w500{poster_path}")
        else:
            # Default placeholder image if no poster found
            poster_urls.append("https://via.placeholder.com/150x225?text=No+Image")
    df['poster_url'] = poster_urls
    return df

# Apply poster fetching
df = add_poster_urls(df)

# Recommendation function
def recommend_movies(movie_name, top_n=5):
    idx = df[df['title'].str.lower() == movie_name.lower()].index
    if len(idx) == 0:
        return pd.DataFrame()  # empty dataframe if movie not found
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices][['title', 'poster_url', 'overview']]

# ---------------- Streamlit UI ---------------- #
st.title("🎬 Movie Recommendation System")
st.write("Enter a movie name below to get similar movie recommendations.")

movie_input = st.text_input("Enter a movie name:")

recommendations = pd.DataFrame()

if st.button("Recommend"):
    if movie_input:
        recommendations = recommend_movies(movie_input)
        if recommendations.empty:
            st.warning("Movie not found. Try another one!")

# Show recommendations in columns with poster + overview
if not recommendations.empty:
    st.subheader(f"Top {len(recommendations)} movies similar to: {movie_input}")
    cols = st.columns(min(len(recommendations), 5))

    for idx, row in enumerate(recommendations.itertuples()):
        with cols[idx]:
            st.image(row.poster_url, use_container_width=True)
            st.markdown(f"**🎬 {row.title}**", unsafe_allow_html=True)
            st.caption(row.overview[:180] + "..." if row.overview else "No description available.")

