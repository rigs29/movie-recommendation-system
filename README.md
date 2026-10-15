# 🎬 Movie Recommendation System Using NLP and Python

A content-based movie recommender web application built with **Python**, **Natural Language Processing (NLP)**, **TF-IDF vectorization**, **Cosine Similarity**, and **Streamlit**.

---

## 👥 Development Team
- **Rugved Kulkarni**
- **Siddhant Gaikwad**

---

## 📖 Project Overview
This project implements a content-based movie recommendation system using Natural Language Processing. It processes movie information, transforms textual features using TF-IDF vectorization, and calculates cosine similarity to identify and recommend movies with similar content.

The system provides an interactive web interface through which users can select movies and explore recommendations.

---

## ✨ Features
- **Smart Natural Language Processing:** Combines genres, keywords, and plot overviews to understand thematic nuances.
- **On-the-Fly Cosine Similarity:** Computes similarity in real-time (~10ms) against a sparse TF-IDF matrix, keeping memory footprint ultra-low (~2MB) and avoiding heavy matrix pre-allocation.
- **Cinematic Dark Theme:** A modern streaming-style responsive web interface built with Streamlit.
- **Interactive Search & Dropdown:** Filter and choose from over 4,300 movies instantly.
- **Dynamic Poster Fetching:** Fetches high-resolution posters via The Movie Database (TMDb) API on demand for recommended films.
- **Graceful Fallbacks:** If the TMDb API key is not configured or if network timeouts occur, the app seamlessly provides clean visual placeholders without crashing.
- **Deployment Ready:** Configured for one-click deployment on Streamlit Community Cloud.

---

## 🛠️ Technologies Used
- **Python 3.10+** — Core programming language
- **Streamlit** — Web interface and interactive components
- **Pandas & NumPy** — Data manipulation and numerical operations
- **Scikit-learn** — TF-IDF Vectorizer and pairwise Cosine Similarity
- **NLTK (Natural Language Toolkit)** — Text tokenization (`TreebankWordTokenizer`) and English stopword filtering
- **Requests** — HTTP client for dynamic TMDb API poster retrieval
- **Python-dotenv** — Environment variable management

---

## 🧠 How the Recommendation Algorithm Works

```
Raw Movie Data (movies.csv)
       │
       ▼
Extract Features (genres + keywords + overview)
       │
       ▼
NLP Preprocessing (Clean text, Tokenize, Remove Stopwords)
       │
       ▼
TF-IDF Vectorization (Extract 5,000 top n-gram features into a sparse matrix)
       │
       ▼ [User selects a movie]
Compute Cosine Similarity:
   Similarity(u, v) = (u · v) / (||u|| * ||v||)
       │
       ▼
Rank top N movies by similarity score & fetch posters dynamically
```

1. **Feature Merging:** The textual fields `genres`, `keywords`, and `overview` are concatenated into a single string for each movie.
2. **Text Preprocessing:** Characters outside the alphabet are removed, text is lowercased, and tokens are extracted. Stopwords (e.g., *the*, *and*, *is*) are filtered out.
3. **TF-IDF Vectorization:** The corpus is transformed into a high-dimensional sparse TF-IDF matrix ($5,000$ features).
4. **On-demand Similarity:** When a user chooses a movie, its vector is compared against all movie vectors using cosine similarity, computing the angular difference between textual features.
5. **Ranking:** The movies with the highest similarity scores (excluding the selected film itself) are presented with their match percentage, genre tags, and overview.

---

## 📁 Project Structure

```
movie-recommendation-system/
├── app.py                  # Main Streamlit application entry point
├── movies.csv              # TMDb dataset (metadata of 4,800+ movies)
├── requirements.txt        # Python package dependencies
├── run.bat                 # One-click Windows runner script
├── .gitignore              # Git ignore rules for secrets and virtual envs
└── README.md               # Complete project documentation
```

---

## 💻 Local Installation & Setup (Windows)

### Prerequisites
- Python 3.10 or higher installed ([python.org](https://www.python.org))
- Git installed ([git-scm.com](https://git-scm.com))

### Step 1: Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/movie-recommendation-system.git
cd movie-recommendation-system
```

### Step 2: Create a Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### Step 3: Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 🔑 Configuring the TMDb API Key (Optional)

The application functions completely without an API key using the local dataset. To display official movie posters:

### Method A: Permanent Windows Environment Variable (Recommended)
In PowerShell, run:
```powershell
[Environment]::SetEnvironmentVariable("TMDB_API_KEY", "YOUR_ACTUAL_TMDB_API_KEY", "User")
```
*Note: Restart VS Code or open a new terminal after running this command so the environment variable takes effect.*

### Method B: Using a `.env` File
Create a `.env` file in the project root:
```env
TMDB_API_KEY=YOUR_ACTUAL_TMDB_API_KEY
```
*(The `.env` file is excluded from Git to protect your credentials.)*

---

## 🚀 Running the Application

### Option 1: Quick Launch (Windows)
Double-click [`run.bat`](run.bat).

### Option 2: Terminal
```powershell
.\.venv\Scripts\activate
streamlit run app.py
```
Open **http://localhost:8501** in your browser.

---

## 🌐 Deploying to Streamlit Community Cloud (Free)

1. **Push your code to GitHub:**
   - Create a new public repository on GitHub named `movie-recommendation-system`.
   - Push your project code (ensure `.env` and `.venv` are ignored).

2. **Connect to Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with your GitHub account.
   - Click **New app**.
   - Select your repository: `YOUR_USERNAME/movie-recommendation-system`.
   - Branch: `main` (or `master`).
   - Main file path: `app.py`.

3. **Configure Secrets on Streamlit Cloud:**
   - Under **Advanced settings** (or in **App Settings → Secrets**), paste:
     ```toml
     TMDB_API_KEY = "YOUR_ACTUAL_TMDB_API_KEY"
     ```
   - Click **Save**.

4. **Deploy:**
   - Click **Deploy!** Your app will build and go live on a public URL (`https://<your-app-name>.streamlit.app`).

---

## 🛠️ Troubleshooting

| Issue | Cause | Solution |
|---|---|---|
| **Posters showing placeholder images** | TMDb API key missing or invalid | Verify `TMDB_API_KEY` is set in Windows environment variables or Streamlit Secrets. |
| **Environment variable not recognized** | Terminal was opened before setting variable | Completely close and reopen VS Code / PowerShell. |
| **NLTK `punkt_tab` error** | Outdated NLTK corpus cache | The app auto-downloads `punkt_tab` on boot. Ensure internet access is available during first run. |
| **ModuleNotFoundError** | Virtual environment not active | Activate with `.\.venv\Scripts\activate` before running `streamlit run app.py`. |

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).