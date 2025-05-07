import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import datetime
import logging

# NOTE: PyTorch may trigger harmless RuntimeErrors with Streamlit's file watcher.
# Safe to ignore unless app functionality breaks.

# Paths
BASE_DIR = os.path.dirname(__file__)
EMBEDDING_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "embeddings"))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))
AUX_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "auxiliary"))
LOGS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "logs"))
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOGS_DIR, f"main-{datetime.date.today()}.txt")
MODEL_NAME = "all-MiniLM-L6-v2"

# Setup logging
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

# Load FAISS index
@st.cache_resource
def load_faiss_index():
    index_path = os.path.join(EMBEDDING_DIR, "index.faiss")
    return faiss.read_index(index_path)

# Load book metadata
@st.cache_data
def load_books():
    metadata_path = os.path.join(EMBEDDING_DIR, "books_indexed.csv")
    return pd.read_csv(metadata_path)

# App layout
st.set_page_config(page_title="Local Book Recommender", page_icon="📚", layout="wide")
st.title("Local Book Recommender")
st.markdown("Search for books using a short description, idea, or theme. All processing is local.")

# User inputs
query = st.text_input("Enter your book query:", placeholder="e.g., A dystopian society controlled by AI")
min_rating = st.slider("Minimum average rating:", min_value=0.0, max_value=5.0, value=3.0, step=0.1)
genre_filter = st.text_input("Filter by genre/category (optional):", placeholder="e.g., Fiction")

if query:
    with st.spinner("Finding recommendations..."):
        model = load_model()
        index = load_faiss_index()
        books = load_books()

        # Embed query
        query_embedding = model.encode([query], convert_to_numpy=True)
        D, I = index.search(query_embedding, k=50)  # search wider, filter later

        # Show results
        st.subheader("🔍 Recommendations")

        def get_cover_url(isbn):
            if pd.isna(isbn) or str(isbn).lower() == "nan":
                return None
            return f"https://covers.openlibrary.org/b/isbn/{isbn}-M.jpg"

        shown = 0
        cols = st.columns(2)

        for idx in I[0]:
            if idx >= len(books):
                continue

            book = books.iloc[idx]
            if book['average_rating'] < min_rating:
                continue
            if genre_filter and genre_filter.lower() not in str(book.get("categories", "")).lower():
                continue

            with cols[shown % 2]:
                cover_url = get_cover_url(book.get('isbn13'))
                if cover_url:
                    st.image(cover_url, width=120)
                else:
                    fallback_path = os.path.join(AUX_DIR, "cover-not-found.jpg")
                    st.image(fallback_path, width=120)
                st.markdown(f"**{book['title']}** by *{book['authors']}*")
                st.markdown(f"⭐ Rating: {book['average_rating']}")
                st.markdown(f"> {book['description'][:500]}...")
                st.markdown("---")
                shown += 1