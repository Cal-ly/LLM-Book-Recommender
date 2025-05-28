import os
import sys
import faiss
import pandas as pd
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
import time

# Setup paths
APP_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR = APP_DIR  # Since this file is in 'app', SCRIPT_DIR is the same as APP_DIR
REPO_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app.classes.main_helper import MainHelper

# Paths
LOG_NAME = "streamlit_main"
helper = MainHelper(REPO_ROOT, LOG_NAME)
EMBEDDING_DIR = helper.get_path("embeddings")
DATA_DIR = helper.get_path("data")
MODEL_NAME = "all-MiniLM-L6-v2"
CSV_FILE = os.path.join(DATA_DIR, "books_ui.csv")
INDEX_FILE = os.path.join(EMBEDDING_DIR, "index.faiss")
DEFAULT_COVER = os.path.join("figures", "0-cover-not-found.jpg")

# Load model and data
@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

@st.cache_data
def load_books():
    df = pd.read_csv(CSV_FILE)
    index = faiss.read_index(INDEX_FILE)
    return df, index

# Streamlit UI
st.set_page_config(page_title="Book Recommender", page_icon="📚", layout="wide")
st.title("Semantic Book Recommender")
st.markdown("Search for books using a short description, idea, or theme. All processing is local.")
query = st.text_input("Enter your book query:", placeholder="e.g., A dystopian society controlled by AI")

model = load_model()
df, index = load_books()

# Category filter and sorting
all_tags = sorted({tag for sublist in df['refined_categories'].apply(eval) for tag in sublist})
selected_tags = st.multiselect("Filter by genre/tag:", options=all_tags)
sort_order = st.radio("Sort by rating:", options=["High to low", "Low to high"], horizontal=True)

# Perform search
if query:
    helper.logger.info(f"User query: {query}")  # Log the entered query

    start_embed = time.time()
    query_vector = model.encode([query])
    embed_time = time.time() - start_embed
    helper.logger.info(f"Embedding time: {embed_time:.3f} seconds")  # Log embedding time

    start_search = time.time()
    D, I = index.search(query_vector, 60)  # Search up to 60 results
    results = [df.iloc[idx] for idx in I[0]]
    search_time = time.time() - start_search
    helper.logger.info(f"Search time: {search_time:.3f} seconds")  # Log search time

    # Apply tag filtering
    if selected_tags:
        results = [book for book in results if any(tag in eval(book['refined_categories']) for tag in selected_tags)]

    # Sort by average_rating
    results.sort(key=lambda x: x['average_rating'], reverse=(sort_order == "High to low"))

    # Pagination setup
    items_per_page = 6
    total_pages = (len(results) - 1) // items_per_page + 1
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
    start_idx = (page - 1) * items_per_page
    end_idx = start_idx + items_per_page
    display_results = results[start_idx:end_idx]

    st.markdown("### Results")
    cols = st.columns(2)

    for i, book in enumerate(display_results):
        with cols[i % 2]:
            with st.container():
                # Lazy load external image with fallback to local
                if pd.notna(book["thumbnail"]) and book["thumbnail"].startswith("http"):
                    st.image(book["thumbnail"], width=150)
                else:
                    st.image(DEFAULT_COVER, width=150)

                st.subheader(book["full_title"])
                st.markdown(f"**Author:** {book['authors']}")
                st.markdown(f"**Published:** {book['published_year']}")
                st.markdown(f"**Rating:** {book['average_rating']:.1f} ⭐")
                st.markdown(f"**Pages:** {int(book['num_pages']) if pd.notna(book['num_pages']) else '?'}")
                st.markdown(f"**Tags:** {', '.join(eval(book['refined_categories']))}")
                st.write(book["description"][:250] + ("..." if len(book["description"]) > 250 else ""))
