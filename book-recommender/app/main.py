import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

# Paths
BASE_DIR = os.path.dirname(__file__)
EMBEDDING_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "embeddings"))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))
MODEL_NAME = "all-MiniLM-L6-v2"

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
st.set_page_config(page_title="Local Book Recommender", layout="wide")
st.title("📚 Local Book Recommender")
st.markdown("Search for books using a short description, idea, or theme. All processing is local.")

query = st.text_input("Enter your book query:", placeholder="e.g., A dystopian society controlled by AI")

if query:
    with st.spinner("Finding recommendations..."):
        model = load_model()
        index = load_faiss_index()
        books = load_books()

        # Embed query
        query_embedding = model.encode([query], convert_to_numpy=True)
        D, I = index.search(query_embedding, k=5)

        # Show results
        st.subheader("🔍 Recommendations")
        for idx in I[0]:
            book = books.iloc[idx]
            st.markdown(f"**{book['title']}** by *{book['authors']}*")
            st.markdown(f"⭐ Rating: {book['average_rating']}")
            st.markdown(f"> {book['description'][:500]}...")
            st.markdown("---")