import os
import sys
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

# Setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app.classes.main_helper import MainHelper
LOG_NAME = "test_faiss_query"
helper = MainHelper(REPO_ROOT, LOG_NAME)
logger = helper.logger

# Paths
EMBEDDING_DIR = helper.get_path("embeddings")
MODEL_NAME = "all-MiniLM-L6-v2"
DATA_FILE = os.path.join(EMBEDDING_DIR, "books_indexed_with_embeddings.csv")
INDEX_FILE = os.path.join(EMBEDDING_DIR, "index.faiss")

# Load model, data, and index
logger.info("Loading model: %s", MODEL_NAME)
model = SentenceTransformer(MODEL_NAME)
logger.info("Loading dataset from: %s", DATA_FILE)
df = pd.read_csv(DATA_FILE)
logger.info("Loading FAISS index from: %s", INDEX_FILE)
index = faiss.read_index(INDEX_FILE)

# Query
query = "space exploration in the future"
k = 5
logger.info("Encoding query: '%s'", query)
query_embedding = model.encode([query])

# Search
D, I = index.search(query_embedding, k)
logger.info("Top %d results retrieved.", k)

# Print results
print(f"\n🔍 Query: {query}\n")
for i, idx in enumerate(I[0]):
    print(f"{i+1}. {df.loc[idx, 'full_title']} by {df.loc[idx, 'authors']}")
    print(f"   Categories: {df.loc[idx, 'refined_categories']}")
    print(f"   Description: {df.loc[idx, 'description'][:200]}...\n")
