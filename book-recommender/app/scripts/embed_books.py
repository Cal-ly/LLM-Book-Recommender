import os
import sys
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from datetime import datetime

# Setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app.classes.main_helper import MainHelper
LOG_NAME = "embed_books"
helper = MainHelper(REPO_ROOT, LOG_NAME)
logger = helper.logger

DATA_PATH = os.path.join(helper.get_path("data"), "books_indexed.csv")
MODEL_DIR = helper.get_path("models")
EMBEDDING_DIR = helper.get_path("embeddings")
MODEL_NAME = "all-MiniLM-L6-v2"

# Load data
df = pd.read_csv(DATA_PATH)
logger.info("Loaded indexed dataset with %d entries.", len(df))

# Load embedding model
logger.info("Loading embedding model: %s", MODEL_NAME)
model = SentenceTransformer(MODEL_NAME)

# Generate embeddings
texts = df["search_text"].astype(str).tolist()
logger.info("Generating embeddings for %d entries...", len(texts))
embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

# Save model metadata
model_meta_path = os.path.join(MODEL_DIR, f"{MODEL_NAME.replace('/', '_')}.txt")
with open(model_meta_path, "w") as f:
    f.write(f"Model used: {MODEL_NAME}\n")
logger.info("Model info saved to %s", model_meta_path)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
index_path = os.path.join(EMBEDDING_DIR, "index.faiss")
faiss.write_index(index, index_path)
logger.info("FAISS index written to %s", index_path)

# Save reduced dataset with metadata for UI
OUTPUT_META = os.path.join(EMBEDDING_DIR, "books_indexed_with_embeddings.csv")
df.to_csv(OUTPUT_META, index=False)
logger.info("Metadata with search_text saved to %s", OUTPUT_META)

print("Embedding and indexing complete.")