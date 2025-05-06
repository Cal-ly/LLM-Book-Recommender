import pandas as pd
import os
import logging
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import datetime

# Setup paths
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))
EMBEDDING_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "embeddings"))
LOGS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "logs"))
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOGS_DIR, f"embed_books-{datetime.date.today()}.txt")
MODEL_NAME = "all-MiniLM-L6-v2"

# Ensure output dirs exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EMBEDDING_DIR, exist_ok=True)

# Logging
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load cleaned dataset
if not os.path.exists(DATA_PATH):
    logger.error("Cleaned dataset not found at path: %s", DATA_PATH)
    raise FileNotFoundError("books_cleaned.csv not found.")

df = pd.read_csv(DATA_PATH)
logger.info("Loaded cleaned dataset with %d entries.", df.shape[0])

# Load model
logger.info("Loading model: %s", MODEL_NAME)
model = SentenceTransformer(MODEL_NAME)

# Prepare texts for embedding
texts = df["description"].astype(str).tolist()

# Generate embeddings
logger.info("Generating embeddings for %d descriptions...", len(texts))
embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

# Save model info
model_path = os.path.join(MODEL_DIR, MODEL_NAME.replace("/", "_") + ".txt")
with open(model_path, "w") as f:
    f.write(f"Model used: {MODEL_NAME}\\n")

# Store embeddings in FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, os.path.join(EMBEDDING_DIR, "index.faiss"))
df[["title", "authors", "average_rating", "description"]].to_csv(os.path.join(EMBEDDING_DIR, "books_indexed.csv"), index=False)

logger.info("Embeddings and FAISS index saved successfully.")
print("Embedding complete. Index saved to embeddings/.")