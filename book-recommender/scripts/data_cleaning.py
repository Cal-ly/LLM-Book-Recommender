import pandas as pd
import os
import logging
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import datetime

# Setup paths
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "books_explored.csv"))
DATA_OUTPUT_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "books_cleaned.csv"))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))
EMBEDDING_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "embeddings"))
LOGS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "logs"))
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOGS_DIR, f"data_cleaning-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log")
MODEL_NAME = "all-MiniLM-L6-v2"

# Ensure output dirs exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EMBEDDING_DIR, exist_ok=True)

# Logging
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load explored dataset
if not os.path.exists(DATA_PATH):
    logger.error("Explored dataset not found at path: %s", DATA_PATH)
    raise FileNotFoundError("books_explored.csv not found.")

df = pd.read_csv(DATA_PATH)
logger.info("Loaded explored dataset with %d entries.", df.shape[0])



# ---

# Filter clean subset
clean_df = df.dropna(subset=["description", "num_pages", "average_rating", "published_year"]).copy()
clean_df["words_in_description"] = clean_df["description"].str.split().str.len()
logger.info("Filtered dataset to %d entries with complete and usable descriptions.", clean_df.shape[0])

# Export cleaned dataset
cleaned_csv_path = DATA_OUTPUT_PATH
if os.path.exists(cleaned_csv_path):
    logger.warning("Cleaned dataset already exists. Overwriting: %s", cleaned_csv_path)
else: 
    logger.info("Exporting cleaned dataset to: %s", cleaned_csv_path)
clean_df.to_csv(cleaned_csv_path, index=False)
logger.info("Cleaned dataset exported to: %s", cleaned_csv_path)