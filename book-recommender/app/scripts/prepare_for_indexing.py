import os
import sys
import pandas as pd

# Setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app.classes.main_helper import MainHelper
LOG_NAME = "prepare_for_indexing"
helper = MainHelper(REPO_ROOT, LOG_NAME)
logger = helper.logger

INPUT_PATH = os.path.join(helper.get_path("data"), "books_cleaned_final.csv")
INDEXED_PATH = os.path.join(helper.get_path("data"), "books_indexed.csv")
UI_PATH = os.path.join(helper.get_path("data"), "books_ui.csv")

# Load data
df = pd.read_csv(INPUT_PATH)
logger.info("Loaded %d rows for indexing preparation.", len(df))

# Keep only high-confidence categories
def filter_categories(pred_scores, threshold=0.2):
    scored = eval(pred_scores)
    filtered = [label for label, score in scored.items() if score >= threshold]
    if not filtered and scored:
        filtered = [max(scored, key=scored.get)]
    return filtered

df["refined_categories"] = df["predicted_scores"].apply(lambda x: filter_categories(x))

# Create search_text to be embedded
df["search_text"] = df.apply(
    lambda row: f"Title: {row['full_title']}. Author: {row['authors']}. Description: {row['description']}", axis=1
)

# Clean and prepare for indexing
df_indexed = df[[
    "isbn13", "full_title", "authors", "published_year", "description", "refined_categories", "search_text"
]].copy()
df_ui = df[[
    "isbn13", "full_title", "authors", "published_year", "description", "refined_categories",
    "average_rating", "num_pages", "thumbnail"
]].copy()

# Save outputs
df_indexed.to_csv(INDEXED_PATH, index=False)
df_ui.to_csv(UI_PATH, index=False)
logger.info("Saved indexed dataset to: %s", INDEXED_PATH)
logger.info("Saved UI dataset to: %s", UI_PATH)
print("Indexing preparation complete.")
