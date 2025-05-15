import os
from dotenv import load_dotenv
import requests
import pandas as pd
from time import sleep
import sys

# Add repository root to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from app.classes.main_helper import MainHelper

# Initialize helper
LOG_NAME = "data_augment_google"
helper = MainHelper(REPO_ROOT, LOG_NAME)
logger = helper.logger

# Paths
DATA_PATH = os.path.join(helper.get_path("data"), "books_augment_openlib.csv")
DATA_OUTPUT_PATH = os.path.join(helper.get_path("data"), "books_augment_google.csv")
FIG_DIR = helper.get_path("figures")

# Load environment variables
load_dotenv(dotenv_path='C:/Users/Cal-l/Documents/GitHub/LLM-Book-Recommender/book-recommender/app.env')

# Google Books API key
API_KEY = os.getenv("GOOGLE_BOOKS_API")

if not API_KEY:
    raise ValueError("API key not found. Please check your app.env file.")

# Load augmented OpenLibrary dataset
df = pd.read_csv(DATA_PATH)
logger.info("Loaded dataset with %d entries for Google Books augmentation.", df.shape[0])

# Add Google metadata columns if not already present
for col in ['authors_google', 'title_google', 'description_google', 'categories_google', 'num_pages_google']:
    if col not in df.columns:
        df[col] = ""

# Load and prioritize books needing better descriptions
df['words_in_description'] = df['words_in_description'].fillna(0).astype(int)
df_sorted = df.sort_values(by='words_in_description')

# Track API usage
requests_used = 0
REQUEST_LIMIT = 1000

# Google Books API fetch function
def fetch_google_books_metadata(isbn):
    url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}&key={API_KEY}"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Request failed for ISBN {isbn}: {e}")
        return None

# Metadata extraction function
def extract_fields(volume_info):
    return {
        'title_google': volume_info.get('title', ''),
        'authors_google': ', '.join(volume_info.get('authors', [])),
        'description_google': volume_info.get('description', ''),
        'categories_google': ', '.join(volume_info.get('categories', [])),
        'num_pages_google': volume_info.get('pageCount', '')
    }

# Main augmentation loop
for idx, row in df_sorted.iterrows():
    if requests_used >= REQUEST_LIMIT:
        logger.info("Reached daily Google Books API request limit of %d.", REQUEST_LIMIT)
        break

    if row['description_google'] and str(row['description_google']).strip():
        continue  # Skip already processed rows

    isbn = row['isbn13']
    data = fetch_google_books_metadata(isbn)

    if data and data.get('totalItems', 0) > 0:
        volume_info = data['items'][0].get('volumeInfo', {})
        extracted = extract_fields(volume_info)

        for key, value in extracted.items():
            if value:
                df.at[idx, key] = value

        requests_used += 1
        logger.info("[%d/%d] Updated Google metadata for ISBN %s.", requests_used, REQUEST_LIMIT, isbn)
    else:
        logger.warning("No Google Books data found for ISBN %s.", isbn)

    sleep(0.6)  # Stay under 100 requests/min

# Save updated dataset
df.to_csv(DATA_OUTPUT_PATH, index=False)
logger.info("Google metadata augmentation complete. File saved to: %s", DATA_OUTPUT_PATH)
