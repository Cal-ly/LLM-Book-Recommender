import sys
import os

# Add repository root to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import numpy as np
from time import sleep
from app.classes.main_helper import MainHelper

# Initialize helper
LOG_NAME = "data_augment_openlib"
helper = MainHelper(REPO_ROOT, LOG_NAME)
logger = helper.logger

# Paths
DATA_PATH = os.path.join(helper.get_path("data"), "books_explored.csv")
DATA_OUTPUT_PATH = os.path.join(helper.get_path("data"), "books_augment_openlib.csv")
FIG_DIR = helper.get_path("figures")

# Load dataset
df = pd.read_csv(DATA_PATH)
logger.info("Loaded explored dataset with %d entries.", df.shape[0])

# Explicitly cast columns to avoid dtype issues
df['subjects'] = df['subjects'].astype(str)
df['published_year'] = df['published_year'].astype('Int64').astype(str)
df['subtitle'] = df['subtitle'].astype(str)
df['authors'] = df['authors'].astype(str)
df['thumbnail'] = df['thumbnail'].astype(str)
df['categories'] = df['categories'].astype(str)

# Function to get data from OpenLibrary
def get_openlibrary_data(isbn):
    url = f"https://openlibrary.org/api/books?bibkeys=ISBN:{isbn}&format=json&jscmd=data"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.json().get(f"ISBN:{isbn}", {})
    except requests.RequestException as e:
        logger.error(f"Request failed for ISBN {isbn}: {e}")
        return {}

# Function to augment book data safely
def augment_book_from_openlib(row):
    isbn = row['isbn13']
    book_data = get_openlibrary_data(isbn)
    
    updated = False

    # Subtitle
    if not row['has_subtitle']:
        subtitle = book_data.get('subtitle', '')
        if subtitle:
            row['subtitle'] = subtitle
            row['has_subtitle'] = 1  # Update has_subtitle to 1 (True)
            updated = True

    # Authors
    if not row['has_authors']:
        authors = ', '.join([a['name'] for a in book_data.get('authors', [])])
        if authors:
            row['authors'] = authors
            row['has_authors'] = 1  # Update has_authors to 1 (True)
            updated = True

    # Thumbnail (medium cover)
    if not row['has_thumbnail']:
        cover = book_data.get('cover', {}).get('medium', '')
        if cover:
            row['thumbnail'] = cover
            row['has_thumbnail'] = 1  # Update has_thumbnail to 1 (True)
            updated = True

    # Published year
    if not row['has_published_year']:
        publish_date = book_data.get('publish_date', '')
        if publish_date:
            year = ''.join(filter(str.isdigit, publish_date))[:4]
            if year:
                row['published_year'] = year
                row['has_published_year'] = 1  # Update has_published_year to 1 (True)
                updated = True

    # Number of pages
    if not row['has_num_pages']:
        num_pages = book_data.get('number_of_pages', '')
        if num_pages:
            row['num_pages'] = num_pages
            row['has_num_pages'] = 1  # Update has_num_pages to 1 (True)
            updated = True

    # Subjects
    if not row['has_categories']:  # Assuming 'has_categories' corresponds to 'subjects'
        subjects = ', '.join([sub['name'] for sub in book_data.get('subjects', [])])
        if subjects:
            row['subjects'] = subjects
            row['has_categories'] = 1  # Update has_categories to 1 (True)
            updated = True

    return row, updated

# Augmentation loop
total_books, updates, requests_made, skipped = 0, 0, 0, 0

for idx, row in df.iterrows():
    total_books += 1
    if total_books % 500 == 0:
        print(f"Processed {total_books} books...")

    # Check if all flags are set to 1 (True)
    flags_true = all([
        row['has_subtitle'] == 1, 
        row['has_authors'] == 1, 
        row['has_thumbnail'] == 1,
        row['has_published_year'] == 1, 
        row['has_num_pages'] == 1, 
        row['has_categories'] == 1
    ])

    # Use the existing 'words_in_description' column
    description_word_count = row['words_in_description']

    # Skip if all flags are true and description is sufficiently long
    if flags_true and description_word_count >= 30:
        skipped += 1
        continue

    # Perform augmentation
    df.loc[idx], updated = augment_book_from_openlib(row)
    requests_made += 1

    if updated:
        updates += 1
        logger.info(f"ISBN {row['isbn13']} updated from OpenLibrary.")
        sleep(1)  # Sleep to avoid hitting the API too hard

# Logging final stats
logger.info(f"Total updates: {updates}, Requests made: {requests_made}, Books skipped: {skipped}")

# Create a new dataframe and remove the 'has_*' columns for analysis of the augmented dataset
analysis_df = df.copy()
analysis_df.drop(columns=[
    'has_subtitle', 'has_authors', 'has_thumbnail',
    'has_published_year', 'has_num_pages', 'has_categories', 'words_in_description', 'subjects'
], inplace=True)

# Check the updated dataset for missing values
missing_values = analysis_df.isnull().sum()
missing_values = missing_values[missing_values > 0]
if not missing_values.empty:
    logger.info("Missing values found in the following columns:\n%s", missing_values)
else:
    logger.info("No missing values found in the dataset.")

# Seaborn heatmap for missing values
plt.figure(figsize=(10, 8))
ax = plt.axes()
sns.heatmap(analysis_df.isna().transpose(), cbar=False, cmap="viridis", ax=ax)
plt.title("Missing Values Heatmap")
plt.xlabel("Columns")
plt.ylabel("Missing Values")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
heatmap_path = os.path.join(FIG_DIR, "openlib_values_heatmap.png")
plt.savefig(heatmap_path)
plt.close()

# Save the augmented dataset
df.to_csv(DATA_OUTPUT_PATH, index=False)
logger.info(f"Augmented dataset saved as {DATA_OUTPUT_PATH}")