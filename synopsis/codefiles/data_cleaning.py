import os
import sys
import pandas as pd

# Setup paths and logger
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app.classes.main_helper import MainHelper
LOG_NAME = "data_cleaning"
helper = MainHelper(REPO_ROOT, LOG_NAME)
logger = helper.logger

DATA_PATH = os.path.join(helper.get_path("data"), "books_augment_google.csv")
OUTPUT_PATH = os.path.join(helper.get_path("data"), "books_cleaned.csv")
ALT_OUTPUT_PATH = os.path.join(helper.get_path("data"), "books_cleaned_alt.csv")

WORD_DIFF_THRESHOLD = 10

# Load dataset
df = pd.read_csv(DATA_PATH)
logger.info("Loaded dataset with %d entries for cleaning.", df.shape[0])

# Ensure *_google columns are treated as string
google_cols = ['title_google', 'authors_google', 'description_google', 'categories_google', 'num_pages_google']
for col in google_cols:
    if col in df.columns:
        df[col] = df[col].fillna('').astype(str)

# Set 'nan' to empty string for *_google columns
for col in google_cols:
    if col in df.columns:
        df[col] = df[col].replace('nan', '')

# Set the 'description' column to empty string if it only contains 'nan'
df['description'] = df['description'].replace('nan', '')

# Create 'alt_title', 'alt_authors', and 'alt_description' columns if mismatches or small word difference
def build_alternatives(row):
    original_title = str(row['title']).strip()
    google_title = str(row['title_google']).strip()
    alt_title = google_title if original_title.lower() != google_title.lower() and google_title else ''

    original_authors = str(row['authors']).strip()
    google_authors = str(row['authors_google']).strip()
    alt_authors = google_authors if original_authors.lower() != google_authors.lower() and google_authors else ''

    original_desc = str(row['description']).strip()
    google_desc = str(row['description_google']).strip()
    word_diff = len(google_desc.split()) - len(original_desc.split())
    alt_description = google_desc if 0 < word_diff <= WORD_DIFF_THRESHOLD else ''

    return alt_title, alt_authors, alt_description

df[['alt_title', 'alt_authors', 'alt_description']] = df.apply(
    lambda row: pd.Series(build_alternatives(row)), axis=1
)

logger.info(
    "Created 'alt_title', 'alt_authors', and 'alt_description' columns. "
    "Entries with alt_title: %d, alt_authors: %d, alt_description: %d.",
    df['alt_title'].astype(bool).sum(),
    df['alt_authors'].astype(bool).sum(),
    df['alt_description'].astype(bool).sum()
)

# Replace description if Google version is 'WORD_DIFF_THRESHOLD' words longer
def select_best_description(row):
    original = str(row['description']).strip()
    google = str(row['description_google']).strip()
    if google and len(google.split()) - len(original.split()) > WORD_DIFF_THRESHOLD:
        return google
    return original

df['description'] = df.apply(select_best_description, axis=1)
logger.info("Updated 'description' field where Google version is significantly longer.")

# Fill missing num_pages with num_pages_google
if 'num_pages_google' in df.columns:
    before_fill = df['num_pages'].isna().sum()
    df['num_pages'] = df['num_pages'].fillna(df['num_pages_google'])
    df['num_pages'] = pd.to_numeric(df['num_pages'], errors='coerce')
    after_fill = df['num_pages'].isna().sum()
    logger.info("Filled 'num_pages' using Google values. Missing before: %d, after: %d", before_fill, after_fill)

# Merge categories: categories, categories_google, and subjects
def merge_categories(row):
    tags = set()
    for source in ['categories', 'categories_google', 'subjects']:
        if pd.notna(row.get(source)) and isinstance(row[source], str):
            tags.update([x.strip().lower() for x in row[source].split(',') if x.strip()])
    return ', '.join(sorted(tags))

df['final_categories'] = df.apply(merge_categories, axis=1)
logger.info("Merged 'categories', 'categories_google', and 'subjects' into 'final_categories'.")

# Remove rows with 'description' containing less than WORD_DIFF_THRESHOLD - 1 words
initial_count = df.shape[0]
df = df[df['description'].str.split().str.len() >= WORD_DIFF_THRESHOLD - 1]
removed_count = initial_count - df.shape[0]
logger.info("Removed %d rows with 'description' containing less than %d words.", removed_count, WORD_DIFF_THRESHOLD - 1)


# Final selection of columns
columns_to_keep = ['isbn13', 'title', 'subtitle', 'authors', 'average_rating', 'num_pages', 'published_year', 'description', 'final_categories', 'thumbnail']
columns_to_keep_alt = ['isbn13', 'title', 'subtitle', 'authors', 'average_rating', 'num_pages', 'published_year', 'description', 'final_categories', 'thumbnail', 'alt_title', 'alt_authors', 'alt_description']
df_final = df[columns_to_keep]
df_final_alt = df[columns_to_keep_alt]
logger.info("Final dataset contains %d entries.", df.shape[0])
logger.info("Final dataset has missing values in columns:\n%s", df_final.isna().sum())

# Save to output
df_final.to_csv(OUTPUT_PATH, index=False)
df_final_alt.to_csv(ALT_OUTPUT_PATH, index=False)
logger.info("Cleaned dataset saved to: %s", ALT_OUTPUT_PATH)
logger.info("Cleaned dataset saved to: %s", OUTPUT_PATH)
print("Cleaning complete. Output saved.")