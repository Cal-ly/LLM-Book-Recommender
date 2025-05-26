import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)  # Ensure correct import order

from app.classes.main_helper import MainHelper
LOG_NAME = "data_reexploration"
helper = MainHelper(REPO_ROOT, LOG_NAME)
logger = helper.logger

# Paths
DATA_PATH = os.path.join(helper.get_path("data"), "books_augment_google.csv")
FIG_DIR = helper.get_path("figures")

# Load dataset
df = pd.read_csv(DATA_PATH)
logger.info("Loaded dataset with %d entries.", df.shape[0])

# Fix: Ensure *_google columns are treated as string
google_columns = ['title_google', 'authors_google', 'description_google', 'categories_google', 'num_pages_google']
for col in google_columns:
    if col in df.columns:
        df[col] = df[col].astype(str)

# Step 1: Fill missing values in primary columns from *_google if empty
def fill_if_missing(primary, fallback):
    return primary if pd.notna(primary) and str(primary).strip() else fallback

for col in ['title', 'authors', 'description', 'num_pages']:
    gcol = f"{col}_google"
    df[col] = df.apply(lambda row: fill_if_missing(row[col], row[gcol]), axis=1)

# Step 2: Deduplicate matching *_google fields
for col in ['title', 'authors', 'description', 'num_pages']:
    gcol = f"{col}_google"
    match = df[col].astype(str).str.strip() == df[gcol].astype(str).str.strip()
    df.loc[match, gcol] = ''

# Step 3: Copy to analysis df
analysis_df = df.copy()

# Set 'nan' to empty string for *_google columns
for col in google_columns:
    if col in analysis_df.columns:
        analysis_df[col] = analysis_df[col].replace('nan', '')
        analysis_df[col] = analysis_df[col].replace('None', '')
        analysis_df[col] = analysis_df[col].replace('NaN', '')
        analysis_df[col] = analysis_df[col].replace('none', '')
        analysis_df[col] = analysis_df[col].replace('NAN', '')

# Drop has_* columns
analysis_df = analysis_df.drop(columns=[col for col in analysis_df.columns if col.startswith("has_")])
logger.info("The datatype of each column:\n%s", analysis_df.dtypes)
logger.info("The first 5 rows of the analysis dataset:\n%s", analysis_df.head(5))

# Recalculate word counts
analysis_df['words_in_description'] = analysis_df['description'].fillna('').apply(lambda x: len(x.split()))
analysis_df['words_in_description_google'] = analysis_df['description_google'].fillna('').apply(lambda x: len(x.split()))

# Save binned distribution for descriptions <= 30 words
binned_counts = analysis_df['words_in_description'].value_counts().sort_index()
binned_counts = binned_counts[binned_counts.index <= 30]
logger.info("Binned distribution of description word counts (<= 30 words):\n%s", binned_counts)

# Compare descriptions (Google longer)
longer_desc_mask = (analysis_df['description_google'].str.strip().ne('')) & (analysis_df['words_in_description_google'] > analysis_df['words_in_description'])
longer_desc_df = analysis_df[longer_desc_mask].copy()
longer_desc_df['word_difference'] = longer_desc_df['words_in_description_google'] - longer_desc_df['words_in_description']
logger.info("Books with longer Google descriptions:\n%s", longer_desc_df[['title', 'authors', 'words_in_description', 'words_in_description_google', 'word_difference']])

# Compare mismatched fields with non-empty *_google values
title_mismatch = analysis_df[
    (analysis_df['title_google'].str.strip().ne('')) &
    (analysis_df['title'].str.strip().ne(analysis_df['title_google'].str.strip()))
]
authors_mismatch = analysis_df[
    (analysis_df['authors_google'].str.strip().ne('')) &
    (analysis_df['authors'].str.strip().ne(analysis_df['authors_google'].str.strip()))
]
pages_mismatch = analysis_df[
    (analysis_df['num_pages_google'].str.strip().ne('')) &
    (analysis_df['num_pages'].astype(str).str.strip().ne(analysis_df['num_pages_google'].astype(str).str.strip()))
]
logger.info("Books with title mismatches:\n%s", title_mismatch[['title', 'title_google']])
logger.info("Books with authors mismatches:\n%s", authors_mismatch[['authors', 'authors_google']])
logger.info("Books with page count mismatches:\n%s", pages_mismatch[['num_pages', 'num_pages_google']])

# Plot 1: Histogram of description lengths comparison
plt.figure(figsize=(10, 6))
sns.histplot(analysis_df['words_in_description'], color='blue', label='Original', kde=False, bins=40)
sns.histplot(analysis_df['words_in_description_google'], color='orange', label='Google', kde=False, bins=40)
plt.title("Histogram of Description Word Counts")
plt.xlabel("Word Count")
plt.ylabel("Number of Books")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "reexp_hist_description_word_counts.png"))
plt.close()

# Plot 2: Bar chart of mismatches
mismatch_counts = {
    'Title Mismatch': len(title_mismatch),
    'Authors Mismatch': len(authors_mismatch),
    'Page Count Mismatch': len(pages_mismatch),
    'Longer Google Descriptions': len(longer_desc_df)
}

plt.figure(figsize=(8, 6))
sns.barplot(x=list(mismatch_counts.keys()), y=list(mismatch_counts.values()))
plt.title("Mismatch Counts Between Original and Google Metadata")
plt.ylabel("Number of Books")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "reexp_mismatch_counts.png"))
plt.close()

logger.info("Reexploration complete. Figures saved to: %s", FIG_DIR)
