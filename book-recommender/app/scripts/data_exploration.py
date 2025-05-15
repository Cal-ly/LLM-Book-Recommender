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
import datetime
import numpy as np
from app.classes.main_helper import MainHelper

# Initialize helper
LOG_NAME = "data_exploration"
helper = MainHelper(REPO_ROOT, LOG_NAME)
logger = helper.logger

# Paths
DATA_DIR = helper.get_path("data")
DATA_IN = os.path.join(DATA_DIR, "books.csv")
DATA_OUT = os.path.join(DATA_DIR, "books_explored.csv")
DATA_TO_REVIEW = os.path.join(DATA_DIR, "books_to_review.csv")
FIG_DIR = helper.get_path("figures")

# Ensure dataset exists
try:
    helper.ensure_file_exists(DATA_IN)
except FileNotFoundError as e:
    logger.error(str(e))
    print(str(e))
    sys.exit(1)

# Load dataset
df = pd.read_csv(DATA_IN)
logger.info("Dataset loaded with %d rows and %d columns", df.shape[0], df.shape[1])
logger.info("Columns: %s", df.columns.tolist())

# Check for missing values
missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0]
if not missing_values.empty:
    logger.info("Missing values found in the following columns:\n%s", missing_values)
else:
    logger.info("No missing values found in the dataset.")

# Check for duplicates
duplicates = df.duplicated().sum()
if duplicates > 0:
    logger.warning("Found %d duplicate rows in the dataset.", duplicates)
else:
    logger.info("No duplicate rows found in the dataset.")

# Check for empty strings
empty_strings = (df == "").sum()
empty_strings = empty_strings[empty_strings > 0]
if not empty_strings.empty:
    logger.info("Empty strings found in the following columns:\n%s", empty_strings)
else:
    logger.info("No empty strings found in the dataset.")

# Check for non-numeric values in numeric columns
numeric_columns = ["average_rating", "num_pages", "published_year"]
non_numeric_values = {}
for col in numeric_columns:
    non_numeric = df[~df[col].apply(lambda x: isinstance(x, (int, float)))]
    if not non_numeric.empty:
        non_numeric_values[col] = non_numeric
        review_path = os.path.join(DATA_DIR, f"non_numeric_{col}.csv")
        non_numeric.to_csv(review_path, index=False)
        logger.warning("Non-numeric values found in column '%s': %d entries", col, len(non_numeric))
    else:
        logger.info("No non-numeric values found in column '%s'.", col)

# Check for invalid ISBN10 numbers
invalid_isbn10 = df[~df['isbn10'].astype(str).str.match(r'^\d{10}$', na=False)]
if not invalid_isbn10.empty:
    invalid_isbn10_path = os.path.join(DATA_DIR, "invalid_isbn10.csv")
    invalid_isbn10.to_csv(invalid_isbn10_path, index=False)
    logger.warning("Invalid ISBN10 numbers: %d entries", len(invalid_isbn10))
else:
    logger.info("No invalid ISBN10 numbers found.")

# Check for invalid ISBN13 numbers
invalid_isbn13 = df[~df['isbn13'].astype(str).str.match(r'^\d{13}$', na=False)]
if not invalid_isbn13.empty:
    invalid_isbn13_path = os.path.join(DATA_DIR, "invalid_isbn13.csv")
    invalid_isbn13.to_csv(invalid_isbn13_path, index=False)
    logger.warning("Invalid ISBN13 numbers: %d entries", len(invalid_isbn13))
else:
    logger.info("No invalid ISBN13 numbers found.")

# Check for duplicate ISBN numbers
duplicate_isbn = df[df.duplicated(subset=['isbn10', 'isbn13'], keep=False)]
if not duplicate_isbn.empty:
    duplicate_isbn_path = os.path.join(DATA_DIR, "duplicate_isbn.csv")
    duplicate_isbn.to_csv(duplicate_isbn_path, index=False)
    logger.warning("Duplicate ISBN numbers: %d entries", len(duplicate_isbn))
else:
    logger.info("No duplicate ISBN numbers found.")

# All the books with duplicate titles
duplicate_titles = df[df.duplicated(subset=['title'], keep=False)]
if not duplicate_titles.empty:
    duplicate_titles_path = os.path.join(DATA_DIR, "duplicate_titles.csv")
    duplicate_titles.to_csv(duplicate_titles_path, index=False, mode='a', header=not os.path.exists(duplicate_titles_path))
    logger.info("Duplicate titles: %d entries", len(duplicate_titles))
else:
    logger.info("No duplicate titles found.")

# All the books with missing authors
missing_authors = df[df['authors'].isna()]
if not missing_authors.empty:
    missing_authors_path = os.path.join(DATA_DIR, "missing_authors.csv")
    missing_authors.to_csv(missing_authors_path, index=False, mode='a', header=not os.path.exists(missing_authors_path))
    logger.info("Missing authors: %d entries", missing_authors.shape[0])
else:
    logger.info("No missing authors found in the dataset.")

# All the books with missing descriptions
missing_descriptions = df[df['description'].isna()]
if not missing_descriptions.empty:
    missing_descriptions_path = os.path.join(DATA_DIR, "missing_descriptions.csv")
    missing_descriptions.to_csv(missing_descriptions_path, index=False, mode='a', header=not os.path.exists(missing_descriptions_path))
    logger.info("Missing descriptions: %d entries", missing_descriptions.shape[0])
else:
    logger.info("No missing descriptions found in the dataset.")

# All the books with duplicate descriptions
duplicate_descriptions = df[df.duplicated(subset=['description'], keep=False)]
if not duplicate_descriptions.empty:
    duplicate_descriptions_path = os.path.join(DATA_DIR, "duplicate_descriptions.csv")
    duplicate_descriptions.to_csv(duplicate_descriptions_path, index=False, mode='a', header=not os.path.exists(duplicate_descriptions_path))
    logger.info("Duplicate descriptions: %d entries", len(duplicate_descriptions))
else:
    logger.info("No duplicate descriptions found.")

# All the unique categories
unique_categories = df['categories'].unique()
if len(unique_categories) > 0:
    unique_categories_path = os.path.join(DATA_DIR, "unique_categories.csv")
    pd.DataFrame(unique_categories, columns=['categories']).to_csv(unique_categories_path, index=False, mode='a', header=not os.path.exists(unique_categories_path))
    logger.info("Unique categories: %d entries", len(unique_categories))
else:
    logger.info("No unique categories found in the dataset.")

# Log the number of unique values in title, authors, categories, and description columns
columns_to_check = ["title", "authors", "categories", "description"]
for col in columns_to_check:
    if col in df.columns:
        total_entries = df[col].count()
        unique_values = df[col].nunique()
        logger.info("Column '%s' has %d total entries and %d unique values.", col, total_entries, unique_values)
    else:
        logger.warning("Column '%s' is not present in the dataset.", col)

# Seaborn heatmap for missing values
plt.figure(figsize=(10, 8))
ax = plt.axes()
sns.heatmap(df.isna().transpose(), cbar=False, cmap="viridis", ax=ax)
plt.title("Missing Values Heatmap")
plt.xlabel("Columns")
plt.ylabel("Missing Values")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
heatmap_path = os.path.join(FIG_DIR, "missing_values_heatmap.png")
plt.savefig(heatmap_path)
plt.close()

# Rating distribution (use average_rating)
plt.figure(figsize=(10, 8))
sns.histplot(df['average_rating'], bins=20, kde=True)
plt.title('Average Rating Distribution')
plt.xlabel('Average Rating')
plt.ylabel('Frequency')
plt.xlim(df['average_rating'].min(), df['average_rating'].max())
plt.xticks(rotation=45)
plt.tight_layout()
rating_dist_path = os.path.join(FIG_DIR, "rating_distribution.png")
plt.savefig(rating_dist_path)
plt.close()

# Publication year distribution
plt.figure(figsize=(10, 8))
sns.histplot(df['published_year'], bins=40, kde=False)
plt.title('Publication Year Distribution')
plt.xlabel('Year')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.xlim(df['published_year'].min(), df['published_year'].max())
plt.savefig(os.path.join(FIG_DIR, "publication_year_distribution.png"))
plt.close()

# Add new columns for analysis
df["missing_description"] = np.where(df["description"].isna(), 1, 0)
current_year = datetime.date.today().year
df["age_of_book"] = current_year - df["published_year"].fillna(current_year).astype(int)

# Save correlation heatmap
columns_of_interest = ["num_pages", "age_of_book", "missing_description", "average_rating"]
correlation_matrix = df[columns_of_interest].corr(method="spearman")
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={"label": "Spearman correlation"})
plt.title("Correlation Heatmap")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "correlation_heatmap.png"))
plt.close()

# Top categories
category_counts = df['categories'].value_counts().head(25)
plt.figure(figsize=(10, 8))
sns.barplot(x=category_counts.values, y=category_counts.index)
plt.title("Top Categories of Books")
plt.xlabel("Number of Books")
plt.ylabel("Category")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "top_categories.png"))
plt.close()

# Books with <50 words in description
plt.figure(figsize=(10, 8))
df['words_in_description'] = df['description'].str.split().str.len().fillna(0).astype(int)
less_than_50_words_description = df[df['words_in_description'] <= 50]
# Save the books with less than 50 words in description and sort by number of words in description
less_than_50_words_description = less_than_50_words_description.sort_values(by='words_in_description', ascending=True)
less_than_50_words_description_path = os.path.join(DATA_DIR, "less_than_50_words_description.csv")
less_than_50_words_description.to_csv(less_than_50_words_description_path, index=False, mode='a', header=not os.path.exists(less_than_50_words_description_path))
binned_counts = less_than_50_words_description['words_in_description'].value_counts().sort_index()
binned_counts = binned_counts.groupby(pd.cut(binned_counts.index, bins=[0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 51], right=False), observed=False).sum()
logger.info("Number of books with 50 or less words in description:\n%s", binned_counts)
plt.figure(figsize=(10, 8))
sns.barplot(x=binned_counts.index.astype(str), y=binned_counts.values)
plt.title("Number of Books by Words in Description")
plt.xlabel("Number of Words in Description")
plt.ylabel("Number of Books")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "less_than_50_words_description.png"))
plt.close()

# Drop the 'isbn10', 'missing_description', 'ratings_count' and 'age_of_book' columns
df.drop(columns=['isbn10', 'missing_description', 'ratings_count', 'age_of_book'], inplace=True)

# Add a subject column for later analysis
if 'subjects' not in df.columns:
    df['subjects'] = ""
df['subjects'] = df['subjects'].astype(str)

# Create boolean flags for missing values
df['has_subtitle'] = (df['subtitle'].notna() & df['subtitle'].str.strip().astype(bool)).astype(int)
df['has_authors'] = (df['authors'].notna() & df['authors'].str.strip().astype(bool)).astype(int)
df['has_categories'] = (df['categories'].notna() & df['categories'].str.strip().astype(bool)).astype(int)
df['has_thumbnail'] = (df['thumbnail'].notna() & df['thumbnail'].str.strip().astype(bool)).astype(int)
df['has_description'] = (df['description'].notna() & df['description'].str.strip().astype(bool)).astype(int)
df['has_published_year'] = (df['published_year'].notna() & df['published_year'].astype(str).str.strip().astype(bool)).astype(int)
df['has_average_rating'] = (df['average_rating'].notna() & df['average_rating'].astype(str).str.strip().astype(bool)).astype(int)
df['has_num_pages'] = (df['num_pages'].notna() & df['num_pages'].astype(str).str.strip().astype(bool)).astype(int)

# Format and export
explored_df = df.copy()
explored_df.to_csv(DATA_OUT, index=False)
logger.info("\n--- Data exploration complete ---")
print("Data exploration complete.")
