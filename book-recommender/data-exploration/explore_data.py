# NOTE The script performs data exploration on a book dataset, including loading the dataset, checking for missing values, and generating visualizations.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import sys
import numpy as np
import datetime

# Paths
BASE_DIR = os.path.dirname(__file__)
REPO_BASE_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
SHARED_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "shared"))
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "books.csv"))
DATA_OUTPUT_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "books_explored.csv"))
OUTPUT_DIR = os.path.join(SHARED_DIR, "output")
LOGS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "logs"))
DATE_TIME_STRING = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_FILE = os.path.join(LOGS_DIR, f"explore_data-{DATE_TIME_STRING}.log")

# Ensure directories exists
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Setup logging
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# Check if dataset exists
if not os.path.exists(DATA_PATH):
    logger.error("Dataset not found at expected path: %s", DATA_PATH)
    print(f"Dataset not found at expected path: {DATA_PATH}")
    sys.exit(1)

# Load dataset
df = pd.read_csv(DATA_PATH)
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
        logger.warning("Non-numeric values found in column '%s':\n%s", col, non_numeric)
    else:
        logger.info("No non-numeric values found in column '%s'.", col)

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
plt.savefig(os.path.join(OUTPUT_DIR, "missing_values_heatmap.png"))
plt.close()
logger.debug("Missing values heatmap saved.")

# Rating distribution (use average_rating)
plt.figure(figsize=(10, 8))
sns.histplot(df['average_rating'], bins=20, kde=True)
plt.title('Average Rating Distribution')
plt.xlabel('Average Rating')
plt.ylabel('Frequency')
plt.savefig(os.path.join(OUTPUT_DIR, "average_rating_distribution.png"))
plt.close()
logger.debug("Average rating distribution plot saved.")

# Publication year distribution
plt.figure(figsize=(10, 8))
sns.histplot(df['published_year'], bins=40, kde=False)
plt.title('Publication Year Distribution')
plt.xlabel('Year')
plt.ylabel('Count')
plt.savefig(os.path.join(OUTPUT_DIR, "publication_year_distribution.png"))
plt.close()
logger.debug("Publication year distribution plot saved.")

# Add new columns for analysis
df["missing_description"] = np.where(df["description"].isna(), 1, 0)
current_year = datetime.datetime.now().year
df["age_of_book"] = current_year - df["published_year"]

# Save correlation heatmap
columns_of_interest = ["num_pages", "age_of_book", "missing_description", "average_rating"]
correlation_matrix = df[columns_of_interest].corr(method="spearman")
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={"label": "Spearman correlation"})
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
plt.close()
logger.debug("Correlation heatmap saved.")

# How many different categories of books are in the dataset
category_counts = df['categories'].value_counts()
category_counts = category_counts.head(25)  # Top 25 categories
plt.figure(figsize=(10, 8))
sns.barplot(x=category_counts.values, y=category_counts.index)
plt.title("Top Categories of Books")
plt.xlabel("Number of Books")
plt.ylabel("Category")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "top_categories.png"))
plt.close()
logger.debug("Top categories plot saved.")

# How many books have a description with less than 50 words
plt.figure(figsize=(10, 8))
df['words_in_description'] = df['description'].str.split().str.len()
df['words_in_description'] = df['words_in_description'].fillna(0).astype(int) # Fill NaN with 0 and convert to int
df = df[df['words_in_description'] <= 50] # See only the description with 50 words or less
binned_counts = df['words_in_description'].value_counts().sort_index() # Count the number of books in each bin
binned_counts = binned_counts.groupby(pd.cut(binned_counts.index, bins=[0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 51], right=False)).sum()
logger.info("Number of books with 50 or less words in description:\n%s", binned_counts)
plt.figure(figsize=(10, 8))
sns.barplot(x=binned_counts.index.astype(str), y=binned_counts.values)
plt.title("Number of Books by Words in Description")
plt.xlabel("Number of Words in Description")
plt.ylabel("Number of Books")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "less_than_50_words_description.png"))
plt.close()
logger.debug("Words in description plot saved.")

# Formatting the DataFrame for better readability
explored_df = df.copy()
explored_df["average_rating"] = explored_df["average_rating"].round(2)

# Export explored dataset
explored_csv_path = DATA_OUTPUT_PATH
if os.path.exists(explored_csv_path):
    logger.warning("explored dataset already exists. Overwriting: %s", explored_csv_path)
else: 
    logger.debug("Exporting explored dataset to: %s", explored_csv_path)
explored_df.to_csv(explored_csv_path, index=False)
logger.debug("explored dataset exported to: %s", explored_csv_path)

logger.debug("Data exploration complete.")
print("Data exploration complete. Check the output/ folder for results.")