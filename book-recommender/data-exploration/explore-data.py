import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import sys
import numpy as np

# Paths
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "books.csv"))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_FILE = os.path.join(OUTPUT_DIR, "exploration.log")

# Ensure output directory exists
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
logger.info("Missing values per column:\\n%s", df.isnull().sum())

# Add new columns for analysis
df["missing_description"] = np.where(df["description"].isna(), 1, 0)
df["age_of_book"] = 2024 - df["published_year"]

# Save correlation heatmap
columns_of_interest = ["num_pages", "age_of_book", "missing_description", "average_rating"]
correlation_matrix = df[columns_of_interest].corr(method="spearman")
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={"label": "Spearman correlation"})
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
plt.close()
logger.info("Correlation heatmap saved.")

# Filter clean subset
clean_df = df.dropna(subset=["description", "num_pages", "average_rating", "published_year"])
clean_df["words_in_description"] = clean_df["description"].str.split().str.len()
logger.info("Filtered dataset to %d entries with complete and usable descriptions.", clean_df.shape[0])

# Export cleaned dataset
cleaned_csv_path = os.path.join(OUTPUT_DIR, "books_cleaned.csv")
clean_df.to_csv(cleaned_csv_path, index=False)
logger.info("Cleaned dataset exported to: %s", cleaned_csv_path)

# Rating distribution (use average_rating)
plt.figure(figsize=(8, 5))
sns.histplot(clean_df['average_rating'], bins=20, kde=True)
plt.title('Average Rating Distribution')
plt.xlabel('Average Rating')
plt.ylabel('Frequency')
plt.savefig(os.path.join(OUTPUT_DIR, "average_rating_distribution.png"))
plt.close()
logger.info("Average rating distribution plot saved.")

# Publication year distribution
plt.figure(figsize=(10, 6))
sns.histplot(clean_df['published_year'], bins=40, kde=False)
plt.title('Publication Year Distribution')
plt.xlabel('Year')
plt.ylabel('Count')
plt.savefig(os.path.join(OUTPUT_DIR, "publication_year_distribution.png"))
plt.close()
logger.info("Publication year distribution plot saved.")

# Top authors
top_authors = clean_df['authors'].value_counts().nlargest(20)
plt.figure(figsize=(10, 8))
sns.barplot(x=top_authors.values, y=top_authors.index)
plt.title("Top 20 Authors by Number of Books")
plt.xlabel("Number of Books")
plt.ylabel("Author")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "top_authors.png"))
plt.close()
logger.info("Top authors plot saved.")

print("Data exploration complete. Check the output/ folder for results.")
