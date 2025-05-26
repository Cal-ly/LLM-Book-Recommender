import os
import sys
import pandas as pd

# Setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app.classes.main_helper import MainHelper
LOG_NAME = "category_refined_cleaning"
helper = MainHelper(REPO_ROOT, LOG_NAME)
logger = helper.logger

INPUT_PATH = os.path.join(helper.get_path("data"), "books_refined_categories.csv")
OUTPUT_PATH = os.path.join(helper.get_path("data"), "books_cleaned_final.csv")

# Load data
df = pd.read_csv(INPUT_PATH)
logger.info("Loaded %d rows for final cleaning.", len(df))

# Calculate metrics
df["description_length"] = df["augmented_description"].fillna("").astype(str).str.len()
df["average_score"] = df["predicted_scores"].apply(
    lambda x: pd.Series(eval(x)).mean() if pd.notna(x) and x != "{}" else 0
)
df["max_score"] = df["predicted_scores"].apply(
    lambda x: max(eval(x).values()) if pd.notna(x) and x != "{}" else 0
)

# Filtered average (scores above threshold only)
def filtered_average(score_dict, min_threshold=0.2):
    scores = [v for v in eval(score_dict).values() if v >= min_threshold]
    return sum(scores) / len(scores) if scores else 0

df["filtered_avg_score"] = df["predicted_scores"].apply(lambda x: filtered_average(x))

# Standard deviation of prediction scores
def score_std(score_dict):
    scores = list(eval(score_dict).values())
    return pd.Series(scores).std() if scores else 0

df["score_std"] = df["predicted_scores"].apply(score_std)

# Count number of categories predicted
df["num_categories"] = df["predicted_categories"].apply(
    lambda x: len(eval(x)) if pd.notna(x) and x != "[]" else 0
)

# Apply enhanced cleaning filters
initial_count = len(df)
df_cleaned = df[
    (df["description_length"] >= 200) &
    (df["filtered_avg_score"] >= 0.2) &
    (df["max_score"] >= 0.4) &
    (df["num_categories"] > 0)
].copy()

filtered_count = len(df_cleaned)
dropped = initial_count - filtered_count
logger.info("Filtered out %d rows; retained %d rows (%.2f%%).", dropped, filtered_count, 100 * filtered_count / initial_count)

# Save cleaned output
df_cleaned.to_csv(OUTPUT_PATH, index=False)
logger.info("Saved cleaned dataset to: %s", OUTPUT_PATH)
print("Final cleaning complete. %d rows retained." % filtered_count)
