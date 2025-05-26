import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# Setup paths and logging
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app.classes.main_helper import MainHelper
LOG_NAME = "category_refined_exploration"
helper = MainHelper(REPO_ROOT, LOG_NAME)
logger = helper.logger

INPUT_PATH = os.path.join(helper.get_path("data"), "books_refined_categories.csv")
FIG_DIR = helper.get_path("figures")

# Load data
df = pd.read_csv(INPUT_PATH)
logger.info("Loaded dataset with %d entries for exploration.", df.shape[0])

# Description length and score stats
df["description_length"] = df["augmented_description"].fillna("").astype(str).str.len()
df["average_score"] = df["predicted_scores"].apply(
    lambda x: pd.Series(eval(x)).mean() if pd.notna(x) and x != "{}" else 0
)
df["max_score"] = df["predicted_scores"].apply(
    lambda x: max(eval(x).values()) if pd.notna(x) and x != "{}" else 0
)

# Correlation calculations
pearson_corr, pearson_p = pearsonr(df["description_length"], df["average_score"])
spearman_corr, spearman_p = spearmanr(df["description_length"], df["average_score"])
logger.info("Pearson correlation: r=%.4f (p=%.4f)", pearson_corr, pearson_p)
logger.info("Spearman correlation: rho=%.4f (p=%.4f)", spearman_corr, spearman_p)

# Scatter plot with regression line
plt.figure(figsize=(10, 6))
sns.regplot(x="description_length", y="average_score", data=df, scatter_kws={"s": 10}, line_kws={"color": "red"})
plt.title("Description Length vs. Average Prediction Score")
plt.xlabel("Description Length (characters)")
plt.ylabel("Average Model Score")
plt.grid(True)
plt.tight_layout()
scatter_path = os.path.join(FIG_DIR, "category_refined_description_length_vs_avg_score.png")
plt.savefig(scatter_path)
plt.close()
logger.info("Saved scatter plot to: %s", scatter_path)

# Boxplot by length bins
df["length_bin"] = pd.cut(df["description_length"], bins=[0, 250, 500, 750, 1000, 1500, 2000, float("inf")])
plt.figure(figsize=(12, 6))
sns.boxplot(x="length_bin", y="average_score", data=df)
plt.title("Boxplot of Average Score by Description Length Bins")
plt.xlabel("Description Length (character bins)")
plt.ylabel("Average Model Score")
plt.xticks(rotation=30)
plt.tight_layout()
boxplot_path = os.path.join(FIG_DIR, "category_refined_avg_score_by_length_bin.png")
plt.savefig(boxplot_path)
plt.close()
logger.info("Saved boxplot to: %s", boxplot_path)

print("Exploration complete. Figures and logs updated.")
