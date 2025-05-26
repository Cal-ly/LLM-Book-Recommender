import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import multilabel_confusion_matrix
from collections import Counter
from transformers import pipeline
from tqdm import tqdm

# Flag for testing. Set to True for testing with a smaller dataset
IS_TEST = False
#IS_TEST = True

# Setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app.classes.main_helper import MainHelper
LOG_NAME = "data_refine_categories"
helper = MainHelper(REPO_ROOT, LOG_NAME)
logger = helper.logger

INPUT_PATH = os.path.join(helper.get_path("data"), "books_cleaned.csv")
OUTPUT_PATH = os.path.join(helper.get_path("data"), "books_refined_categories.csv")
FIG_DIR = helper.get_path("figures")

# Define candidate labels (master list)
categories = [
    "Fantasy",
    "Science Fiction",
    "Love",
    "Mystery",                 # Merged Suspense + Detective
    "Adventure",
    "Personal Development",
    "Young Adult",
    "Children's",
    "Religion",
    "Historical",              # Merged Biography + History
    "Philosophy & Poetry",     # Merged Poetry + Philosophy
    "Non-fiction",
    "Horror"
]

# Refined and expanded keyword-based fallback rules
fallback_keywords = {
    "Love": [
        "romantic", "heartbreak", "affair", "soulmate", "first kiss", "wedding", "honeymoon", 
        "jealousy", "crush", "proposal", "valentine", "courtship", "infatuation", "romance", 
        "passion", "love triangle", "devotion", "intimacy", "relationship", "flirting", "adoration"
    ],
    "Horror": [
        "exorcism", "paranormal", "creepy doll", "possession", "evil spirit", "slasher", "nightmare", 
        "macabre", "phantom", "demonic", "supernatural", "terror", "darkness", "eerie", "chilling", 
        "occult", "fear"
    ],
    "Personal Development": [
        "self-help", "resilience", "mindset", "coaching", "productivity", "goal setting", 
        "mental health", "self-discipline", "time management", "career growth", "success guide",
        "motivation", "empowerment", "personal growth", "life skills", "self-improvement", 
        "confidence", "leadership", "habits", "inspiration", "well-being", "self-awareness"
    ],
    "Children's": [
        "storybook", "nursery rhyme", "learning for kids", "bedtime story", "rhyming book", 
        "picture book", "toddler", "preschool", "fairy tale for kids", "talking animals", 
        "fun for children", "illustrated book", "adventure for kids", "playful", "imagination", 
        "childhood", "friendship", "family", "magic", "whimsical", "cartoon", "young reader"
    ],
    "Fantasy": [
        "hobbit", "orc", "middle-earth", "elven", "gondor", "dragonrider", "griffin", 
        "wizarding world", "fae realm", "necromancer", "dark lord", "sorcery", "prophecy",
        "magic", "dragon", "wizard", "kingdom", "sword", "spell", "myth", "realm", "enchanted",
        "fairy", "troll", "epic", "legend", "hero", "castle", "alchemy", "mystical", "mythical"
    ],
    "Science Fiction": [
        "vulcan", "klingon", "warp drive", "starfleet", "neuralink", "nanobot", "terraforming", 
        "android uprising", "parallel dimension", "quantum singularity", "cybernetic", "space",
        "alien", "robot", "future", "dystopia", "cyberpunk", "galaxy", "time travel", "AI",
        "spaceship", "interstellar", "extraterrestrial", "technology", "utopia", "cyborg", 
        "genetics", "post-apocalyptic", "artificial intelligence", "space exploration"
    ],
    "Mystery": [
        "locked room", "whodunit", "private investigator", "cold case", "missing person", 
        "trail of clues", "evidence", "forensics", "unsolved mystery", "twist ending", 
        "interrogation", "red herring", "detective", "crime", "investigation", "suspect", 
        "alibi", "case", "clue", "sleuth", "spy", "riddle", "noir", "justice", "criminal"
    ],
    "Young Adult": [
        "high school drama", "first crush", "coming-of-age", "teen rebellion", "school dance", 
        "prom queen", "college tour", "bullying", "friendship bracelet", "diary entry", 
        "teenager", "peer pressure", "self-discovery", "adolescence", "family drama", "youth",
        "identity", "dreams", "relationships", "growth", "coming of age", "teen romance"
    ],
    "Adventure": [
        "lost city", "hidden treasure", "jungle expedition", "desert crossing", 
        "ancient map", "mountain quest", "pirate island", "globe-trotting", 
        "skydiving", "deep sea dive", "outback survival", "exploration", "journey", 
        "quest", "expedition", "travel", "discovery", "wilderness", "survival", 
        "challenge", "voyage", "treasure", "thrill", "unknown", "explorer", "mission"
    ],
    "Historical": [
        "wwii", "cold war", "napoleonic", "renaissance", "medieval", "victorian england", 
        "tudor", "civil war", "ancient rome", "historical diary", "reformation", 
        "berlin wall", "apollo 11", "historical", "past", "ancient", "civilization", 
        "war", "event", "timeline", "revolution", "dynasty", "empire", "archaeology", 
        "heritage", "culture", "tradition", "military", "battle", "chronicle", "monarchy"
    ],
    "Religion": [
        "biblical", "scripture", "holy text", "jesus", "quran", "faith journey", "pilgrimage", 
        "divine", "spiritual reflection", "sacred", "prayer", "worship", "church", "mosque", 
        "temple", "religious", "spirituality", "god", "belief", "ritual", "holy", "devotion"
    ],
    "Philosophy & Poetry": [
        "plato", "aristotle", "stoicism", "existentialism", "kant", "socrates", 
        "rainer rilke", "haiku", "lyrical", "prose poem", "verses", "introspection", 
        "meaning of life", "moral paradox", "philosophy", "thought", "ethics", "morality", 
        "mind", "reasoning", "logic", "metaphysics", "epistemology", "wisdom", "truth", 
        "justice", "virtue", "human nature", "consciousness", "free will", "ideology", 
        "theory", "reflection", "abstract", "debate", "poetry", "rhyme", "stanza", "lyric"
    ],
    "Non-fiction": [
        "case study", "research analysis", "true account", "reporting", "scientific study", 
        "real events", "journalistic", "field notes", "eyewitness", "infographic", 
        "documentary narrative", "non-fiction", "fact", "reality", "true story", 
        "knowledge", "education", "research", "analysis", "truth", "history", "science", 
        "report", "study", "journalism", "biography", "memoir", "guide", "manual", "reference"
    ]
}

# Load dataset
if IS_TEST:
    # For testing, use a smaller dataset
    df = pd.read_csv(INPUT_PATH).sample(frac=0.05, random_state=42)
    logger.info("Using a sample of %d entries for testing.", df.shape[0])
else:
    # For production, load the full dataset
    df = pd.read_csv(INPUT_PATH)
    logger.info("Loaded dataset with %d entries for category refinement.", df.shape[0])

# Construct full title
df['full_title'] = df.apply(
    lambda row: f"{str(row['title']).strip()}: {str(row['subtitle']).strip()}" if pd.notna(row['subtitle']) and str(row['subtitle']).strip() else str(row['title']).strip(),
    axis=1
)

# Create augmented description
df['augmented_description'] = df.apply(
    lambda row: f"Title: {row['full_title']}. Author: {row['authors']}. Published: {row['published_year']}. Description: {row['description']}",
    axis=1
)

# Log length stats
df['augmented_length'] = df['augmented_description'].str.len()
logger.info("Augmented description length: min=%d, max=%d, mean=%.2f", df['augmented_length'].min(), df['augmented_length'].max(), df['augmented_length'].mean())

# Check if GPU is available
if torch.cuda.is_available():
    logger.info("Using GPU for zero-shot classification.")
    device = 0
else:
    logger.info("Using CPU for zero-shot classification.")
    device = -1

# Load zero-shot classifier
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=device
)

# Predict using classifier as dataset
descriptions = df['augmented_description'].fillna('').astype(str).tolist()
logger.info("Running dataset-based inference on %d entries...", len(descriptions))
results = classifier(descriptions, candidate_labels=categories, multi_label=True)

predicted = []
scores = []
for r in results:
    labels = [label for label, score in zip(r['labels'], r['scores']) if score >= 0.4]
    predicted.append(labels)
    scores.append({label: float(score) for label, score in zip(r['labels'], r['scores'])})

df['predicted_categories'] = predicted
df['predicted_scores'] = scores

# Fallback keyword prediction
def apply_fallback(description, current_preds):
    text = str(description).lower()
    fallbacks = []
    for cat, keywords in fallback_keywords.items():
        if cat not in current_preds and any(kw in text for kw in keywords):
            fallbacks.append(cat)
    return fallbacks

df['fallback_categories'] = df.apply(lambda row: apply_fallback(row['augmented_description'], row['predicted_categories']), axis=1)
logger.info("Applied fallback keyword logic. Total rows with fallback categories: %d", df['fallback_categories'].apply(len).gt(0).sum())

# Parse original categories into list
df['original_categories'] = df['final_categories'].fillna('').apply(lambda x: [cat.strip().title() for cat in x.split(',') if cat.strip()])

# Conflicting flag
df['category_conflict'] = df.apply(lambda row: not any(cat in row['original_categories'] for cat in row['predicted_categories']), axis=1)

# Confusion matrix setup (flatten to binary relevance)
def flatten_multilabels(series, all_labels):
    return np.array([[label in labels for label in all_labels] for labels in series])

y_true = flatten_multilabels(df['original_categories'], categories)
y_pred = flatten_multilabels(df['predicted_categories'], categories)

# Plot true positives
cm = multilabel_confusion_matrix(y_true, y_pred)
category_match_counts = [(categories[i], cm[i][1,1]) for i in range(len(categories))]
category_match_counts.sort(key=lambda x: x[1], reverse=True)
labels, matches = zip(*category_match_counts)

plt.figure(figsize=(12, 6))
sns.barplot(x=list(labels), y=list(matches))
plt.title("Correct Predictions per Category")
plt.ylabel("True Positives")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "refine_category_prediction_matches.png"))
logger.info("Saved category match visualization.")

# Per-category metrics
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
metrics_df = pd.DataFrame({
    'Category': categories,
    'Precision': precision,
    'Recall': recall,
    'F1': f1
})
metrics_df.to_csv(os.path.join(helper.get_path("data"), "refine_category_metrics.csv"), index=False)
logger.info("Saved per-category precision/recall/f1 to CSV.")

# Plot all metrics in subplots
fig, ax = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

sns.barplot(data=metrics_df, x="Category", y="Precision", ax=ax[0])
ax[0].set_title("Precision by Category")
ax[0].set_ylabel("Precision")

sns.barplot(data=metrics_df, x="Category", y="Recall", ax=ax[1])
ax[1].set_title("Recall by Category")
ax[1].set_ylabel("Recall")

sns.barplot(data=metrics_df, x="Category", y="F1", ax=ax[2])
ax[2].set_title("F1 Score by Category")
ax[2].set_ylabel("F1 Score")

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "refine_category_metrics_plot.png"))
logger.info("Saved bar plots for per-category precision/recall/f1.")

# Save conflict examples
conflict_samples = df[df['category_conflict']][['title', 'description', 'original_categories', 'predicted_categories', 'predicted_scores']]
conflict_samples['short_description'] = conflict_samples['description'].str.slice(0, 100)
conflict_samples.drop(columns='description', inplace=True)
conflict_samples.to_csv(os.path.join(helper.get_path("data"), "refine_conflict_samples.csv"), index=False)
logger.info("Saved sample conflicts for inspection.")

# Save final output
df.to_csv(OUTPUT_PATH, index=False)
logger.info("Saved refined dataset with predicted categories to: %s", OUTPUT_PATH)
print("Category refinement complete. Output saved.")