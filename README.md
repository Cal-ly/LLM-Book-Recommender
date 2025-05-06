# Local Semantic Book Recommender

A lightweight book recommendation system that runs entirely **offline**.
This project uses local models and vector search to match user queries to book descriptions.

## 🚀 Features

- All computation runs locally — no OpenAI or internet access required
- Uses [sentence-transformers](https://www.sbert.net/) for semantic embeddings
- Fast similarity search with [FAISS](https://github.com/facebookresearch/faiss)
- Clean dataset from Kaggle: [7k Books with Metadata](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata)
- Simple UI built with [Streamlit](https://streamlit.io/)
- Fetch book covers via ISBN using Open Library or Google Books

## 🗂️ Project Structure

```plaintext
book-recommender/
├── app/ # Streamlit app
│ └── main.py
│
├── data/ # Raw dataset
│ └── books.csv
│
├── data-exploration/ # Jupyter/data analysis scripts
│ ├── explore_data.py
│ └── output/
│
├── embeddings/ # FAISS index and book metadata
│ ├── index.faiss
│ └── books_indexed.csv
│
├── models/ # Local model info
│
├── scripts/ # Embedding and indexing logic
│ └── embed_books.py
│
├── requirements.txt
└── README.md
```

---

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

Dependencies:

sentence-transformers

faiss-cpu

streamlit

pandas, numpy, seaborn, matplotlib

### 📊 1. Data Exploration
Explore the dataset and clean it:

```bash
python data-exploration/explore_data.py
```

Outputs:

Cleaned CSV: data-exploration/output/books_cleaned.csv

Charts and logs for insights

### 🧠 2. Embed and Index
Generate local embeddings and build FAISS index:

```bash
python scripts/embed_books.py
```

Outputs:

FAISS index: embeddings/index.faiss

Book metadata: embeddings/books_indexed.csv

### 🌐 3. Run the App
```bash
streamlit run app/main.py
```

Then open your browser at http://localhost:8501.

### ✅ Example Query

```plaintext
"A dystopian future controlled by artificial intelligence"
```

### 📌 Notes
- Designed to work fully offline (except optional book cover lookups).
- Tested with a dataset of ~7000 entries — easily runs on consumer hardware.
- Hardware used: Intel Core i9, 32 GB RAM, NVIDIA 4070 GPU

### 📄 License