# GenRec Data Directory

This directory contains all data files used in the GenRec pipeline, organized by processing stage. All large files (>10MB) are tracked using **Git LFS** (Large File Storage).

---

## Directory Structure

```
data/
├── raw/                    # Raw input data (source datasets)
├── processed/              # Intermediate processed data (embeddings, indices)
└── clusters/               # User clustering artifacts
```

---

## Large File Downloads

⚠️ **Important:** Three data files exceed GitHub's 2GB limit and must be downloaded separately:

### Files Not Included in Repository (>2GB)

| File | Size | Download Link |
|------|------|---------------|
| `grouped_reviews.parquet` | 6.9GB | [Download from Google Drive](#) |
| `diverse_df.parquet` | 6.9GB | [Download from Google Drive](#) |
| `faiss_item_index.index` | 4.5GB | [Download from Google Drive](#) |

**Installation:**
1. Download the above files from the provided links
2. Place them in the correct directories:
   - `grouped_reviews.parquet` → `data/raw/`
   - `diverse_df.parquet` → `data/processed/`
   - `faiss_item_index.index` → `data/processed/`

**Alternative:** You can regenerate these files by running Phase 3 notebook from scratch (see `notebooks/01_phase3_semantic_retrieval.ipynb`).

---

## Git LFS Setup

Files tracked by Git LFS (included in repository):
- `df10_user_history.parquet` (1.3GB)
- `user_embeddings.json` (930MB)
- `user_cluster_map.json` (15MB)
- `asin_mapping.csv` (17MB)

**Setup:**
```bash
# Install Git LFS
git lfs install

# Pull all LFS-tracked files
git lfs pull
```

---

## Data Files

### 📁 `raw/` - Raw Input Data

#### `df10_user_history.parquet` (1.3GB)
**Source:** Amazon product reviews dataset
**Rows:** 6,765,421 interactions
**Schema:**
```
- reviewerID (string): Unique user identifier
- asin (string): Amazon Standard Identification Number (product ID)
- rating (float): User rating (1.0 - 5.0)
- reviewText (string): Full review text
- summary (string): Review summary/title
- unixReviewTime (int): Timestamp of review
- category (string): Product category
```

**Purpose:** User-item interaction history for personalization and training

**Sample:**
```
reviewerID    asin         rating  reviewText
U123456789    B001ABC123   5.0     "Great product! Works perfectly..."
U987654321    B002DEF456   4.0     "Good value for money..."
```

---

#### `grouped_reviews.parquet` (6.9GB)
**Source:** Processed from `df10_user_history.parquet`
**Rows:** 1,574,844 unique products
**Schema:**
```
- asin (string): Product ID
- text (string): Concatenated review texts (all reviews for this product)
- avg_rating (float): Average rating across all reviews
- review_count (int): Number of reviews for this product
- category (string): Product category
```

**Purpose:** Item-level aggregated text for semantic embedding

**Sample:**
```
asin         text                           avg_rating  review_count
B001ABC123   "Great product! Works... | ..." 4.5         127
B002DEF456   "Good value... | ..."           4.2         89
```

**Processing Pipeline:**
1. Group `df10_user_history` by `asin`
2. Concatenate all `reviewText` fields (separated by " | ")
3. Compute aggregate statistics (mean rating, count)
4. Save as Parquet for efficient columnar access

---

### 📁 `processed/` - Processed & Intermediate Data

#### `diverse_df.parquet` (6.9GB)
**Source:** Diversity-filtered subset of `grouped_reviews.parquet`
**Rows:** ~1,574,844 products (after deduplication)
**Schema:** Same as `grouped_reviews.parquet`

**Purpose:** Diversified product set with near-duplicates removed (cosine similarity > 0.95)

**Processing:**
- Build item-to-item similarity matrix
- Remove products with >0.95 similarity to others
- Retain most diverse representatives

---

#### `faiss_item_index.index` (4.5GB)
**Type:** FAISS IndexFlatIP (Inner Product)
**Vectors:** 1,574,844 items × 768 dimensions
**Model:** `sentence-transformers/all-mpnet-base-v2`

**Purpose:** Fast approximate nearest neighbor search for item retrieval

**Index Details:**
- **Index Type:** Flat (exhaustive search, 100% recall)
- **Metric:** Inner Product (equivalent to cosine similarity for normalized vectors)
- **Normalization:** All vectors L2-normalized before indexing
- **Search Speed:** ~43ms average latency for top-k retrieval

**Usage:**
```python
import faiss
index = faiss.read_index("data/processed/faiss_item_index.index")
distances, indices = index.search(query_vector, k=10)
```

---

#### `asin_mapping.csv` (17MB)
**Rows:** 1,574,844 products
**Schema:**
```
- index (int): Position in FAISS index (0-indexed)
- asin (string): Product ID
```

**Purpose:** Maps FAISS index positions to Amazon product IDs

**Why Needed:**
FAISS only stores vectors (no metadata). This file bridges index positions to product identifiers.

**Usage:**
```python
import pandas as pd
mapping = pd.read_csv("data/processed/asin_mapping.csv")
asin_list = mapping.iloc[faiss_indices]['asin'].tolist()
```

---

#### `user_embeddings.json` (930MB)
**Format:** JSON dictionary
**Entries:** 728,000+ users
**Schema:**
```json
{
  "user_id": [0.12, -0.45, 0.78, ...],  // 64-dimensional embedding
  ...
}
```

**Purpose:** Low-dimensional user representations learned from DeepMF (Deep Matrix Factorization)

**Model Details:**
- **Architecture:** DeepMF (user embeddings + item embeddings + MLP)
- **Embedding Dimension:** 64
- **Training:** 5 epochs on 6.7M interactions
- **Loss:** Mean Squared Error (rating prediction)

**Usage:**
```python
import json
with open("data/processed/user_embeddings.json") as f:
    user_embs = json.load(f)
user_vector = np.array(user_embs["U123456789"])
```

---

### 📁 `clusters/` - User Clustering Artifacts

#### `user_cluster_map.json` (15MB)
**Format:** JSON dictionary
**Entries:** 728,000+ users
**Schema:**
```json
{
  "user_id": 12,  // Cluster ID (0-19)
  ...
}
```

**Purpose:** Maps each user to their assigned persona cluster

**Clustering Method:**
- K-Means with k=20 clusters
- Features: 64-dim user embeddings from DeepMF
- Initialization: K-Means++

---

#### `cluster_to_label.json` (1.1KB)
**Format:** JSON dictionary
**Entries:** 20 clusters
**Schema:**
```json
{
  "0": "interested in electronics, gadgets, phone",
  "1": "interested in books, reading, novel",
  ...
}
```

**Purpose:** Human-readable labels for each persona cluster

**Generation Method:**
1. Aggregate all review texts for users in each cluster
2. Compute TF-IDF scores
3. Extract top-5 keywords per cluster
4. Format as "interested in [keyword1], [keyword2], ..."

---

#### `product_info.json` (1.3KB)
**Format:** JSON dictionary
**Sample Schema:**
```json
{
  "B001ABC123": {
    "category": "Electronics",
    "avg_rating": 4.5,
    "review_count": 127
  }
}
```

**Purpose:** Quick lookup for product metadata (used in prompts and generation)

---

## Data Sources

### Amazon Product Reviews Dataset
- **Source:** [Amazon Review Data (2018)](http://jmcauley.ucsd.edu/data/amazon/)
- **Citation:**
  ```
  Justifying recommendations using distantly-labeled reviews and fined-grained aspects
  Jianmo Ni, Jiacheng Li, Julian McAuley
  Empirical Methods in Natural Language Processing (EMNLP), 2019
  ```
- **License:** Used for research and educational purposes

---

## Data Generation Pipeline

The full data processing pipeline:

```
Amazon Reviews (Raw JSON/CSV)
    ↓
[Phase 1-2: Preprocessing]
    ↓
df10_user_history.parquet (user-item interactions)
grouped_reviews.parquet (item-level aggregation)
    ↓
[Phase 3: Embedding & Indexing]
    ↓
faiss_item_index.index (FAISS vectors)
asin_mapping.csv (index mapping)
diverse_df.parquet (deduplicated items)
    ↓
[Phase 4: User Modeling]
    ↓
user_embeddings.json (64-dim user vectors)
    ↓
[Phase 5: Clustering]
    ↓
user_cluster_map.json (cluster assignments)
cluster_to_label.json (persona labels)
product_info.json (metadata)
```

---

## Storage Requirements

| Directory | Size | File Count |
|-----------|------|------------|
| `raw/` | 8.2GB | 2 files |
| `processed/` | 12.3GB | 4 files |
| `clusters/` | 15MB | 3 files |
| **Total** | **~20.5GB** | **9 files** |

**Recommendation:** Ensure at least 35GB free disk space for comfortable operation.

---

## Data Statistics

### User Activity Distribution
- **Mean reviews per user:** 9.3
- **Median reviews per user:** 5
- **Users with 10+ reviews:** 285,000 (39%)
- **Users with 50+ reviews:** 12,500 (1.7%)

### Item Popularity Distribution
- **Mean reviews per item:** 4.3
- **Median reviews per item:** 2
- **Items with 10+ reviews:** 156,000 (9.9%)
- **Items with 100+ reviews:** 8,500 (0.5%)

### Rating Distribution
| Rating | Count | Percentage |
|--------|-------|------------|
| 5.0 | 3,910,000 | 57.8% |
| 4.0 | 1,520,000 | 22.5% |
| 3.0 | 685,000 | 10.1% |
| 2.0 | 365,000 | 5.4% |
| 1.0 | 285,000 | 4.2% |

**Observation:** Ratings are heavily skewed toward 5 stars (positive sentiment bias).

---

## Data Privacy & Ethics

- **Anonymization:** All user IDs are anonymized by Amazon
- **No PII:** Dataset contains no personally identifiable information
- **Research Use:** Data used strictly for academic/research purposes
- **Compliance:** Usage follows Amazon's data sharing guidelines

---

## Regenerating Data

If you need to regenerate any intermediate files:

1. **FAISS Index**
   ```bash
   jupyter notebook notebooks/01_phase3_semantic_retrieval.ipynb
   # Run cells up to "Building FAISS Index"
   ```

2. **User Embeddings**
   ```bash
   jupyter notebook notebooks/03_phase5_personalization.ipynb
   # Run "Train DeepMF Model" section
   ```

3. **Cluster Labels**
   ```bash
   jupyter notebook notebooks/03_phase5_personalization.ipynb
   # Run "K-Means Clustering" and "Generate Cluster Labels" sections
   ```

---

## Questions?

For data-related issues:
- Check [setup_instructions.md](../setup_instructions.md) for Git LFS setup
- See notebook READMEs for data generation steps
- Open a GitHub issue for missing or corrupted files

---

**Last Updated:** 2025
**Maintained By:** GenRec Team
