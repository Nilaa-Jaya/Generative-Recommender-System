# GenRec: Generative Retrieval-Augmented Recommender System

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg)

A state-of-the-art **personalized recommendation system** that combines semantic retrieval, deep learning embeddings, and large language models to generate explainable, context-aware product recommendations at scale.

## Overview

GenRec bridges traditional collaborative filtering with modern generative AI by implementing a sophisticated RAG (Retrieval-Augmented Generation) framework. The system processes millions of user reviews and histories to deliver recommendations that are not only accurate but also human-readable and explainable.

### Key Highlights

- **Scalable Retrieval**: Embedded 1.57M+ Amazon products using Sentence-BERT and indexed with FAISS for ~43ms retrieval latency
- **Personalization at Scale**: Modeled 6.7M+ user histories into 20 persona clusters with automated labeling via TF-IDF
- **Generative Recommendations**: Fine-tuned LLaMA2-7B with QLoRA (1.1M input-output pairs) for natural language explanations
- **RLHF Alignment**: Custom reward model with PPO simulation improved BERTScore from 0.79 → 0.86
- **Performance Gains**:
  - Precision@10: 0.57 → 0.69 (+21%)
  - Diversity@10: 0.42 → 0.51 (+21%)
  - Simulated CTR uplift: ~13%

---

## Architecture

GenRec implements a **4-phase pipeline** that progressively builds from semantic search to personalized generation:

```
┌─────────────────────────────────────────────────────────────────┐
│                        GENREC PIPELINE                           │
└─────────────────────────────────────────────────────────────────┘

Phase 3: Semantic Retrieval
├─ Sentence-BERT embeddings (all-mpnet-base-v2)
├─ FAISS IndexFlatIP (1.57M vectors, 768-dim)
├─ Diversity re-ranking (cosine similarity > 0.95 deduplication)
└─ Query expansion with LLM
        ↓
Phase 4: Generative Layer
├─ LLaMA2-7B + LoRA adapter
├─ Template-based prompt engineering
└─ One-sentence recommendation generation
        ↓
Phase 5: Personalization
├─ User profiling (activity, rating, keywords)
├─ K-Means clustering (20 personas)
├─ User-conditioned retrieval & generation
└─ OpenChat-3.5 for persona-aware explanations
        ↓
Phase 6: Fine-tuning & RLHF
├─ QLoRA fine-tuning on Phi-2/LLaMA2
├─ Custom reward model (personalization + sentiment + length)
├─ PPO-style optimization
└─ BERTScore evaluation
```

For detailed architecture breakdown, see [docs/architecture.md](docs/architecture.md).

---

## Technologies

| Category | Technologies |
|----------|-------------|
| **Deep Learning** | PyTorch, Transformers (HuggingFace), Sentence-Transformers |
| **LLM Training** | PEFT (LoRA/QLoRA), Accelerate, BitsAndBytes, TRL |
| **Retrieval** | FAISS, Sentence-BERT (all-mpnet-base-v2) |
| **Data Processing** | Pandas, NumPy, PyArrow (Parquet) |
| **Evaluation** | BERTScore, scikit-learn, BLEU/ROUGE/METEOR |
| **Models** | LLaMA2-7B, Phi-2, OpenChat-3.5 |
| **Infrastructure** | Google Colab (A100 GPU), Google Cloud Storage |

---

## Repository Structure

```
GenRec/
├── notebooks/                      # Jupyter notebooks (execution order)
│   ├── 01_phase3_semantic_retrieval.ipynb
│   ├── 02_phase4_generative_layer.ipynb
│   ├── 03_phase5_personalization.ipynb
│   └── 04_phase6_finetuning_rlhf.ipynb
│
├── scripts/                        # Standalone Python scripts
│   └── phase6_training.py         # Phase 6 training script
│
├── data/                           # Data files (Git LFS tracked)
│   ├── raw/                        # Raw Amazon review data
│   ├── processed/                  # Embeddings, FAISS index, mappings
│   └── clusters/                   # User clustering artifacts
│
├── docs/                           # Documentation
│   ├── architecture.md            # System architecture details
│   ├── metrics.md                 # Performance metrics & evaluation
│   └── images/                    # Diagrams and visualizations
│
├── requirements.txt               # Python dependencies
├── setup_instructions.md          # Detailed setup guide
├── .gitattributes                 # Git LFS configuration
└── LICENSE                        # MIT License
```

---

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- Git LFS installed ([installation guide](https://git-lfs.github.com/))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Nilaa-Jaya/GenRec-Generative-Retrieval-Augmented-Recommender-System-.git
   cd GenRec-Generative-Retrieval-Augmented-Recommender-System-
   ```

2. **Install Git LFS and pull data**
   ```bash
   git lfs install
   git lfs pull
   ```

3. **Download large data files (>2GB)**

   ⚠️ Three files exceed GitHub's size limit and must be downloaded separately:
   - `grouped_reviews.parquet` (6.9GB)
   - `diverse_df.parquet` (6.9GB)
   - `faiss_item_index.index` (4.5GB)

   See [data/README.md](data/README.md) for download links and installation instructions.

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   For GPU support (recommended):
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   pip install faiss-gpu
   ```

**Note:** For complete setup instructions, see [setup_instructions.md](setup_instructions.md).

### Running the Notebooks

Execute notebooks in order:

```bash
jupyter notebook notebooks/01_phase3_semantic_retrieval.ipynb
```

**Recommended Execution Order:**
1. **Phase 3**: Semantic Retrieval (builds FAISS index)
2. **Phase 4**: Generative Layer (loads pre-trained LoRA)
3. **Phase 5**: Personalization (clusters users, generates personas)
4. **Phase 6**: Fine-tuning & RLHF (trains QLoRA, runs RLHF)

See [notebooks/README.md](notebooks/README.md) for detailed instructions.

---

## Data

The project uses **Amazon product reviews** across multiple categories:

| File | Size | Description |
|------|------|-------------|
| `df10_user_history.parquet` | 1.3GB | 6.7M user-item interactions |
| `grouped_reviews.parquet` | 6.9GB | 1.57M products with aggregated reviews |
| `faiss_item_index.index` | 4.5GB | FAISS vector index (768-dim) |
| `user_embeddings.json` | 930MB | 64-dim user embeddings (DeepMF) |
| `user_cluster_map.json` | 15MB | User-to-cluster assignments |

All large files are tracked with **Git LFS**. See [data/README.md](data/README.md) for schemas and generation details.

---

## Results

### Retrieval Performance

| Metric | Baseline | After Tuning | Improvement |
|--------|----------|--------------|-------------|
| **Precision@10** | 0.57 | 0.69 | +21% |
| **Diversity@10** | 0.42 | 0.51 | +21% |
| **Latency** | - | ~43ms | - |

### Generation Quality

| Metric | Pre-RLHF | Post-RLHF | Improvement |
|--------|----------|-----------|-------------|
| **BERTScore F1** | 0.79 | 0.86 | +8.8% |
| **Semantic Alignment** | Good | Excellent | - |

### Business Impact

- **Simulated CTR Uplift**: ~13% on top-10 ranked items
- **User Engagement**: Improved through explainable recommendations
- **Scalability**: Sub-50ms retrieval supports real-time serving

For detailed metrics and evaluation methodology, see [docs/metrics.md](docs/metrics.md).

---

## Key Features

### 1. Hybrid Retrieval Strategy
- **Two-stage retrieval**: FAISS for recall, diversity re-ranking for precision
- **Deduplication**: Removes near-duplicates (cosine similarity > 0.95)
- **Query expansion**: LLM-based query diversification for better coverage

### 2. User Personalization
- **Persona clustering**: Automatic user segmentation into 20 personas
- **Behavioral modeling**: Average rating, review sentiment, keyword extraction
- **Cold-start handling**: Cluster-based fallback for new users

### 3. Explainable AI
- **Natural language rationales**: "You might enjoy X because..."
- **User-context awareness**: References user history and preferences
- **Transparency**: Clear reasoning improves trust and engagement

### 4. Production-Ready Design
- **Modular architecture**: Each phase is independently executable
- **Efficient training**: QLoRA enables 7B model fine-tuning on single GPU
- **Reproducible**: Complete pipeline from raw data to trained models

---

## Use Cases

- **E-commerce recommendation engines**
- **Content discovery platforms**
- **Personalized marketing campaigns**
- **Research on RAG + recommender systems**
- **LLM fine-tuning and alignment case studies**

---

## Future Work

- [ ] Implement multi-armed bandit for online exploration-exploitation
- [ ] Add session-based recommendations with temporal modeling
- [ ] Deploy as REST API with FastAPI
- [ ] A/B testing framework for live evaluation
- [ ] Multi-modal recommendations (text + images)
- [ ] Incorporate real-time user feedback loop

---

## Citation

If you use this work in your research, please cite:

```bibtex
@software{genrec2024,
  author = {Your Name},
  title = {GenRec: Generative Retrieval-Augmented Recommender System},
  year = {2024},
  url = {https://github.com/yourusername/GenRec}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Amazon Reviews Dataset**: Source of product reviews and user interactions
- **HuggingFace**: Transformers library and model hub
- **Facebook AI**: FAISS library for efficient similarity search
- **Anthropic Claude**: Development assistance

---

## Contact

**Author**: Your Name
**Email**: your.email@example.com
**LinkedIn**: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)
**GitHub**: [@yourusername](https://github.com/yourusername)

---

<p align="center">
  <b>⭐ Star this repository if you found it helpful!</b>
</p>
