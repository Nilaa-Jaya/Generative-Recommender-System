# GenRec Notebooks

This directory contains the core Jupyter notebooks that implement the GenRec pipeline. The notebooks are designed to be executed **sequentially**, as each phase builds on the outputs of previous phases.

## Execution Order

Execute notebooks in this order:

```
01_phase3_semantic_retrieval.ipynb (Phase 3)
    ↓
02_phase4_generative_layer.ipynb (Phase 4)
    ↓
03_phase5_personalization.ipynb (Phase 5)
    ↓
04_phase6_finetuning_rlhf.ipynb (Phase 6)
```

---

## Notebook Descriptions

### 📘 01_phase3_semantic_retrieval.ipynb

**Phase 3: Semantic Embedding and Retrieval**

Build a semantic search engine over 1.57M Amazon products using Sentence-BERT and FAISS.

**Key Tasks:**
- Load and aggregate product reviews by ASIN
- Encode all items using `all-mpnet-base-v2` (768-dim embeddings)
- Build FAISS IndexFlatIP for fast similarity search
- Implement diversity re-ranking to remove near-duplicates
- Add LLM-based query expansion

**Input Files:**
- `data/raw/grouped_reviews.parquet` (1.57M products)

**Output Files:**
- `data/processed/faiss_item_index.index` (4.5GB)
- `data/processed/asin_mapping.csv` (17MB)
- `data/processed/diverse_df.parquet` (6.9GB)

**Runtime:** ~30-45 minutes (GPU), ~2-3 hours (CPU)

**Hardware Requirements:**
- GPU: 16GB+ VRAM (recommended)
- CPU: 16GB+ RAM
- Storage: 20GB+ free space

---

### 📗 02_phase4_generative_layer.ipynb

**Phase 4: Generative Recommendations with LLaMA2**

Add natural language generation on top of retrieval using a fine-tuned LLaMA2 model.

**Key Tasks:**
- Load LLaMA2-7B with pre-trained LoRA adapter
- Design prompt templates for recommendations
- Generate one-sentence explanations for retrieved items
- Evaluate with BLEU, ROUGE, METEOR
- Measure retrieval + generation latency

**Input Files:**
- `data/processed/grouped_reviews.parquet`
- `data/processed/faiss_item_index.index`
- `data/processed/asin_mapping.csv`
- Pre-trained LoRA weights (from Google Drive)

**Output Files:**
- Sample recommendations (inline display)

**Runtime:** ~20-30 minutes (with pre-loaded model)

**Hardware Requirements:**
- GPU: 24GB+ VRAM (for LLaMA2-7B)
- CPU: Not recommended (too slow)

---

### 📙 03_phase5_personalization.ipynb

**Phase 5: User Profiling and Persona-Based Recommendations**

Build user-centric recommendations by clustering 6.7M user histories into personas.

**Key Tasks:**
- Load user-item interaction history
- Aggregate per-user statistics (reviews, ratings, keywords)
- Build persona strings from user profiles
- Train DeepMF model for user embeddings (64-dim)
- Cluster users with K-Means (20 clusters)
- Generate cluster labels with TF-IDF
- Implement user-conditioned retrieval and generation

**Input Files:**
- `data/raw/df10_user_history.parquet` (1.3GB, 6.7M interactions)
- `data/processed/faiss_item_index.index`
- `data/processed/grouped_reviews.parquet`

**Output Files:**
- `data/processed/user_embeddings.json` (930MB)
- `data/clusters/user_cluster_map.json` (15MB)
- `data/clusters/cluster_to_label.json` (1.1KB)
- `data/clusters/product_info.json` (1.3KB)

**Runtime:** ~1-2 hours (GPU), ~4-6 hours (CPU)

**Hardware Requirements:**
- GPU: 12GB+ VRAM
- RAM: 32GB+ (for large data processing)

---

### 📕 04_phase6_finetuning_rlhf.ipynb

**Phase 6: QLoRA Fine-tuning and RLHF**

Fine-tune a language model on recommendation tasks and align with human preferences using RLHF.

**Key Tasks:**
- Analyze user history distributions (activity, ratings, text length)
- Create 100K+ training pairs (user profile + item → recommendation)
- Fine-tune Phi-2 (or LLaMA2-7B) with QLoRA
- Design custom reward model (personalization, sentiment, length, diversity)
- Simulate PPO-style RLHF loop (50 episodes)
- Evaluate with BERTScore

**Input Files:**
- `data/raw/df10_user_history.parquet`
- `data/raw/grouped_reviews.parquet`
- `data/clusters/user_cluster_map.json`
- `data/clusters/cluster_to_label.json`

**Output Files:**
- Model checkpoints (saved to Google Drive)
- Training logs
- BERTScore evaluation results

**Runtime:** ~2-4 hours (A100 GPU), longer on smaller GPUs

**Hardware Requirements:**
- GPU: A100 (40GB) or V100 (32GB) recommended
- RAM: 32GB+
- Storage: 50GB+ for model checkpoints

---

## Running in Google Colab

All notebooks are **Colab-compatible** and include automatic setup:

1. **Open notebook in Colab**
   - Click "Open in Colab" badge at the top of each notebook
   - Or upload manually to Google Colab

2. **Mount Google Drive** (for data storage)
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. **Install dependencies**
   ```python
   !pip install -q -r requirements.txt
   ```

4. **Select GPU runtime**
   - Runtime → Change runtime type → GPU → A100 (recommended)

---

## Running Locally

1. **Install Jupyter**
   ```bash
   pip install jupyter ipywidgets
   ```

2. **Start Jupyter**
   ```bash
   jupyter notebook
   ```

3. **Update file paths**
   - Replace Google Drive paths (`/content/drive/MyDrive/...`) with local paths
   - Example: Change to `../data/raw/df10_user_history.parquet`

---

## Notebook Best Practices

### Memory Management
- Use `del` to free memory after large operations
- Restart kernel between phases if needed
- Monitor GPU memory with `nvidia-smi`

### Checkpointing
- Save intermediate results frequently
- Use `tqdm` progress bars for long operations
- Enable auto-save in Jupyter settings

### Reproducibility
- Set random seeds: `np.random.seed(42)`, `torch.manual_seed(42)`
- Document hyperparameters clearly
- Save configuration dictionaries

---

## Troubleshooting

### Common Issues

**1. Out of Memory (OOM)**
- Reduce batch size in encoding/training
- Use gradient checkpointing
- Enable `low_cpu_mem_usage=True` when loading models

**2. FAISS Index Not Found**
- Ensure Phase 3 completed successfully
- Check file paths match your environment
- Re-run FAISS index building section

**3. Colab Disconnects**
- Use Colab Pro for longer runtimes
- Save checkpoints frequently
- Use `keep_alive.js` script for background connections

**4. ModuleNotFoundError**
- Reinstall requirements: `!pip install -r requirements.txt`
- Check Python version (3.8+ required)

---

## Performance Tips

1. **Use GPU runtime** for all phases (10-50x faster)
2. **Enable mixed precision** (bf16/fp16) for training
3. **Batch operations** instead of loops
4. **Cache embeddings** to avoid recomputation
5. **Use multiprocessing** for data loading

---

## Expected Outputs

After running all notebooks, you should have:

✅ FAISS index with 1.57M product vectors
✅ User embeddings for 728K+ users
✅ 20 user persona clusters with labels
✅ Fine-tuned QLoRA model weights
✅ BERTScore evaluation showing ~0.86 F1
✅ Sample recommendations with explanations

---

## Additional Resources

- [Main README](../README.md) - Project overview
- [Architecture Documentation](../docs/architecture.md) - System design
- [Metrics & Evaluation](../docs/metrics.md) - Performance details
- [Setup Instructions](../setup_instructions.md) - Detailed setup guide

---

## Questions?

If you encounter issues:
1. Check [setup_instructions.md](../setup_instructions.md)
2. Review notebook markdown cells for guidance
3. Open an issue on GitHub

Happy experimenting! 🚀
