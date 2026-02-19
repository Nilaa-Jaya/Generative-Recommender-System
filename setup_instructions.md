# GenRec Setup Instructions

Complete setup guide for running GenRec locally or on Google Colab. Tested on Python 3.10+.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Installation](#local-installation)
3. [Google Colab Setup](#google-colab-setup)
4. [Git LFS Setup](#git-lfs-setup)
5. [GPU Configuration](#gpu-configuration)
6. [Troubleshooting](#troubleshooting)
7. [Verification](#verification)

---

## Prerequisites

### System Requirements

**Minimum (CPU-only):**
- Python 3.8+
- 16GB RAM
- 50GB free disk space
- Ubuntu 18.04+ / macOS 10.15+ / Windows 10+

**Recommended (GPU):**
- Python 3.8+
- CUDA 11.8+ compatible GPU (16GB+ VRAM)
- 32GB RAM
- 100GB free disk space
- Ubuntu 20.04+ / Windows 11

**For Training (Phase 6):**
- NVIDIA A100 (40GB) or V100 (32GB)
- 64GB+ system RAM
- 200GB+ free disk space

---

## Local Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/GenRec.git
cd GenRec
```

### Step 2: Set Up Virtual Environment

**Option A: Using venv (recommended)**
```bash
# Create virtual environment
python -m venv genrec-env

# Activate (Linux/macOS)
source genrec-env/bin/activate

# Activate (Windows)
genrec-env\Scripts\activate
```

**Option B: Using conda**
```bash
# Create conda environment
conda create -n genrec python=3.10

# Activate
conda activate genrec
```

### Step 3: Install PyTorch (GPU)

**For CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CPU only:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Verify PyTorch:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

Expected output (GPU):
```
PyTorch: 2.1.0+cu118
CUDA Available: True
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

**For GPU-accelerated FAISS:**
```bash
pip uninstall faiss-cpu
pip install faiss-gpu
```

### Step 5: Install Git LFS

See [Git LFS Setup](#git-lfs-setup) section below.

### Step 6: Pull Data Files

```bash
git lfs pull
```

This will download all large data files (~20GB). May take 10-30 minutes depending on internet speed.

### Step 7: Install Jupyter

```bash
pip install jupyter ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

### Step 8: Launch Jupyter

```bash
jupyter notebook
```

Navigate to `notebooks/` and start with `01_phase3_semantic_retrieval.ipynb`.

---

## Google Colab Setup

### Step 1: Open Notebook in Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. **File → Open Notebook → GitHub**
3. Enter: `https://github.com/yourusername/GenRec`
4. Select a notebook (e.g., `notebooks/01_phase3_semantic_retrieval.ipynb`)

### Step 2: Enable GPU Runtime

1. **Runtime → Change runtime type**
2. **Hardware accelerator:** GPU
3. **GPU type:** A100 (for Phase 6) or T4 (for Phases 3-5)
4. Click **Save**

### Step 3: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 4: Install Dependencies

```python
!pip install -q transformers sentence-transformers faiss-cpu peft accelerate bitsandbytes trl
```

### Step 5: Clone Repository (if needed)

```python
!git clone https://github.com/yourusername/GenRec.git
%cd GenRec
```

### Step 6: Update File Paths

Replace local paths with Google Drive paths:

```python
# Original (local)
data_path = "../data/raw/df10_user_history.parquet"

# Updated (Colab)
data_path = "/content/drive/MyDrive/GenRec/data/raw/df10_user_history.parquet"
```

**Tip:** Upload data files to your Google Drive for persistent storage.

### Step 7: Run Notebook

Execute cells sequentially. Use **Runtime → Run all** for end-to-end execution.

---

## Git LFS Setup

Git Large File Storage (LFS) is required to download data files.

### Installation

**Ubuntu/Debian:**
```bash
sudo apt-get install git-lfs
git lfs install
```

**macOS (Homebrew):**
```bash
brew install git-lfs
git lfs install
```

**Windows:**
1. Download installer from [git-lfs.github.com](https://git-lfs.github.com/)
2. Run installer
3. Open Git Bash:
   ```bash
   git lfs install
   ```

### Verify Installation

```bash
git lfs version
```

Expected output: `git-lfs/3.4.0 (GitHub; ...)`

### Pull LFS Files

After cloning the repository:

```bash
cd GenRec
git lfs pull
```

**Progress Tracking:**
```
Downloading data/raw/df10_user_history.parquet (1.3 GB)
Downloading data/raw/grouped_reviews.parquet (6.9 GB)
Downloading data/processed/faiss_item_index.index (4.5 GB)
...
```

### Troubleshooting LFS

**Issue:** "This repository is over its data quota"

**Solution:** LFS bandwidth limits may apply on free GitHub accounts. Options:
1. Wait for quota reset (monthly)
2. Upgrade to GitHub Pro
3. Download files from alternative source (Google Drive link in README)

**Issue:** "git-lfs smudge filter failed"

**Solution:**
```bash
git lfs install --force
git lfs pull
```

---

## GPU Configuration

### Check CUDA Installation

```bash
nvidia-smi
```

Expected output:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.05   Driver Version: 535.104.05   CUDA Version: 12.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA A100-SXM...  On   | 00000000:00:04.0 Off |                    0 |
| N/A   30C    P0    45W / 400W |      0MiB / 40960MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

### Install CUDA Toolkit (if needed)

**Ubuntu:**
```bash
# CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda-11-8
```

**Verify:**
```bash
nvcc --version
```

### Configure PyTorch for Multi-GPU

```python
import torch

# Check available GPUs
print(f"Available GPUs: {torch.cuda.device_count()}")

# Set default GPU
torch.cuda.set_device(0)

# Enable cuDNN autotuner
torch.backends.cudnn.benchmark = True
```

---

## Troubleshooting

### Issue 1: Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```

**Solutions:**

**A. Reduce batch size**
```python
# Original
batch_size = 512

# Reduced
batch_size = 128
```

**B. Enable gradient checkpointing**
```python
model.gradient_checkpointing_enable()
```

**C. Use mixed precision**
```python
from torch.cuda.amp import autocast

with autocast():
    output = model(input)
```

**D. Clear GPU cache**
```python
import torch
torch.cuda.empty_cache()
```

---

### Issue 2: FAISS Index Not Found

**Error:**
```
FileNotFoundError: data/processed/faiss_item_index.index not found
```

**Solution:**

**Option 1:** Ensure Git LFS pulled correctly
```bash
git lfs pull
ls -lh data/processed/faiss_item_index.index
```

**Option 2:** Rebuild index from Phase 3 notebook
```bash
jupyter notebook notebooks/01_phase3_semantic_retrieval.ipynb
# Run cells up to "Save FAISS Index"
```

---

### Issue 3: ModuleNotFoundError

**Error:**
```
ModuleNotFoundError: No module named 'sentence_transformers'
```

**Solution:**
```bash
pip install sentence-transformers
```

Or reinstall all dependencies:
```bash
pip install -r requirements.txt --force-reinstall
```

---

### Issue 4: Notebook Kernel Crashes

**Symptoms:** Kernel dies during large data loading

**Solutions:**

**A. Increase system swap**
```bash
# Linux
sudo fallocate -l 32G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

**B. Load data in chunks**
```python
# Instead of
df = pd.read_parquet("large_file.parquet")

# Use
df = pd.read_parquet("large_file.parquet", columns=["col1", "col2"])
```

**C. Restart kernel and clear outputs**
- **Kernel → Restart & Clear Output**

---

### Issue 5: Slow FAISS Search

**Symptom:** Retrieval takes > 1 second per query

**Solutions:**

**A. Switch to GPU-FAISS**
```bash
pip install faiss-gpu
```

**B. Use IVF index for large datasets**
```python
# Replace IndexFlatIP with IndexIVFFlat
quantizer = faiss.IndexFlatIP(dim)
index = faiss.IndexIVFFlat(quantizer, dim, nlist=100)
```

**C. Optimize nprobe parameter**
```python
index.nprobe = 10  # Trade-off: higher = slower but more accurate
```

---

## Verification

### Test 1: Import All Libraries

```python
import torch
import transformers
import sentence_transformers
import faiss
import pandas as pd
import numpy as np
from peft import LoraConfig
from accelerate import Accelerator
import bitsandbytes
from trl import PPOTrainer

print("All imports successful!")
```

### Test 2: GPU Availability

```python
import torch

assert torch.cuda.is_available(), "CUDA not available!"
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

### Test 3: Load Sample Data

```python
import pandas as pd

df = pd.read_parquet("data/raw/df10_user_history.parquet", nrows=100)
print(f"Loaded {len(df)} rows")
print(df.head())
```

### Test 4: Load FAISS Index

```python
import faiss

index = faiss.read_index("data/processed/faiss_item_index.index")
print(f"Index size: {index.ntotal} vectors")
print(f"Index dimension: {index.d}")
```

### Test 5: Load Sentence-BERT

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-mpnet-base-v2')
embedding = model.encode(["test sentence"])
print(f"Embedding shape: {embedding.shape}")
```

---

## Next Steps

After successful setup:

1. **Start with Phase 3:**
   ```bash
   jupyter notebook notebooks/01_phase3_semantic_retrieval.ipynb
   ```

2. **Read Documentation:**
   - [Architecture Overview](docs/architecture.md)
   - [Performance Metrics](docs/metrics.md)
   - [Notebook Guide](notebooks/README.md)

3. **Experiment:**
   - Try different retrieval parameters
   - Test custom queries
   - Visualize embeddings

---

## Getting Help

**Issues with setup?**
- Check [GitHub Issues](https://github.com/yourusername/GenRec/issues)
- Review [Troubleshooting](#troubleshooting) section
- Contact: your.email@example.com

**Documentation:**
- [Main README](README.md)
- [Data Documentation](data/README.md)
- [Notebooks Guide](notebooks/README.md)

---

**Last Updated:** 2024
**Maintained By:** GenRec Team
