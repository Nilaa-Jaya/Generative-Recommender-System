# GenRec System Architecture

This document provides a comprehensive technical overview of the GenRec (Generative Retrieval-Augmented Recommender System) architecture.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Phase Breakdown](#phase-breakdown)
4. [Data Flow](#data-flow)
5. [Model Architecture](#model-architecture)
6. [Technical Design Decisions](#technical-design-decisions)
7. [Scalability Considerations](#scalability-considerations)

---

## System Overview

GenRec is a **hybrid recommender system** that effectively combines:
- **Collaborative Filtering**: User-item interaction patterns
- **Content-Based Filtering**: Semantic understanding of product descriptions
- **Generative AI**: Natural language explanations powered by LLMs

The system operates in **4 sequential phases**, each building on the outputs of the previous phase:

1. **Phase 3**: Semantic Retrieval (FAISS + Sentence-BERT)
2. **Phase 4**: Generative Layer (LLaMA2 + LoRA)
3. **Phase 5**: Personalization (User Clustering + DeepMF)
4. **Phase 6**: Fine-tuning & RLHF (QLoRA + Custom Reward Model)

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            GENREC ARCHITECTURE                                │
└─────────────────────────────────────────────────────────────────────────────┘

                                 USER QUERY
                                     │
                                     ▼
        ┌────────────────────────────────────────────────────┐
        │         PHASE 3: SEMANTIC RETRIEVAL                │
        │  ┌──────────────────────────────────────────────┐  │
        │  │ Sentence-BERT Encoder                        │  │
        │  │ (all-mpnet-base-v2, 768-dim)                 │  │
        │  └──────────────────┬───────────────────────────┘  │
        │                     │                               │
        │                     ▼                               │
        │  ┌──────────────────────────────────────────────┐  │
        │  │ FAISS IndexFlatIP                            │  │
        │  │ (1.57M vectors, Inner Product Search)        │  │
        │  └──────────────────┬───────────────────────────┘  │
        │                     │                               │
        │                     ▼                               │
        │  ┌──────────────────────────────────────────────┐  │
        │  │ Diversity Re-ranking                         │  │
        │  │ - Cosine similarity > 0.95 removal           │  │
        │  │ - Diversity score calculation                │  │
        │  └──────────────────┬───────────────────────────┘  │
        │                     │                               │
        │                     ▼                               │
        │           Top-K Diverse Items                       │
        └────────────────────┬────────────────────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────────────────┐
        │       PHASE 5: USER PERSONALIZATION                │
        │  ┌──────────────────────────────────────────────┐  │
        │  │ User Profile Builder                         │  │
        │  │ - Aggregate user history                     │  │
        │  │ - Extract keywords (TF-IDF)                  │  │
        │  │ - Compute statistics (avg_rating, count)     │  │
        │  └──────────────────┬───────────────────────────┘  │
        │                     │                               │
        │                     ▼                               │
        │  ┌──────────────────────────────────────────────┐  │
        │  │ DeepMF Model (User Embeddings)               │  │
        │  │ - 64-dim embeddings                          │  │
        │  │ - Trained on 6.7M interactions               │  │
        │  └──────────────────┬───────────────────────────┘  │
        │                     │                               │
        │                     ▼                               │
        │  ┌──────────────────────────────────────────────┐  │
        │  │ K-Means Clustering (k=20)                    │  │
        │  │ - Persona labels via TF-IDF                  │  │
        │  └──────────────────┬───────────────────────────┘  │
        │                     │                               │
        │                     ▼                               │
        │             User Persona String                     │
        └────────────────────┬────────────────────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────────────────┐
        │     PHASE 4 & 6: GENERATIVE LAYER + RLHF          │
        │  ┌──────────────────────────────────────────────┐  │
        │  │ Prompt Builder                               │  │
        │  │ - User persona                               │  │
        │  │ - Retrieved items                            │  │
        │  │ - Product metadata                           │  │
        │  └──────────────────┬───────────────────────────┘  │
        │                     │                               │
        │                     ▼                               │
        │  ┌──────────────────────────────────────────────┐  │
        │  │ LLaMA2-7B / Phi-2 + QLoRA                    │  │
        │  │ - Fine-tuned on 1.1M pairs                   │  │
        │  │ - LoRA rank=16, alpha=32                     │  │
        │  └──────────────────┬───────────────────────────┘  │
        │                     │                               │
        │                     ▼                               │
        │  ┌──────────────────────────────────────────────┐  │
        │  │ RLHF Reward Model                            │  │
        │  │ - Personalization score                      │  │
        │  │ - Sentiment alignment                        │  │
        │  │ - Length penalty                             │  │
        │  │ - Diversity bonus                            │  │
        │  └──────────────────┬───────────────────────────┘  │
        │                     │                               │
        │                     ▼                               │
        │         Personalized Recommendation                 │
        └────────────────────┬────────────────────────────────┘
                             │
                             ▼
                    FINAL OUTPUT
          "You might enjoy [Product] because
           it matches your interest in [X] and
           has excellent reviews for [Y]."
```

---

## Phase Breakdown

### Phase 3: Semantic Retrieval

**Goal:** Build a fast, semantic search engine over 1.57M products.

#### Components

**1. Sentence Encoder**
- **Model:** `sentence-transformers/all-mpnet-base-v2`
- **Architecture:** MPNet (Masked and Permuted Pre-training)
- **Output Dimension:** 768
- **Why MPNet?**
  - Superior performance on semantic textual similarity tasks
  - Pre-trained on 1B+ sentence pairs (NLI + STS datasets)
  - Balanced trade-off between speed and accuracy

**2. FAISS Index**
- **Index Type:** `IndexFlatIP` (Inner Product)
- **Why Flat?** Guarantees 100% recall (exhaustive search)
- **Why Inner Product?** Equivalent to cosine similarity for L2-normalized vectors
- **Index Size:** 4.5GB (1.57M × 768 × 4 bytes)
- **Search Complexity:** O(n × d) where n = index size, d = dimension

**3. Diversity Re-ranking**
- **Step 1:** FAISS retrieves top-30 candidates
- **Step 2:** Compute pairwise cosine similarity matrix (30×30)
- **Step 3:** Remove near-duplicates (similarity > 0.95 threshold)
- **Step 4:** Score remaining items by average cosine distance
- **Step 5:** Return top-10 most diverse items

**Pipeline:**
```
Query Text
   → Sentence-BERT Encode (768-dim)
   → FAISS Search (top-30)
   → Diversity Filter (remove duplicates)
   → Diversity Score (rank by distance)
   → Top-10 Diverse Items
```

**Latency Breakdown:**
- Encoding: ~5ms
- FAISS search: ~30ms
- Diversity re-ranking: ~8ms
- **Total: ~43ms**

---

### Phase 4: Generative Layer

**Goal:** Add natural language explanations to retrieved items.

#### Components

**1. Base Model: LLaMA2-7B**
- **Parameters:** 7 billion
- **Architecture:** Decoder-only transformer (Llama architecture)
- **Context Window:** 4096 tokens
- **Pre-training:** 2 trillion tokens (web text, books, code)

**2. LoRA Adapter**
- **Method:** Low-Rank Adaptation
- **Target Modules:** Query/Key/Value projections in attention layers
- **Rank (r):** 16 (reduces 4096×4096 matrix to 4096×16 × 16×4096)
- **Alpha:** 32 (scaling factor)
- **Trainable Parameters:** ~0.2% of base model
- **Why LoRA?** Enables fine-tuning on single GPU without catastrophic forgetting

**3. Prompt Template**
```
Instruction: You are a recommendation assistant.
User: [user_query]
Item: [item_title]
Metadata: [category, rating]
Task: Generate a one-sentence recommendation explaining why this item matches the user's needs.

Response:
```

**4. Generation Configuration**
- **Max New Tokens:** 60
- **Temperature:** 0.7 (balanced creativity/coherence)
- **Top-p (Nucleus Sampling):** 0.9
- **Repetition Penalty:** 1.2
- **Stop Tokens:** ["\n", ".", "!"]

**Output Example:**
```
Input: "Looking for a durable lightweight iPad case"
Output: "You might enjoy this case because it's made from
         military-grade materials while weighing only 4 ounces,
         perfect for portability."
```

---

### Phase 5: User Personalization

**Goal:** Build user-centric recommendations by modeling preferences and personas.

#### Components

**1. User Profiling**
```python
UserProfile = {
    "activity": {
        "n_reviews": int,
        "n_unique_items": int,
        "review_frequency": float
    },
    "preferences": {
        "avg_rating": float,
        "rating_std": float,
        "favorite_categories": List[str]
    },
    "language": {
        "top_keywords": List[str],
        "avg_review_length": float,
        "sentiment": float
    }
}
```

**2. DeepMF Model (Deep Matrix Factorization)**

Architecture:
```
Input: (user_id, item_id)
   ↓
User Embedding (64-dim)  +  Item Embedding (64-dim)
   ↓
Concatenate → [128-dim vector]
   ↓
MLP Decoder:
   - Linear(128 → 64) + ReLU
   - Linear(64 → 32) + ReLU
   - Linear(32 → 1) → Rating Prediction
   ↓
Output: Predicted Rating (1-5)
```

**Training:**
- **Dataset:** 6.7M user-item-rating triplets
- **Loss:** Mean Squared Error (MSE)
- **Optimizer:** Adam (lr=1e-3)
- **Batch Size:** 1024
- **Epochs:** 5
- **Val Split:** 80/20 train/val

**User Embedding Extraction:**
After training, extract the 64-dim user embedding layer:
```python
user_vector = model.user_embedding(user_id).detach().cpu().numpy()
```

**3. K-Means Clustering**
- **Algorithm:** K-Means with K-Means++ initialization
- **Features:** 64-dim user embeddings
- **K (clusters):** 20 (optimized via elbow method)
- **Distance Metric:** Euclidean distance

**4. Persona Label Generation**
For each cluster:
1. Aggregate all review texts from users in cluster
2. Compute TF-IDF scores
3. Extract top-5 keywords
4. Format: "interested in [keyword1], [keyword2], ..."

**Example Personas:**
- Cluster 0: "interested in electronics, phone, charger, cable, power"
- Cluster 7: "interested in books, reading, novel, story, author"
- Cluster 14: "interested in outdoor, camping, hiking, gear, backpack"

---

### Phase 6: Fine-tuning & RLHF

**Goal:** Align the generative model with human preferences using RLHF.

#### Components

**1. Training Data Creation**

Generate 100K+ instruction-style pairs:
```python
def create_training_example(user, item, rating):
    # Input
    input_text = f"""
    Instruction: Generate a personalized recommendation.
    User: {user.persona_string}
    Item: {item.title} ({item.category})
    Rating: {rating}/5
    """

    # Output (rating-aware template)
    if rating >= 4.5:
        output_text = f"Perfect match for your preferences! Based on your
                        interest in {user.top_keywords}, this {item.category}
                        offers {item.key_features}."
    elif rating >= 3.5:
        output_text = f"Based on your history, this item aligns well with
                        your typical {item.category} purchases."
    else:
        output_text = f"This might not meet your usual expectations for
                        {item.category}."

    return {"input": input_text, "output": output_text}
```

**2. QLoRA (Quantized LoRA)**

Why QLoRA over standard LoRA?
- **4-bit quantization** reduces memory from 28GB → 8GB for LLaMA2-7B
- Enables training on single A100 (40GB) or consumer GPUs
- Maintains full-precision performance

Configuration:
```python
LoraConfig(
    r=16,                          # Rank
    lora_alpha=32,                 # Scaling factor
    target_modules=[               # Attention layers to adapt
        "q_proj", "k_proj",
        "v_proj", "o_proj"
    ],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",     # Normal Float 4-bit
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)
```

**3. RLHF Reward Model**

Custom reward function scoring 4 dimensions:

```python
def compute_reward(generated_text, user_rating, user_persona):
    score = 0.0

    # 1. Length reward (15-60 words optimal)
    word_count = len(generated_text.split())
    if 15 <= word_count <= 60:
        score += 0.3
    else:
        score -= 0.2

    # 2. Personalization bonus (mentions user context)
    personalization_keywords = ["your", "you", "based on", "preference"]
    if any(kw in generated_text.lower() for kw in personalization_keywords):
        score += 0.4

    # 3. Sentiment alignment (positive for high ratings)
    if user_rating >= 4:
        positive_words = ["excellent", "great", "perfect", "love"]
        if any(word in generated_text.lower() for word in positive_words):
            score += 0.2

    # 4. Diversity penalty (avoid generic phrases)
    generic_phrases = ["this product is good", "you might like this"]
    if any(phrase in generated_text.lower() for phrase in generic_phrases):
        score -= 0.3

    return np.clip(score, -1.0, 1.0)  # Normalize to [-1, 1]
```

**4. PPO Training Loop (Simulated)**

```python
for episode in range(50):
    # Sample batch of prompts
    batch = sample_training_batch(size=32)

    # Generate responses
    responses = model.generate(batch["input"])

    # Compute rewards
    rewards = [compute_reward(resp, batch["rating"][i], batch["persona"][i])
               for i, resp in enumerate(responses)]

    # Update policy with PPO
    loss = ppo_loss(old_logprobs, new_logprobs, rewards, advantages)
    optimizer.step()
```

---

## Data Flow

### End-to-End Recommendation Flow

```
1. USER INPUT
   ├─ Query: "Best noise-canceling headphones for commute"
   └─ User ID: U123456789

2. USER LOOKUP (Phase 5)
   ├─ Load user embedding (64-dim)
   ├─ Find cluster assignment → Cluster 3
   └─ Retrieve persona: "interested in electronics, audio, music, quality"

3. SEMANTIC RETRIEVAL (Phase 3)
   ├─ Encode query → 768-dim vector
   ├─ FAISS search → Top-30 headphones
   ├─ Diversity filter → Remove 8 near-duplicates
   └─ Output: 10 diverse headphones

4. CONTEXT PREPARATION
   ├─ User persona: "interested in electronics, audio..."
   ├─ Item 1: Sony WH-1000XM5 (avg_rating=4.7, category=Electronics)
   ├─ Item 2: Bose QC45 (avg_rating=4.6, category=Electronics)
   └─ ...

5. GENERATION (Phase 4 & 6)
   ├─ Build prompt:
   │  Instruction: Generate personalized recommendation
   │  User: Cluster 3 - interested in electronics, audio, music
   │  Item: Sony WH-1000XM5 (Noise-Canceling Headphones)
   │
   ├─ LLaMA2 + LoRA generates:
   │  "Perfect for your daily commute! These headphones offer
   │   industry-leading noise cancellation and 30-hour battery
   │   life, ideal for long trips and high-quality audio you
   │   appreciate."
   │
   └─ RLHF reward score: 0.85 (high personalization + good length)

6. OUTPUT
   └─ Return top-10 recommendations with explanations
```

---

## Model Architecture

### Sentence-BERT (all-mpnet-base-v2)

```
Input: "Product description text"
   ↓
Tokenizer (WordPiece)
   ↓
MPNet Encoder (12 layers, 768 hidden, 12 heads)
   ├─ Masked Language Modeling objective
   ├─ Permuted Language Modeling objective
   └─ Pre-trained on 1B+ sentence pairs
   ↓
Mean Pooling (across all tokens)
   ↓
L2 Normalization
   ↓
Output: 768-dim unit vector
```

### DeepMF (Deep Matrix Factorization)

```
User ID (int)          Item ID (int)
   ↓                      ↓
Embedding(64)          Embedding(64)
   └────────┬─────────────┘
            │
         Concat(128)
            ↓
      Linear(128→64)
            ↓
         ReLU()
            ↓
      Linear(64→32)
            ↓
         ReLU()
            ↓
      Linear(32→1)
            ↓
    Sigmoid() × 4 + 1  (scale to 1-5)
            ↓
      Rating Prediction
```

### LLaMA2-7B + QLoRA

```
Input Tokens (4096 max)
   ↓
Embedding Layer (4096 hidden dim)
   ↓
32× Transformer Decoder Blocks:
   ├─ Multi-Head Attention (32 heads)
   │  ├─ Q, K, V projections ← LoRA adapters here!
   │  └─ Attention(Q, K, V)
   ├─ Feed-Forward Network
   │  ├─ Linear(4096 → 11008) ← LoRA here
   │  ├─ SiLU activation
   │  └─ Linear(11008 → 4096) ← LoRA here
   └─ Layer Norm + Residual
   ↓
Output Layer (vocab size ~32K)
   ↓
Softmax → Token Probabilities
   ↓
Autoregressive Sampling
   ↓
Generated Text
```

**LoRA Injection:**
```
Original:  y = W × x       (W is 4096×4096)
With LoRA: y = (W + ΔW) × x
           where ΔW = B × A  (B is 4096×16, A is 16×4096)
```
Only B and A are trained → **6.5M parameters** vs 7B base model.

---

## Technical Design Decisions

### Why Sentence-BERT over OpenAI Embeddings?
| Criterion | Sentence-BERT | OpenAI text-embedding-ada-002 |
|-----------|---------------|-------------------------------|
| **Cost** | Free (open-source) | ~$0.10 per 1M tokens |
| **Latency** | ~5ms (local GPU) | ~50-100ms (API call) |
| **Control** | Full control | Black-box |
| **Offline Use** | ✅ Yes | ❌ Requires internet |
| **Customization** | Can fine-tune | No fine-tuning |

**Decision:** Sentence-BERT for cost, speed, and control.

---

### Why FAISS Flat over HNSW/IVF?
| Index Type | Build Time | Search Time | Recall | Memory |
|------------|------------|-------------|--------|--------|
| **Flat** | Fast | O(n×d) | 100% | High |
| **HNSW** | Slow | O(log n) | ~95% | Very High |
| **IVF** | Medium | O(√n×d) | ~90% | Medium |

**Decision:** Flat for guaranteed 100% recall (critical for small-medium datasets < 10M items).

---

### Why DeepMF over Standard Matrix Factorization?
- **Non-linear interactions:** MLP captures complex user-item patterns
- **Better cold-start:** Embeddings generalize to unseen users
- **Flexible:** Can add side features (category, price) easily

---

### Why PPO over DPO for RLHF?
| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **PPO** | Continuous rewards, stable | Slower, more complex | Custom reward functions |
| **DPO** | Simpler, faster | Needs pairwise preferences | Human-labeled data |

**Decision:** PPO because we had **scalar rewards** (personalization score) not pairwise preferences.

---

## Scalability Considerations

### Current Scale
- **Items:** 1.57M products
- **Users:** 728K users
- **Interactions:** 6.7M reviews
- **FAISS Index:** 4.5GB
- **Latency:** ~43ms per query

### Scaling to 10M+ Items

**1. FAISS Optimization**
- Switch to `IndexIVFFlat` (Inverted File Index)
- Use GPU-FAISS for 10-50x speedup
- Shard index across multiple servers

**2. Embedding Caching**
- Cache popular query embeddings (80/20 rule)
- Use Redis for distributed caching
- Precompute embeddings for trending items

**3. Model Serving**
- Deploy LLaMA2 with TensorRT or vLLM
- Use model quantization (INT8) for faster inference
- Batch generation requests (dynamic batching)

**4. Distributed Retrieval**
- Partition FAISS index by category
- Parallel search across shards
- Merge and re-rank results

---

## Future Enhancements

1. **Multi-modal Embeddings**: Add image + text fusion
2. **Session-based Recommendations**: RNN/Transformer for sequence modeling
3. **Online Learning**: Incremental updates from user feedback
4. **A/B Testing Framework**: Bayesian optimization for hyperparameters
5. **Explainability**: Attention visualization for transparency

---

**Document Version:** 1.0
**Last Updated:** 2024
**Maintained By:** GenRec Team
