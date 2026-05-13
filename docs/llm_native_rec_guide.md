# Building a Strong LLM-Native Recommendation Model

A synthesis of best practices from PLUM, OneRec-Think, GR2, and the broader generative recommendation literature. Each section covers the design choice, why it matters, and what to ablate.

---

## 1. Data

### 1.1 What to collect

| Signal | Value | Used in |
|---|---|---|
| Interaction sequences (clicks, watches, purchases) | Core training signal | All |
| Ratings / scores | Engagement auxiliary loss | PLUM, ours |
| Watch time / dwell time | Stronger engagement signal than binary | PLUM |
| Review text | Rich semantic grounding for CPT | Amazon-style |
| Item metadata (title, category, description) | Item embedding + verbalization | All |
| Co-occurrence / CF signals | SID quality | PLUM SIDv2 |
| Timestamps | Temporal ordering, recency weighting | All |

### 1.2 Sequence construction

- **Sort by timestamp** — recency matters; never shuffle within a user's history
- **Positive-only filtering** — for ratings data, threshold at ≥3.5/5 to keep positives; do not include dislikes in the context window
- **Max history length** — 20–50 interactions covers most users; longer contexts hurt training speed more than they help
- **Minimum history** — drop users with <5 interactions; they lack enough signal for meaningful sequence modeling
- **Leave-one-out split** — industry standard; val = second-to-last, test = last item

### 1.3 Verbalization

The verbalization strategy is an under-studied ablation axis. Key decisions:

```
# Full verbalization (most information)
"User history:
1. The Dark Knight [Action] — 5/5
   'One of the best films ever made.'
2. Inception [Sci-Fi] — 4/5
Predict the next item.
Next:"

# Minimal (fastest, good baseline)
"History: The Dark Knight → Inception → The Prestige
Next:"
```

**What matters:** Including ratings helps when engagement is a strong signal. Review text helps on Amazon-style data but adds noise on pure ratings data (MovieLens). Genre/category consistently helps. Ablate all three independently.

---

## 2. Item Representation (Embeddings / Tokenization)

This is the most architecturally important decision.

### 2.1 Text tokens (baseline)

Items are their title text, tokenized by the existing LLM vocabulary. No vocab extension needed.

- **Pro:** Zero setup, works out of the box, title carries semantic meaning
- **Con:** Title tokenization is inconsistent (e.g. "Star Wars" → 2 tokens, "The Shawshank Redemption" → 5 tokens), generative decoding is slow (must decode the entire title string)

### 2.2 Semantic IDs (SID) — recommended

Items are assigned discrete codebook codes via RQ-VAE, then mapped to new special tokens in the LLM vocabulary.

```
item_id → RQ-VAE encode → (42, 17, 5) → <item_L1_42><item_L2_17><item_L3_5>
```

**Why this works:**
- Fixed-length representation (3 tokens per item regardless of title length)
- Codebook structure encodes semantic similarity — items with the same L1 code are semantically related
- Constrained decoding becomes tractable (beam search over codebook indices)
- Scales to millions of items without inflating vocabulary proportionally

**What to feed into RQ-VAE:**

| Input | What it captures | When to use |
|---|---|---|
| Text embeddings only | Semantic / content similarity | Cold-start heavy catalogs |
| CF embeddings only (BPR/MF) | Collaborative similarity | Dense interaction data |
| Text + CF blend | Both | Default; best of both worlds |
| Text + engagement (CTR, watch rate) | Popularity-aware similarity | When engagement signal is strong |

**Codebook design:**
- 3 levels × 8192 codes = 24,576 new tokens (OneRec-Think scale)
- 3 levels × 256 codes = 768 tokens (good for research / smaller catalogs)
- Larger codebooks → more unique codes, fewer collisions, but bigger vocab

### 2.3 Vocabulary extension

When adding item tokens:
1. Initialize new token embeddings from the mean of existing embeddings (not random) — this warms up the LLM's embedding space
2. Resize both input embeddings and LM head together (`resize_token_embeddings`)
3. Only train new token embeddings during CPT; unfreeze all during SFT

---

## 3. Model

### 3.1 Base LLM selection

| Model | Size | Notes |
|---|---|---|
| Qwen2.5-7B-Instruct | 7B | Strong chat template, good multilingual, default choice |
| Llama-3.1-8B-Instruct | 8B | Strong English, widely benchmarked |
| Gemma-2-9B | 9B | Strong reasoning, heavier |
| Qwen2.5-3B | 3B | Fast iteration / ablations on Colab L4 |

**Rule of thumb:** Larger base = better zero-shot, but LoRA fine-tuning closes the gap quickly. Start with 3B for ablations, validate on 7B before submitting.

### 3.2 LoRA configuration

```yaml
r: 16              # rank — higher = more capacity, more params
lora_alpha: 32     # scaling = alpha/r; keep alpha = 2×r
target_modules:    # which projections to adapt
  - q_proj, k_proj, v_proj   # attention
  - o_proj                    # attention output
  - gate_proj, up_proj, down_proj  # FFN (important for rec tasks)
lora_dropout: 0.05
```

**What to ablate:**
- `r=8` vs `r=16` vs `r=64` — rec tasks are often well-served by low rank
- Attention-only vs attention+FFN — FFN matters more than expected for item token prediction
- Full fine-tune vs LoRA — for <7B models on rec data, LoRA usually matches full FT at 5% of the params

### 3.3 Auxiliary heads

**Engagement head** — a linear layer on the last hidden state that predicts rating/CTR:
```python
engage_loss = MSE(linear(h_last) / 5.0, rating / 5.0) * λ
```
Adds ~0.001% parameters. Empirically helps when engagement variance is high (Amazon reviews) and hurts slightly on sparse ratings data (MovieLens 1M).

---

## 4. Training Stages

The multi-stage pipeline is the consensus across PLUM, OneRec-Think, and GR2.

```
Stage 1: CPT          Stage 2: SFT          Stage 3: RL/DPO
(domain adaptation)   (task fine-tuning)    (preference alignment)
       ↓                     ↓                      ↓
Learn item tokens     Learn next-item       Prefer good items
+ behavior patterns   prediction format     over bad ones
```

### 4.1 CPT (Continued Pre-Training)

**Goal:** Teach the LLM about the item vocabulary and user behavior patterns before asking it to predict.

- **Format:** Plain sequences of verbalized interactions, no instruction template
- **Data:** All available user histories + item text (titles, descriptions, reviews)
- **Loss:** Standard causal LM loss over all tokens including context
- **LR:** 5e-5, lower than SFT — you're adapting, not specializing
- **When to skip:** If your dataset is small (<1M interactions) and you're time-constrained, SFT-only often works surprisingly well

**CPT-specific tips:**
- Pack multiple short user histories into one training sequence to maximize GPU utilization
- Shuffle at the user level, not the token level — preserve within-user temporal order
- Include item metadata documents ("Inception is a 2010 sci-fi thriller directed by...") to ground item tokens in text meaning

### 4.2 SFT (Supervised Fine-Tuning)

**Goal:** Teach the model the exact task format — given a user history, predict the next item.

- **Format:** Instruction template with chat format; mask context tokens from loss (only supervise the response)
- **Data:** (context = history[:i], target = history[i]) pairs for all i
- **Loss:** Causal LM loss on response tokens only
- **LR:** 2e-4 with LoRA (higher than CPT — task specialization needs more signal)
- **Epochs:** 3–5; watch for overfitting on small datasets

**Loss masking is critical:** Without it, the model optimizes for predicting the user history (which it already knows) rather than the next item. Always mask `input_len` tokens.

### 4.3 RL / Preference Alignment

**Goal:** Push the model to prefer items users actually engage with over irrelevant alternatives.

**DPO (vanilla — what we have):**
```
chosen: (context, item_user_liked)
rejected: (context, random_negative)
loss: -log σ(β · (log π_θ(chosen) - log π_ref(chosen) - log π_θ(rejected) + log π_ref(rejected)))
```

**DAPO (GR2 style — stronger):**
- Decoupled clip: separate clip thresholds for chosen vs. rejected
- Dynamic sampling: adaptively re-samples training pairs based on current model difficulty
- Verifiable reward: reward = 1 if target item is ranked in top-K by the model, else 0
- More stable than DPO on recommendation tasks where "preference" is noisy

**What to use as negatives for DPO:**
- Random items from catalog: easy but weak signal
- Hard negatives (items from the same category): much stronger, closer to what the model already scores highly
- Popularity-stratified negatives: mix popular + long-tail for balanced coverage

---

## 5. Sampling

### 5.1 Training negatives

The model never sees explicit negatives during LM training, but negative sampling matters for DPO/RLHF and evaluation.

| Strategy | Cost | Quality |
|---|---|---|
| Uniform random | Very low | Weak — most negatives are obviously irrelevant |
| In-category hard negatives | Low | Good — tests fine-grained discrimination |
| In-batch negatives | Zero extra cost | Good at scale — other items in the batch are negatives |
| BM25 / embedding retrieval | Medium | Best — surface the hardest false positives |

**Recommendation:** Start with uniform random to get a working baseline, then switch to in-category hard negatives for the DPO stage.

### 5.2 Rejection sampling for reasoning traces (GR2 / OneRec-Think)

To generate high-quality SFT data with reasoning:
1. Prompt a large teacher model (Qwen-72B, GPT-4o) with the user history
2. Ask it to reason step-by-step before predicting the next item
3. Run the student model on the same context
4. **Keep only traces where the student's prediction matches the target** (rejection sampling)
5. Use accepted traces as SFT data

This is expensive but produces much higher-quality supervision than template-based reasoning.

---

## 6. Reasoning

Chain-of-thought reasoning is the key differentiator of OneRec-Think and GR2 over PLUM.

### 6.1 Reasoning scaffold

**Training format:**
```
User history: [...]

Predict the next item.
<think>
This user has a strong preference for psychological thrillers rated 4+ stars.
Their recent watches skew toward complex narratives (Inception, Memento).
They have not watched any Christopher Nolan film from 2014 onward.
</think>
Next: Interstellar
```

**Why it helps:** Forces the model to explicitly surface the relevant preference dimensions before committing to an item. Especially valuable for users with diverse histories where a shallow pattern-match would pick the wrong genre.

### 6.2 Distillation pipeline

```
Teacher LLM (large)                     Student LLM (small, being trained)
      |                                           |
      | generate reasoning trace                  |
      | for (context → target)                    |
      ↓                                           |
 <think>...</think> + target item                 |
      |                                           |
      | rejection sampling:                       |
      | does student agree with target?    ───────┘
      |
      ↓
SFT dataset: (context, <think>trace</think> + target)
```

### 6.3 Reward functions for RL

For recommendation, rewards should be **verifiable** (not just preference-based):

| Reward | Description | Strength |
|---|---|---|
| Hit reward | 1 if target item in top-K, else 0 | Clean, sparse |
| NDCG reward | Continuous — higher rank = higher reward | Dense gradient |
| Engagement reward | Predicted engagement of generated item | Risk of reward hacking |
| Diversity penalty | Penalize if top-K is too homogeneous | Secondary objective |

---

## 7. Inference

### 7.1 Constrained decoding

For SID-based models, restrict beam search to valid item token sequences:

```
Beam search at L1: only tokens in {<item_L1_0>, ..., <item_L1_K>}
Beam search at L2: only tokens in {<item_L2_0>, ..., <item_L2_K>}
Beam search at L3: only tokens in {<item_L3_0>, ..., <item_L3_K>}
```

This guarantees every generated token sequence maps to a valid item and enables true top-K retrieval without scoring all candidates explicitly.

### 7.2 Log-prob scoring (what we currently use)

Score each candidate item by `log P(item tokens | context)` and rank. Works for any tokenization strategy but scales as O(N × forward_passes). Fine for evaluation with 100 candidates; too slow for production over 1M items.

### 7.3 Trade-offs

| Method | Speed | Requires | Use case |
|---|---|---|---|
| Constrained beam search | Fast (one forward pass → top-K) | Semantic IDs | Production retrieval |
| Log-prob scoring | Slow (N forward passes) | Any tokenization | Research evaluation |
| Embedding similarity | Very fast | Separate encoder | Dual-encoder baseline |

---

## 8. Evaluation

### 8.1 Standard offline metrics

Always report over the 100-candidate pool (99 random negatives + 1 target):

| Metric | What it measures |
|---|---|
| HR@1 | Did the model get it exactly right? |
| HR@10, HR@20 | Did the model surface the item in top-10/20? |
| NDCG@10 | Rank-weighted precision |
| MRR | Mean reciprocal rank — sensitive to very high ranks |

### 8.2 Beyond standard metrics

| Metric | Why it matters |
|---|---|
| Coverage@K | What fraction of the item catalog appears in top-K across all users? |
| Intra-list diversity | Average pairwise dissimilarity of items in top-K |
| Cold-start HR@10 | Subset metric on items with <10 interactions |
| Serendipity | Did the model surface items outside the user's obvious genre? |

### 8.3 Ablation checklist for a paper

- [ ] Text tokens vs. semantic IDs (item tokenization)
- [ ] CPT + SFT vs. SFT only (does CPT help?)
- [ ] LM loss only vs. + engagement loss (does engagement signal help?)
- [ ] Rating-included verbalization vs. title-only
- [ ] History length: 10 vs. 20 vs. 50 interactions
- [ ] LoRA rank: 8 vs. 16 vs. 64
- [ ] With vs. without reasoning scaffold (if implemented)
- [ ] CF-augmented SIDs vs. text-only SIDs
