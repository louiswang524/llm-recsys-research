# Building a Strong LLM-Native Recommendation Model

A synthesis of best practices from PLUM, OneRec-Think, GR2, and the broader generative recommendation literature. Each section covers the design choice, the underlying mechanism, and what to ablate.

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
- **Max history length** — 20–50 interactions covers most users; longer contexts hurt training speed more than they help. With SIDs at 3 tokens/item, 50 items = 150 context tokens; with title text it scales to 300–600 tokens depending on title length
- **Minimum history** — drop users with <5 interactions; they lack enough signal for meaningful sequence modeling
- **Leave-one-out split** — industry standard: `train = history[:-2]`, `val = history[:-1]` (last item is val target), `test = history[:]` (last item is test target). This ensures every user appears in all splits

### 1.3 Verbalization

Verbalization has two independent axes that must be considered together:
- **Context representation** — how the user's history is written in the prompt
- **Target representation** — how the next item is expressed in the prediction

These axes interact directly with your choice of item tokenization (Section 2).

#### The three verbalization modes

**Mode 1 — Text-only (baseline, no SIDs)**

Context and target are both plain text. No vocabulary extension needed.

```
User history:
1. The Dark Knight [Action] — 5/5
   "One of the best films ever made."
2. Inception [Sci-Fi] — 4/5
Predict the next item.
Next: Interstellar
```

Context items: title + metadata text.
Target: item title decoded as standard LLM vocabulary subword tokens.

Training objective: `L = -sum_t log P(title_token_t | context, title_token_{<t})` over the title string.

Limitation: generative decoding requires beam search over the full vocabulary at every step. For a title like "The Shawshank Redemption" this means 5+ decoding steps with |V|=32K candidates each. Inference is O(title_len × |V|) and there is no mechanism to guarantee the decoded string is a real item.

---

**Mode 2 — SID target (text context → discrete item prediction)**

Context stays human-readable text. Target switches to SID tokens. This is the standard SFT format for SID-based models.

```
User history:
1. The Dark Knight [Action] — 5/5
2. Inception [Sci-Fi] — 4/5
Predict the next item.
Next: <item_L1_15><item_L2_7><item_L3_33>
```

Training objective: `L = -sum_{l=1}^{3} log P(<item_Ll_k_l> | context, <item_L1..l-1>)`

This is a cross-modal mapping — given a natural language description of user taste, emit the correct 3-token discrete item code. The model must bridge text understanding and item vocabulary entirely through the SFT stage, which makes it harder without CPT pre-alignment.

---

**Mode 3 — Cross-modal CPT (text + SID context → SID target)**

Both context items and the target use SIDs. This is the CPT alignment stage from PLUM and OneRec-Think: the model learns that a text description co-occurs with specific SID tokens.

```
User history:
1. The Dark Knight [Action] <item_L1_3><item_L2_17><item_L3_5> — 5/5
2. Inception [Sci-Fi] <item_L1_8><item_L2_2><item_L3_19> — 4/5
Predict the next item.
Next: <item_L1_15><item_L2_7><item_L3_33>
```

Context items: title + metadata text **and** the item's SID tokens together.
Target: SID tokens only.

What the model learns: `"The Dark Knight" ↔ <item_L1_3><item_L2_17><item_L3_5>`. By seeing each item's text and its codes in the same context, the model builds aligned representations in a shared embedding space. Without this stage, SID tokens start as random vectors (or mean-initialized vectors — see Section 2.3) and the SFT stage must do alignment and task learning simultaneously, which is harder.

The CPT causal LM loss here is computed over all tokens including the SID tokens in context, not just the target SIDs. This trains the model to fluently predict SID codes given text context, and vice versa.

#### Mapping modes to training stages

| Stage | Mode | Loss supervision |
|---|---|---|
| CPT | Mode 3 (cross-modal) | All tokens (context + target SIDs) |
| SFT | Mode 2 (SID target) | Target SID tokens only (input masked) |
| DPO | Mode 2 (SID target) | Chosen/rejected differ in target SID sequence |
| Text-only baseline | Mode 1 | Target title tokens only (input masked) |

#### Context verbalization options (independent of mode)

```
# Full: title + genre + rating + review excerpt
"1. The Dark Knight [Action] — 5/5\n   'One of the best films ever made.'"

# No review (MovieLens has no reviews)
"1. The Dark Knight [Action] — 5/5"

# No rating (ablate whether rating helps)
"1. The Dark Knight [Action]"

# Minimal chain (fastest, lowest context overhead)
"The Dark Knight → Inception → The Prestige"
```

**What matters:** Ratings help when engagement variance is high (Amazon, σ ≈ 1.2/5). Review text helps on Amazon-style data, adds noise on pure ratings data (MovieLens). Genre/category consistently helps by providing categorical groupings that transfer to SID codebook structure. SID tokens in the context (Mode 3) add alignment signal but increase sequence length by ~3 tokens per history item.

#### Implementation note (formatter.py)

`InstructionFormatter` accepts an optional `item_tokenizer`. When provided, `_target_text()` calls `item_tokenizer.item_to_tokens(target_id)` and joins the results — the assistant response becomes the SID token string (Mode 2/3). Without `item_tokenizer`, the response is the item title (Mode 1).

```python
# Mode 2: SID target
formatter = InstructionFormatter(verbalizer, hf_tokenizer, stage="sft", item_tokenizer=sid_tokenizer)

# Mode 1: text target
formatter = InstructionFormatter(verbalizer, hf_tokenizer, stage="sft")
```

---

## 2. Item Representation (Embeddings / Tokenization)

This is the most architecturally consequential decision in the pipeline.

### 2.1 Text tokens (baseline)

Items are their title text, tokenized by the existing LLM vocabulary. No vocab extension needed.

- **Pro:** Zero setup, works out of the box, title carries semantic meaning
- **Con:** Inconsistent length ("Star Wars" = 2 tokens, "The Shawshank Redemption" = 5 tokens). Generative decoding is slow and unconstrained — can generate strings that are not valid items

### 2.2 Semantic IDs (SID) — recommended

Items are assigned discrete codebook codes via RQ-VAE, then mapped to new special tokens in the LLM vocabulary.

```
item_id → embed (sentence-transformer) → RQ-VAE encode → (42, 17, 5) → <item_L1_42><item_L2_17><item_L3_5>
```

**Why residual quantization:**

RQ-VAE (Residual Quantization VAE) applies K-means quantization hierarchically. At each level, the quantization residual from the previous level is quantized:

```
Level 1:  c_1 = argmin_k ||e - z_k^(1)||   residual: r_1 = e - z_{c1}^(1)
Level 2:  c_2 = argmin_k ||r_1 - z_k^(2)|| residual: r_2 = r_1 - z_{c2}^(2)
Level 3:  c_3 = argmin_k ||r_2 - z_k^(3)||
```

The 3-tuple `(c_1, c_2, c_3)` is the SID. Items sharing `c_1` are coarsely similar (same genre cluster); items sharing `(c_1, c_2)` are more similar (same sub-genre); items sharing all three codes are near-duplicates.

This hierarchical structure makes constrained beam search tractable: at decode time, fixing L1 reduces the L2 candidate set to items in that coarse cluster, and so on.

**What to feed into RQ-VAE:**

| Input | What it captures | When to use |
|---|---|---|
| Text embeddings only | Semantic / content similarity | Cold-start heavy catalogs |
| CF embeddings (BPR/MF) | Collaborative filtering similarity | Dense interaction data |
| Text + CF blend (concat or weighted sum) | Both | Default; best of both worlds |
| Text + engagement (CTR, watch rate) | Popularity-aware similarity | When engagement signal is strong |

**Codebook design:**

The vocabulary expansion is `n_levels × codebook_size` new tokens:
- 3 levels × 8192 codes = 24,576 new tokens (OneRec-Think scale; collision rate < 0.001% for 10M items)
- 3 levels × 256 codes = 768 tokens (good for research / smaller catalogs; collision rate ~1% for 100K items)

Collision = two distinct items assigned the same SID tuple. At 3 levels × 256 codes, the space is 256³ = 16M unique tuples — more than enough for any realistic catalog, but collisions can still occur because K-means doesn't guarantee unique assignments. Track collision rate as a data quality metric.

### 2.3 Vocabulary extension

When adding item tokens, the correct order of operations is:

```python
# 1. Load base model (vocab size = X)
model = AutoModelForCausalLM.from_pretrained(base_model_name)

# 2. Extend vocabulary (X → X+Y rows in embed_tokens and lm_head)
orig_vocab_size = model.config.vocab_size
model.resize_token_embeddings(orig_vocab_size + vocab_size_delta)

# 3. Mean-initialize new rows — avoids random-noise gradients in early training
with torch.no_grad():
    emb = model.get_input_embeddings()
    emb.weight[orig_vocab_size:] = emb.weight[:orig_vocab_size].mean(dim=0)
    if not model.config.tie_word_embeddings:
        lm_head = model.get_output_embeddings()
        lm_head.weight[orig_vocab_size:] = lm_head.weight[:orig_vocab_size].mean(dim=0)

# 4. Apply LoRA (only AFTER resize — LoRA wraps the already-extended matrices)
model = get_peft_model(model, lora_config)
```

**Why this order matters:**

- `resize_token_embeddings` after `get_peft_model` would attempt to resize a LoRA-wrapped matrix, which is not supported.
- Mean initialization after LoRA would overwrite LoRA adapter contributions in the embedding rows — initialize first, then wrap.
- New token rows (indices `X..X+Y-1`) are always trainable regardless of LoRA because they are not part of the frozen pretrained weight matrix — they are the newly appended rows, which have no LoRA A/B adapter and are updated directly by the optimizer.

**Why mean, not random:**

At initialization, SID tokens appear in training sequences and contribute to the attention computation. Random embeddings have norm ≈ `sqrt(hidden_dim)` / `sqrt(hidden_dim)` = 1 (if initialized from N(0, 1/hidden_dim)), but their direction is random relative to all existing representations. This produces large unexpected query-key dot products that corrupt attention patterns for the first N steps. Mean initialization places new tokens at the centroid of the embedding distribution — zero mean, moderate norm — which is the least-disruptive possible starting point.

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

LoRA decomposes each weight update as `ΔW = BA` where `B ∈ R^{d×r}`, `A ∈ R^{r×k}`, and `r << min(d, k)`. The trainable parameter count per module is `r × (d + k)` vs `d × k` for full fine-tuning.

```yaml
r: 16              # rank — higher = more capacity, more params
lora_alpha: 32     # effective lr scaling = alpha/r; keep alpha = 2×r so scale = 2.0
target_modules:    # which projections to adapt
  - q_proj, k_proj, v_proj   # attention (key for preference modeling)
  - o_proj                    # attention output projection
  - gate_proj, up_proj, down_proj  # FFN — critical for item token prediction
lora_dropout: 0.05
```

For a 7B model with hidden_dim=4096, FFN intermediate=11008:
- Attention LoRA (q/k/v/o, r=16): 4 × 16 × (4096 + 4096) = 524K params
- FFN LoRA (gate/up/down, r=16): 2 × 16 × (4096 + 11008) + 1 × 16 × (11008 + 4096) = 725K params
- Total LoRA: ~1.2M params per layer × 32 layers ≈ 40M params (≈0.5% of 7B)

**What to ablate:**
- `r=8` vs `r=16` vs `r=64` — rec tasks are often well-served by low rank; r=16 is a good default
- Attention-only vs attention+FFN — FFN modules matter more than expected for item token prediction because item tokens must be decoded from FFN outputs at the LM head
- Full fine-tune vs LoRA — for <7B models on rec data, LoRA usually matches full FT at 5% of the params

### 3.3 Auxiliary heads

**Engagement head** — a linear layer `W_e ∈ R^{hidden_dim×1}` on the last non-padding hidden state:

```python
h_last = hidden_states[-1][torch.arange(B), seq_lens]  # (B, hidden_dim)
rating_hat = engage_head(h_last).squeeze(-1)            # (B,)
engage_loss = MSE(rating_hat / 5.0, rating / 5.0) * lambda_engage
```

The head adds `hidden_dim` ≈ 4096 parameters (~0.00006% of 7B). The engagement signal back-propagates through the last hidden state into the LoRA adapters, giving the model a secondary supervision signal that correlates with item quality beyond rank position. Empirically useful when engagement variance is high (Amazon reviews, σ > 1.0/5); marginally helpful or neutral on MovieLens 1M (σ ≈ 0.9/5).

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

- **Format:** Plain sequences of verbalized interactions (Mode 3 if SIDs are used), no instruction template
- **Data:** All available user histories + item text (titles, descriptions, reviews)
- **Loss:** Standard causal LM loss over all tokens including context. For Mode 3, this means the model predicts SID tokens in context as well as the target — full sequence supervision
- **LR:** 5e-5, lower than SFT because you are adapting pretrained representations, not learning a new task mapping. A cosine schedule with warmup_ratio=0.05 works well
- **When to skip:** If your dataset is small (<1M interactions) and you are time-constrained, SFT-only often works surprisingly well. CPT matters most when the gap between SID vocabulary and pretrained text representations is large

**CPT-specific tips:**
- Pack multiple short user histories into one training sequence to maximize GPU utilization (sequence packing). For a 2048-token context, you can fit ~10 user histories of 20 items each
- Shuffle at the user level, not the token level — preserve within-user temporal order
- Include item metadata documents ("Inception is a 2010 sci-fi thriller directed by Christopher Nolan...") interleaved with user histories to ground item tokens in natural language meaning

### 4.2 SFT (Supervised Fine-Tuning)

**Goal:** Teach the model the exact task format — given a user history, predict the next item.

- **Format:** Instruction template with chat format (Mode 2); mask context tokens from loss (supervise response only)
- **Data:** For each user with history `[i_1, ..., i_T]`, create `T-1` training pairs: `(history[:t], i_{t+1})` for `t ∈ {min_len, ..., T-1}`
- **Loss:** `L_SFT = -1/(T - t) * sum_{t=input_len+1}^{T} log P(token_t | token_{<t})`
- **LR:** 2e-4 with LoRA (higher than CPT — task specialization needs stronger signal per step)
- **Epochs:** 3–5; watch for overfitting on small datasets (< 100K users). Use val HR@10 as early stopping criterion

**Loss masking is critical:** Without masking, the model optimizes for predicting the user history (which it already saw in context) rather than the next item. The collator sets `labels[i] = -100` for all `i < input_len`. Verify this is working by confirming `train_loss` at step 1 is close to `-log(1 / (vocab_size + Y))` ≈ log(32K) ≈ 10.4 for random initialization.

### 4.3 RL / Preference Alignment

**Goal:** Push the model to prefer items users actually engage with over irrelevant alternatives.

**DPO (vanilla — what we have):**

DPO reparameterizes the RL objective using the log-ratio between the policy and a frozen reference model:

```
L_DPO = -E[log σ(β · (log π_θ(y_w|x) - log π_ref(y_w|x) - log π_θ(y_l|x) + log π_ref(y_l|x)))]
```

Where `y_w` = chosen item SID sequence, `y_l` = rejected item SID sequence, `β` controls the KL penalty against the reference. `β=0.1` is a common starting point.

The log probability of a SID sequence is `log P(sequence) = sum_{l=1}^{3} log P(<item_Ll_k_l> | context, preceding tokens)`.

**DAPO (GR2 style — stronger):**

DAPO (Decoupled CLIP and Dynamic Sampling Policy Optimization) addresses two failure modes of DPO:

1. **Decoupled clip:** Standard PPO clips both the `π_θ/π_ref` ratio for chosen and rejected responses to `[1-ε, 1+ε]`. DAPO uses separate thresholds: a higher upper bound for chosen (encouraging more probability mass on correct items) and a lower bound of 0 for rejected (allowing aggressive suppression of wrong items). This asymmetric treatment is more appropriate for recommendation, where the cost of missing the right item differs from the cost of recommending a wrong one.

2. **Dynamic sampling:** Re-samples training pairs online based on the model's current scoring. Pairs where the model already strongly separates chosen from rejected (large log-ratio) contribute little gradient signal and waste compute. DAPO prioritizes near-boundary pairs where `log π_θ(y_w) - log π_θ(y_l) ≈ 0`.

3. **Verifiable reward:** Rather than using soft preference labels, use a binary reward: `r = 1` if the model ranks the target item in top-K, else `r = 0`. This avoids reward hacking because the reward is grounded in an objective metric.

**What to use as negatives for DPO:**
- Random items from catalog: easy but weak — the model already assigns very low probability to obviously irrelevant items, so the gradient signal is near-zero
- Hard negatives (items from the same category, same decade, same director): tests fine-grained discrimination
- Popularity-stratified negatives: mix popular + long-tail for balanced coverage
- BM25 / embedding-retrieved negatives: surface the hardest false positives — items that textually look like the target but are not what the user chose

---

## 5. Sampling

### 5.1 Training negatives

The model never sees explicit negatives during LM training, but negative sampling matters for DPO/RLHF and evaluation.

| Strategy | Cost | Quality | Why |
|---|---|---|---|
| Uniform random | Very low | Weak | Most items are obviously irrelevant — near-zero gradient |
| In-category hard negatives | Low | Good | Tests fine-grained discrimination within genre/topic |
| In-batch negatives | Zero extra cost | Good at scale | Other items in the batch are implicit negatives; scales naturally |
| BM25 / embedding retrieval | Medium | Best | Surfaces hardest false positives — items the model could plausibly score highly |

**Recommendation:** Start with uniform random to get a working baseline, then switch to in-category hard negatives for the DPO stage.

**Cold-start bias:** Uniform sampling over-represents popular items (because there are few of them) and under-represents long-tail items. For a catalog of 100K items where the top 1K items have 80% of interactions, uniform sampling from catalog will produce easy negatives 99% of the time. Popularity-inverse weighting corrects this: `P(item) ∝ 1 / sqrt(interaction_count)`.

### 5.2 Rejection sampling for reasoning traces (GR2 / OneRec-Think)

To generate high-quality SFT data with reasoning:
1. Prompt a large teacher model (Qwen-72B, GPT-4o) with the user history
2. Ask it to reason step-by-step before predicting the next item
3. Run the student model on the same context
4. **Keep only traces where the student's prediction matches the target** (rejection sampling)
5. Use accepted traces as SFT data

Rejection rate varies significantly: a well-pretrained student might accept 30–60% of teacher traces; a weaker student accepts fewer. The accepted traces are high signal precisely because the student can follow the reasoning chain to the correct answer. This is expensive but produces much higher-quality supervision than template-based reasoning.

In practice, batch the teacher calls via API (GPT-4o) overnight and run rejection filtering on the student in a separate pass.

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
The gap suggests Interstellar or Dunkirk.
</think>
<item_L1_15><item_L2_7><item_L3_33>
```

**Why it helps:** Forces explicit extraction of preference dimensions before committing to an item. Particularly valuable for users with diverse histories where a shallow pattern-match would pick the wrong mode (e.g., user watches action films on weekdays and drama films on weekends, but the context only shows weekday items).

The `<think>` block is supervised during SFT (contributes to loss). At inference, you can set `max_reasoning_tokens` to control latency vs. quality.

### 6.2 Distillation pipeline

```
Teacher LLM (Qwen-72B / GPT-4o)            Student LLM (7B, being trained)
      |                                             |
      | generate <think>...</think> + item SID      |
      |                                             |
      ↓                                             |
 (context, reasoning_trace, target_SID)            |
      |                                             |
      | rejection sampling:                         |
      | student top-1 prediction == target_SID? ───┘
      |
      ↓
SFT dataset: (context, <think>trace</think> + target_SID)
```

The rejection filter ensures the trace is actually useful for the student — if the student can reach the right answer with this trace as supervision, it is a good training example.

### 6.3 Reward functions for RL

For recommendation, rewards should be **verifiable** (grounded in objective criteria rather than a preference model):

| Reward | Formula | Strength | Risk |
|---|---|---|---|
| Hit reward | `r = 1[target ∈ top-K]` | Clean, sparse | High variance; needs many samples |
| NDCG reward | `r = log2(2) / log2(rank + 1)` | Continuous, dense gradient | Still rank-discontinuous |
| Engagement reward | `r = predicted_ctr(generated_item)` | Dense | Reward hacking if ctr model is weak |
| Diversity penalty | `r -= λ * mean_pairwise_sim(top-K)` | Secondary objective | Hyper-parameter sensitivity |

**NDCG as a training reward** requires differentiating through the ranking function, which is non-trivial. In practice, GR2 uses the verifiable hit reward with GRPO (Group Relative Policy Optimization) — sample G completions per context, compute hit reward for each, normalize within the group to get advantage estimates, and update the policy.

---

## 7. Inference

### 7.1 Constrained decoding

For SID-based models, restrict beam search to valid item token sequences via a prefix trie built over the SID vocabulary:

```python
# At L1: valid next tokens = {<item_L1_0>, ..., <item_L1_(K-1)>}
# At L2: valid next tokens depend on which L1 was selected
# At L3: same — constrained by (L1, L2) prefix

trie = PrefixTrie()
for item_id, (c1, c2, c3) in sid_assignments.items():
    trie.insert([c1_token, c2_token, c3_token])

# During beam search, logits for invalid tokens are set to -inf before softmax
logits[:, invalid_token_ids] = float('-inf')
```

This guarantees every generated 3-token sequence maps to a valid item. True top-K retrieval requires only one forward pass: the beam search produces K candidates directly, without scoring all N items.

**Prefix trie construction:** For a catalog of N items with 3-level SIDs, the trie has at most K₁ × K₂ + K₁ + 1 nodes (K codebook size). For 256-code codebooks, this is ~66K nodes — trivially small.

### 7.2 Log-prob scoring (what we currently use)

Score each candidate item by the sum of log probabilities of its SID tokens:

```
score(item) = sum_{l=1}^{3} log P(<item_Ll_k_l> | context, <item_L1..l-1>)
```

For text tokens: `score(item) = sum_{t=1}^{T_title} log P(title_token_t | context, title_token_{<t})`

Rank all N_eval candidates by score, report position of the ground-truth item. Scales as O(N × 1) forward passes with KV cache reuse for the shared context — the context is encoded once, then each candidate appends its tokens to the cached KV state.

For evaluation with 100 candidates, this requires 100 decoder steps per user (3 SID tokens × 100 candidates, with shared context KV cache). Feasible for offline evaluation; too slow for production over 1M items.

### 7.3 Trade-offs

| Method | Speed | Requires | Use case |
|---|---|---|---|
| Constrained beam search | O(K) per forward pass → top-K in one pass | Semantic IDs + prefix trie | Production retrieval |
| Log-prob scoring | O(N × L_item) with KV cache | Any tokenization | Research evaluation |
| Embedding similarity | O(1) ANN lookup | Separate encoder | Dual-encoder baseline |

---

## 8. Evaluation

### 8.1 Standard offline metrics

Always report over the 100-candidate pool (99 random negatives + 1 target). This is the industry standard for offline evaluation — scoring all items is expensive and noisy.

| Metric | Formula | What it measures |
|---|---|---|
| HR@K | `1[rank ≤ K]` | Did the model surface the target in top-K? |
| NDCG@K | `log(2) / log(rank+1)` if rank ≤ K, else 0 | Rank-weighted precision |
| MRR | `1 / rank` | Mean reciprocal rank — sensitive to very high ranks |

**HR@1** is the strictest: the model must get it exactly right. **HR@10** is more forgiving and closer to what matters in a real system (showing 10 results per page).

**Statistical significance:** When comparing two models, use a paired t-test over per-user metric values. For MovieLens 1M with ~6K test users, differences of >0.005 HR@10 are usually significant at p<0.05. Report confidence intervals, not just point estimates.

**Negative sampling bias:** Sampling 99 random negatives creates an artificially easy evaluation. Random negatives are easy to beat — a random ranker would score HR@10 = 0.10. In real systems, candidates are retrieved from ANN search over the entire catalog, so the actual negatives include many semantically similar items. When possible, use popularity-weighted negative sampling or retrieval-based negatives for a harder and more realistic evaluation.

### 8.2 Beyond standard metrics

| Metric | Formula | Why it matters |
|---|---|---|
| Coverage@K | `|∪_u top-K(u)| / |catalog|` | What fraction of items ever appear in top-K? |
| Intra-list diversity | `1 - mean_{i≠j} sim(i, j)` in top-K | Penalizes repetitive recommendations |
| Cold-start HR@10 | HR@10 restricted to items with <10 interactions | How well the model generalizes to new items |
| Serendipity | Fraction of top-K outside user's observed genres | Did the model surface unexpected relevant items? |

### 8.3 Ablation checklist for a paper

- [ ] Text tokens vs. semantic IDs (item tokenization)
- [ ] CPT + SFT vs. SFT only (does CPT help, and by how much?)
- [ ] LM loss only vs. + engagement loss (does auxiliary engagement signal help?)
- [ ] Rating-included verbalization vs. title-only
- [ ] History length: 10 vs. 20 vs. 50 interactions
- [ ] LoRA rank: 8 vs. 16 vs. 64
- [ ] With vs. without reasoning scaffold (if implemented)
- [ ] CF-augmented SIDs vs. text-only SIDs
- [ ] Random negatives vs. hard negatives in DPO (quality of negatives)
- [ ] Mean-initialized SID embeddings vs. random initialization (training stability)
