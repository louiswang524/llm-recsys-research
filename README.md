# LLM RecSys Research

A research framework for training and evaluating LLMs on verbalized user behavior for recommendation. Supports next-item prediction via continued pre-training (CPT) and supervised fine-tuning (SFT), with modular loss objectives and swappable item tokenization strategies.

Inspired by [PLUM](https://arxiv.org/abs/2510.07784) (Google/YouTube) and [OneRec-Think](https://arxiv.org/abs/2510.11639) (Kuaishou).

**[Architecture diagrams →](docs/architecture.md)** · **[Gap analysis vs. PLUM / OneRec-Think / GR2 →](docs/gap_analysis.md)** · **[LLM-native rec guide →](docs/llm_native_rec_guide.md)**

## What it does

Converts user interaction histories into natural language:

```
User history:
1. The Shawshank Redemption [Drama] — 5/5
2. Pulp Fiction [Crime, Drama] — 4/5
3. The Matrix [Action, Sci-Fi] — 5/5

Predict the next item.
Next: Inception
```

Then fine-tunes an LLM (Qwen, Llama, etc.) to predict the next item token — either as plain text or as discrete **Semantic ID tokens** (RQ-VAE quantized item embeddings, PLUM-style).

---

## Setup

```bash
git clone <this-repo>
cd llm-recsys-research
pip install -e .
pip install -r requirements.txt
```

For A100 / CUDA 11.8+, uncomment `flash-attn` and `bitsandbytes` in `requirements.txt` before installing.

**Colab:**
```python
!git clone https://github.com/louiswang524/llm-recsys-research
%cd /content/llm-recsys-research
!pip install -e . -r requirements.txt -q
```

---

## Quick Start (end-to-end)

```bash
# Full pipeline: download → preprocess → vocab → train → evaluate
python scripts/run_experiment.py

# With custom options
python scripts/run_experiment.py \
  --dataset movielens_1m \
  --model qwen2_7b_lora \
  --item-tokenizer text \
  --stage sft \
  --loss lm_only
```

Or run steps individually:

```bash
python scripts/01_download_data.py --movielens ml-1m
python scripts/02_preprocess.py
python scripts/03_build_item_vocab.py
python scripts/04_train.py
python scripts/05_evaluate.py checkpoint=outputs/<timestamp>/final
```

---

## Datasets

| Dataset | Config | Notes |
|---|---|---|
| MovieLens 1M | `data=movielens_1m` | ~6K users, 3.7K movies, ratings + genres |
| MovieLens 20M | `data=movielens_20m` | ~138K users, larger scale |
| Amazon Beauty | `data=amazon_beauty` | ~1.2K items, includes review text |

Add more Amazon categories by creating a new config under `configs/data/` and changing `category:`.

---

## Config System (Hydra)

Every axis of the experiment is a config group. Mix and match on the command line:

```bash
# axes
data=           movielens_1m | movielens_20m | amazon_beauty
model=          qwen2_7b_lora | llama3_8b_lora
model/item_tokenizer=  text | semantic_id
training=       cpt | sft | dpo
loss=           lm_only | lm_engagement | lm_mtp
```

### Key overrides

```bash
# Smaller LoRA rank for faster iteration
python scripts/04_train.py model.lora.r=8

# QLoRA (4-bit) for smaller GPUs
python scripts/04_train.py model.load_in_4bit=true

# Longer history
python scripts/04_train.py data.max_history_len=50

# Use minimal verbalization (title-only, good ablation)
python scripts/04_train.py data.verbalization_template=minimal

# CPT → SFT two-stage pipeline
python scripts/04_train.py training=cpt
python scripts/04_train.py training=sft training.resume_from_checkpoint=outputs/<cpt-run>/final

# Enable WandB
python scripts/04_train.py use_wandb=true
```

---

## Item Tokenization

Two strategies, swappable via config:

**Text** (`model/item_tokenizer=text`): Items are their title text. No vocab change. Simple baseline.

**Semantic ID** (`model/item_tokenizer=semantic_id`): Items → sentence-transformer embeddings → RQ-VAE quantization → 3 new special tokens per item added to LLM vocab. The model learns to predict `<item_L1_42><item_L2_17><item_L3_5>`.

```bash
# Must run build_item_vocab with the right tokenizer before training
python scripts/03_build_item_vocab.py model/item_tokenizer=semantic_id
python scripts/04_train.py model/item_tokenizer=semantic_id
```

---

## Loss Objectives

| Config | Active losses |
|---|---|
| `loss=lm_only` | Standard causal LM loss on next-item tokens |
| `loss=lm_engagement` | LM + MSE on rating prediction (auxiliary engagement head) |
| `loss=lm_mtp` | LM + multi-token prediction (next K items) |

Weights are tunable: `loss.engage_weight=0.3`, `loss.mtp_weight=0.5`.

---

## Evaluation

Evaluation uses **log-probability scoring**: for each test user, score 99 random negatives + 1 ground-truth item by `log P(item | context)`, then rank. Reports:

- `HR@1`, `HR@5`, `HR@10`, `HR@20`
- `NDCG@5`, `NDCG@10`, `NDCG@20`
- `MRR`

```bash
python scripts/05_evaluate.py checkpoint=outputs/<run>/final
```

---

## Baselines

`src/llm_recsys/models/baselines/` includes:

- **SASRec** — self-attentive sequential recommendation
- **BERT4Rec** — masked item modeling

These use integer item IDs, not text, and are trained separately as comparison points.

---

## Research Ablations

Common paper experiments this framework supports out of the box:

| Research question | Command |
|---|---|
| Text vs. Semantic ID tokens | swap `model/item_tokenizer` |
| CPT helps? | compare `training=cpt→sft` vs `training=sft` alone |
| Engagement loss helps? | compare `loss=lm_only` vs `loss=lm_engagement` |
| Verbalization style matters? | swap `data.verbalization_template` |
| History length sensitivity | sweep `data.max_history_len` |
| Review text vs. title only | set `data.include_review_text=false` on Amazon |
| Model family | swap `model=llama3_8b_lora` |

---

## Project Structure

```
configs/              Hydra config groups
  data/               dataset configs
  model/              model + LoRA configs
  model/item_tokenizer/  text or semantic_id
  training/           cpt, sft, dpo
  loss/               loss weight configs
  eval/               evaluation settings
src/llm_recsys/
  data/
    datasets/         MovieLens, Amazon loaders
    verbalization/    history → text templates
    item_tokenizer/   TextItemTokenizer, SemanticIDTokenizer (RQ-VAE)
    collator.py       DataCollator with SFT label masking
    splits.py         leave-one-out, temporal splits
  models/
    llm_rec.py        HF model + LoRA + engagement head
    losses.py         modular LM / engagement / MTP loss
    baselines/        SASRec, BERT4Rec
  training/
    trainer.py        RecTrainer (HF Trainer subclass)
    stages.py         run_stage (CPT/SFT/DPO)
    callbacks.py      RecMetrics eval callback
  evaluation/
    metrics.py        HR@K, NDCG@K, MRR
    candidate_scoring.py  log-prob item ranking
    evaluator.py      full test-set evaluation loop
scripts/
  01_download_data.py
  02_preprocess.py
  03_build_item_vocab.py
  04_train.py
  05_evaluate.py
  run_experiment.py   end-to-end pipeline script
```

---

## References

- [PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations](https://arxiv.org/abs/2510.07784)
- [OneRec-Think: In-Text Reasoning for Generative Recommendation](https://arxiv.org/abs/2510.11639)
- [SASRec: Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781)
- [BERT4Rec: Sequential Recommendation with BERT](https://arxiv.org/abs/1904.06690)
