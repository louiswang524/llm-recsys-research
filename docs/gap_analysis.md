# Gap Analysis: vs. PLUM, OneRec-Think, GR2

## Alignment Summary

```
                        PLUM        OneRec-Think    GR2         Ours
────────────────────────────────────────────────────────────────────────
Semantic IDs            ✅ SIDv2    ✅ 3-level      ✅          ✅ text-only RQ-VAE
CPT stage               ✅          ✅              ✅          ✅
SFT stage               ✅          ✅              ✅          ✅
RL / preference align   ❌          ✅ custom RL    ✅ DAPO     ⚠️  vanilla DPO
Chain-of-thought        ❌          ✅              ✅          ❌
Constrained decoding    ✅          ✅              ✅          ❌ log-prob only
Reranking (listwise)    ❌          ❌              ✅          ❌
CF signals in SID       ✅          ✅              ✅          ❌
Engagement aux loss     ❌          ❌              ❌          ✅ unique
Verbalization ablations ❌          ❌              ❌          ✅ unique
```

---

## vs. PLUM (Google / YouTube)

**What aligns:**
- RQ-VAE semantic ID tokenizer with the same 3-level hierarchical structure
- Two-stage CPT → SFT training pipeline
- Amazon Reviews text used during CPT for domain grounding
- Standard LM loss over item token sequences

**Gaps:**

| Gap | PLUM | Ours |
|---|---|---|
| SID input signals | CF embeddings + text (SIDv2) | Text-only |
| CPT scale | 100B+ tokens | Dataset-scale |
| Constrained decoding | Beam search over item vocab | Log-prob scoring only |
| Multi-task CPT | Retrieval + CTR + engagement jointly | LM loss only |

**Research opportunity:** Ablate SID input signal — text-only vs. BPR/MF embeddings vs. blend. This directly tests PLUM's implicit claim that CF signals improve ID quality, on public datasets.

---

## vs. OneRec-Think (Kuaishou)

**What aligns:**
- Qwen as the backbone model family
- 3-level × 8192 hierarchical item token vocabulary
- CPT alignment stage before SFT
- Preference alignment stage (DPO config)

**Gaps:**

| Gap | OneRec-Think | Ours |
|---|---|---|
| Chain-of-thought | `<think>…</think>` before item tokens | Direct next-item prediction |
| Itemic alignment loss | Cross-modal item-text contrastive loss in CPT | Plain LM loss |
| RL reward | Multi-validity recommendation reward | Vanilla DPO |
| Scale | 20B tokens/day, 80 GPUs | Single GPU / Colab |

The defining feature of OneRec-Think — generating a reasoning trace before predicting the next item — is not implemented. The expected sequence format is:

```
User history: …
<think>
  The user prefers action and sci-fi films rated above 4 stars.
  Their recent watch of Inception suggests interest in psychological thrillers.
</think>
<item_L1_3><item_L2_17><item_L3_5>
```

**Research opportunity:** Add a `ReasoningVerbalizer` that prepends `<think>…</think>` blocks distilled from a larger model (Qwen-72B, GPT-4o) via rejection sampling. Research question: *does reasoning improve small-model recommendation quality, and at what cost in latency?*

---

## vs. GR2 (Meta)

**What aligns:**
- Semantic ID alignment stage
- SFT stage
- DPO config exists as a starting point

**Gaps:**

| Gap | GR2 | Ours |
|---|---|---|
| Task | Reranking (listwise input → reordered output) | Retrieval (next-item prediction) |
| SFT data | Teacher LLM reasoning traces + rejection sampling | Direct (context, target) pairs |
| RL algorithm | DAPO (decoupled clip + dynamic sampling) | Vanilla DPO |
| Reward | Verifiable reranking reward | Preference-based |
| Reasoning | ✅ CoT before reranked list | ❌ |

GR2 operates on a fundamentally different task formulation. Input is `(user context, top-K candidate list)` and output is a reordered list. This requires:
- A `RerankerDataset` that packages candidate lists per user
- A listwise verbalization format
- DAPO or a listwise DPO variant instead of pointwise DPO

**Research opportunity:** Implement the reranking task on MovieLens/Amazon as a second head, trained jointly or sequentially after the retrieval SFT. Compare pointwise vs. listwise preference alignment.

---

## Prioritized Implementation Roadmap

| Priority | Feature | Bridges | Effort |
|---|---|---|---|
| 1 | CF-augmented SID (BPR/MF + text → RQ-VAE) | PLUM | Low — swap `_embed_items()` |
| 2 | Constrained beam search decoding | PLUM, OneRec, GR2 | Medium — add to `candidate_scoring.py` |
| 3 | Reasoning verbalizer + trace distillation | OneRec-Think, GR2 | Medium — new verbalizer + data pipeline |
| 4 | Reranking task + dataset | GR2 | High — new task formulation |
| 5 | DAPO / listwise DPO | GR2 | High — new training stage |
