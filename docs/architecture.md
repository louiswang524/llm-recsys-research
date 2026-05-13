# Architecture

## Pipeline State Machine

Top-level states from raw data to evaluation results. Each transition is one script.

```mermaid
stateDiagram-v2
    direction LR

    [*] --> Raw : clone repo

    Raw --> Preprocessed : 01_download_data.py\n02_preprocess.py
    note right of Preprocessed
        split.pkl
        item_meta.pkl
        all_item_ids.pkl
    end note

    Preprocessed --> VocabReady : 03_build_item_vocab.py
    note right of VocabReady
        item_tokenizer.pkl
        extended tokenizer/
    end note

    VocabReady --> CPT : training=cpt
    VocabReady --> SFT : training=sft

    CPT --> SFT : resume_from_checkpoint\n(two-stage pipeline)
    SFT --> DPO : training=dpo\n(optional)

    SFT --> Evaluated : 05_evaluate.py
    DPO --> Evaluated : 05_evaluate.py

    Evaluated --> [*]
```

---

## Data Flow

How raw interactions become tokenized training examples.

```mermaid
flowchart TD
    subgraph Input["📂 Input"]
        ML["MovieLens\n.dat / .csv"]
        AMZ["Amazon Reviews\n.jsonl.gz"]
    end

    subgraph Datasets["data/datasets/"]
        ML --> MLD["MovieLensDataset.load()"]
        AMZ --> AMZD["AmazonReviewsDataset.load()"]
        MLD --> UH["UserHistory\n{user_id, item_ids,\nratings, timestamps,\nitem_metadata, extra}"]
        AMZD --> UH
    end

    subgraph Splits["data/splits.py"]
        UH --> SP["split_users()"]
        SP --> TR["train split"]
        SP --> VA["val split"]
        SP --> TE["test split"]
    end

    subgraph ItemTok["data/item_tokenizer/"]
        direction TB
        IT1["TextItemTokenizer\n'Toy Story (1995)'"]
        IT2["SemanticIDTokenizer\n&lt;item_L1_3&gt;&lt;item_L2_17&gt;&lt;item_L3_5&gt;"]
    end

    subgraph Verbalization["data/verbalization/"]
        TR --> VB
        VB["RatingHistoryVerbalizer\nor MinimalVerbalizer"]
        VB --> FM["InstructionFormatter\napply chat template\nmask input_len for SFT"]
    end

    FM --> COL["RecDataCollator\npad · truncate · label mask"]
    IT2 -.->|"extend vocab"| COL
    IT1 -.->|"no vocab change"| COL

    COL --> DS["SequenceDataset\n{full_text, input_len, rating}"]
```

---

## Model & Training

How the model is built and how losses are computed.

```mermaid
flowchart TD
    subgraph Model["models/llm_rec.py  —  LLMRecModel"]
        BASE["HF CausalLM\nQwen2.5-7B / Llama3-8B"]
        LORA["LoRA adapter\nvia PEFT"]
        HEAD["Engagement head\nLinear(hidden, 1)"]
        BASE --> LORA
        BASE --> HEAD
    end

    subgraph LossFn["models/losses.py  —  RecLoss"]
        direction LR
        LM["LM loss\nnext-token CE\n× lm_weight"]
        ENG["Engagement loss\nMSE on rating/5\n× engage_weight"]
        MTP["MTP loss\nnext-K items CE\n× mtp_weight"]
        TOT["total loss"]
        LM --> TOT
        ENG --> TOT
        MTP --> TOT
    end

    subgraph Trainer["training/trainer.py  —  RecTrainer"]
        FWD["forward()"]
        FWD --> LM
        FWD --> ENG
        FWD --> MTP
    end

    DS["SequenceDataset"] --> Trainer
    Model --> FWD
    HEAD -.->|"predict_engagement()"| ENG

    subgraph Stages["training/stages.py"]
        CPT_ST["Stage: CPT\nlr=5e-5, 3 epochs\nall tokens supervised"]
        SFT_ST["Stage: SFT\nlr=2e-4, 5 epochs\ncontext masked from loss"]
        DPO_ST["Stage: DPO\nlr=5e-5, 2 epochs\npreferred vs negative items"]
        CPT_ST -->|checkpoint| SFT_ST
        SFT_ST -->|checkpoint| DPO_ST
    end

    Trainer --> Stages
```

---

## Evaluation Flow

How a trained model is evaluated on the test split.

```mermaid
flowchart LR
    subgraph EvalLoop["evaluation/evaluator.py"]
        TU["test user"]
        TU --> CTX["context =\nhistory[:-1]"]
        TU --> TGT["target =\nhistory[-1]"]
        CTX --> VB2["verbalizer.verbalize()"]
        VB2 --> CS
        TGT --> CAND
        ALL["all_item_ids"] --> NEG["sample 99 negatives"]
        NEG --> CAND["100 candidates\n[target + negatives]"]
        CAND --> CS["CandidateScorer\nlog P(item | context)\nper candidate"]
        CS --> RANK["ranked list"]
    end

    RANK --> MET["metrics.py\nHR@K · NDCG@K · MRR"]
    MET --> OUT["results dict\nprinted + logged"]
```

---

## Item Tokenization Strategies

```mermaid
flowchart TD
    ITEMS["Item corpus\n{item_id → title, description, ...}"]

    ITEMS --> TEXT_PATH
    ITEMS --> SID_PATH

    subgraph TEXT_PATH["Text mode  (model/item_tokenizer=text)"]
        direction TB
        T1["title text\n'The Dark Knight (2008)'"]
        T2["tokenized by\nexisting LLM vocab"]
        T1 --> T2
    end

    subgraph SID_PATH["Semantic ID mode  (model/item_tokenizer=semantic_id)"]
        direction TB
        S1["SentenceTransformer\nencodes title+description"]
        S2["PCA → 256-dim"]
        S3["ResidualQuantizer\n3-level K-means\ncodebook_size=8192"]
        S4["new special tokens\n&lt;item_L1_42&gt;&lt;item_L2_17&gt;&lt;item_L3_5&gt;\nadded to LLM vocab"]
        S1 --> S2 --> S3 --> S4
    end

    TEXT_PATH --> TRAIN["LLM training\nnext-token prediction"]
    SID_PATH --> TRAIN
```
