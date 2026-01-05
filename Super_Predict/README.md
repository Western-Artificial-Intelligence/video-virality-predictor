# Luke Jang — Virality Prediction (Supervised)

This module implements the **first supervised learning layer** in the project. Its job is to take everything produced so far (multimodal embeddings, clusters, metadata, early signals) and output a **single predicted performance score per video**.

It is intentionally designed to be:
- **Simple to start** (works with embeddings + target only)
- **Extensible** (automatically absorbs new features)
- **Model-agnostic** (easy to swap models later)
- **Production-shaped** (clean training + inference interface)

---

## What This Module Does

At a high level, this module:

1. Loads fused multimodal embeddings (`fused.pt`)
2. Converts embeddings into a tabular format
3. Merges embeddings with labels and optional features
4. Automatically preprocesses numeric and categorical data
5. Trains a supervised regression model
6. Evaluates performance on a held-out set
7. Exposes a simple inference API: `predict_score(video)`

This is the **first point in the pipeline that makes forward-looking predictions**.

---

## Why This Step Is Necessary

Earlier steps in the pipeline (embedding, dimensionality reduction, clustering) are:
- **Descriptive** (what kinds of videos exist?)
- **Exploratory** (what styles cluster together?)

They do *not* answer:

> “Given a new Short, how well will it perform?”

Virality prediction is a **supervised learning problem**, which requires:
- A target metric (views, engagement, etc.)
- A model trained to map features → outcomes

This module provides that layer without hard-coding product assumptions.

---

## Inputs

### 1. Fused Embeddings (`fused.pt`)
Produced by the fusion step.

Expected structure:
```python
{
  "ids": List[str],              # video_id
  "fused": Tensor[n_videos, d]   # fused embeddings
}
```

These embeddings represent the video’s combined:
- visual content
- audio
- text/subtitles
- metadata

They are treated as **fixed semantic features**.

---

### 2. Features / Labels Table (`features_df`)

This table is **external to this module** and typically comes from:
- YouTube Analytics export
- internal ETL / warehouse query
- manual labeling

**Required columns:**
- `video_id` (must match `fused.pt`)
- target column 

**Optional columns (automatically used if present):**
- `cluster_id` (from Dev 2)
- metadata (duration, post hour, language, etc.)
- early engagement signals (views_1h, likes_1h, retention_30s, etc.)

---

## Target Variable (Intentionally Flexible)

This module does **not define what “virality” means (at least for now)**.

Common choices:
- `log(views_7d + 1)` (most stable)
- `likes_7d`
- `engagement_rate_7d`
- `watch_time_7d`
- a composite score

### Why this vagueness is intentional
- Different goals require different targets
- Enables fast experimentation
- Prevents re-writing model code
- Could check which target variable represents virality the best.

The only requirement is that the chosen target column exists in `features_df`.

---

## Model Choice

### Default Model

**HistGradientBoostingRegressor** (scikit-learn)

Chosen because it:
- Handles **non-linear relationships**
- Works well on **tabular data + embeddings**
- Handles mixed feature types
- Requires minimal tuning
- Trains quickly

This is a **strong, low-risk baseline**.

---

### Why Not Deep Learning (Yet)

- Dataset size is likely small-to-medium
- Embeddings already encode semantics
- Tree models often outperform neural nets in this regime
- Faster iteration and easier debugging

Neural models can be explored later if scale increases.

---

### Why Not Only Linear Models

Linear models are useful baselines, but they:
- Miss non-linear feature interactions
- Underperform when style + pacing + engagement interact

That said, the pipeline allows swapping in Ridge/ElasticNet easily.

---

## Training Pipeline Overview

The training process:

1. Load fused embeddings
2. Convert embeddings → `emb_0 … emb_d`
3. Merge with labels/features on `video_id`
4. Infer feature types automatically:
   - numeric → impute + scale
   - categorical → impute + one-hot
5. Train model on train split
6. Evaluate on test split
7. Compute permutation feature importance

All preprocessing is inside an **sklearn Pipeline**, ensuring:
- No train/inference mismatch
- No preprocessing leakage

---

## Evaluation & Leakage Considerations

### Current Setup
- Random train/test split (baseline only)
- MAE and R² metrics

### Known Risks
- Temporal leakage (future signals leaking into training)
- Heavy-tailed targets (views)

### Recommended Next Steps
- Time-based splits (train on older uploads)
- Log-transform view-based targets
- Explicit feature cutoff times

These are data decisions, not model limitations.

---

## Inference Interface

The module exposes a simple API:

```python
predictor.predict_score(video_id, extra_features)
```

Where:
- `video_id` must exist in `fused.pt`
- `extra_features` may include:
  - cluster_id
  - early engagement signals
  - metadata

This wrapper ensures:
- embeddings are constructed identically to training
- preprocessing is reused safely

---

## How This Connects to Other Devs

### Dev 1 — Latent Space
- Reduced embeddings are for visualization/clustering
- Full fused embeddings are better for prediction

### Dev 2 — Clustering
- Cluster IDs are treated as categorical features
- Allows the model to learn style-level performance effects

### Dev 4 — Interpretation
- Cluster labels + feature importance help explain predictions
- Bridges model outputs to human-understandable traits

---

## What This Enables

With even a small labeled dataset, this module allows you to:
- Rank videos by predicted performance
- Estimate upside early
- Compare styles quantitatively
- Analyze which features matter

This turns the system from **descriptive** → **predictive**.

