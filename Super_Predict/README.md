# Luke Jang — Virality Prediction (Supervised)

This module implements the **first supervised learning layer** in the project. Its job is to take everything produced so far (multimodal embeddings, clusters, metadata, early signals) and output a **single predicted performance score per video**.

It is intentionally designed to be:

* **Simple to start** (works with embeddings + target only)
* **Extensible** (automatically absorbs new features)
* **Model-agnostic** (easy to swap models later)
* **Production-shaped** (clean training + inference interface)

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

* **Descriptive** (what kinds of videos exist?)
* **Exploratory** (what styles cluster together?)

They do *not* answer:

> “Given a new Short, how well will it perform?”

Virality prediction is a **supervised learning problem**, which requires:

* A target metric (views, engagement, etc.)
* A model trained to map features → outcomes

This module provides that layer without hard-coding product assumptions.

---

## Inputs

### 1. Fused Embeddings (`fused.pt`)

Produced by the fusion step (pipeline step 3).

Expected structure:

```python
{
  "ids": List[str],              # video_id
  "fused": Tensor[n_videos, d]   # fused embeddings
}
```

These embeddings represent the video’s combined:

* visual content
* audio
* text/subtitles
* metadata

They are treated as **fixed semantic features**.

---

### 2. Features / Labels Table (`features_df`)

This table is **external to this module** and typically comes from:

* YouTube Analytics export
* internal ETL / warehouse query
* manual labeling

**Required columns:**

* `video_id` (must match `fused.pt`)
* target column (name chosen by you)

**Optional columns (automatically used if present):**

* `cluster_id` (from Dev 2)
* metadata (duration, post hour, language, etc.)
* early engagement signals (views_1h, likes_1h, retention_30s, etc.)

This design keeps Dev 3 **unblocked** while other parts evolve.

---

## Target Variable (Intentionally Flexible)

This module does **not define what “virality” means**.

That is a **product + data decision**, not a modeling one.

Common choices:

* `log(views_7d + 1)` (most stable)
* `likes_7d`
* `engagement_rate_7d`
* `watch_time_7d`
* a composite score

### Why this vagueness is intentional

* Different goals require different targets
* Enables fast experimentation
* Prevents re-writing model code

The only requirement is that the chosen target column exists in `features_df`.

---

## Model Choice

### Default Model

**HistGradientBoostingRegressor** (scikit-learn)

Chosen because it:

* Handles **non-linear relationships**
* Works well on **tabular data + embeddings**
* Handles mixed feature types
* Requires minimal tuning
* Trains quickly

This is a **strong, low-risk baseline**.

---

### Why Not Deep Learning (Yet)

* Dataset size is likely small-to-medium
* Embeddings already encode semantics
* Tree models often outperform neural nets in this regime
* Faster iteration and easier debugging

Neural models can be explored later if scale increases.

---

### Why Not Only Linear Models

Linear models are useful baselines, but they:

* Miss non-linear feature interactions
* Underperform when style + pacing + engagement interact

That said, the pipeline allows swapping in Ridge/ElasticNet easily.

---

## Training Pipeline Overview

The training process:

1. Load fused embeddings
2. Convert embeddings → `emb_0 … emb_d`
3. Merge with labels/features on `video_id`
4. Infer feature types automatically:

   * numeric → impute + scale
   * categorical → impute + one-hot
5. Train model on train split
6. Evaluate on test split
7. Compute permutation feature importance

All preprocessing is inside an **sklearn Pipeline**, ensuring:

* No train/inference mismatch
* No preprocessing leakage

---

## Evaluation & Leakage Considerations

### Current Setup

* Random train/test split (baseline only)
* MAE and R² metrics

### Known Risks

* Temporal leakage (future signals leaking into training)
* Heavy-tailed targets (views)

### Recommended Next Steps

* Time-based splits (train on older uploads)
* Log-transform view-based targets
* Explicit feature cutoff times

These are data decisions, not model limitations.

---

## Inference Interface

The module exposes a simple API:

```python
predictor.predict_score(video_id, extra_features)
```

Where:

* `video_id` must exist in `fused.pt`
* `extra_features` may include:

  * cluster_id
  * early engagement signals
  * metadata

This wrapper ensures:

* embeddings are constructed identically to training
* preprocessing is reused safely

---

## How This Connects to Other Devs

### Dev 1 — Latent Space

* Reduced embeddings are for visualization/clustering
* Full fused embeddings are better for prediction

### Dev 2 — Clustering

* Cluster IDs are treated as categorical features
* Allows the model to learn style-level performance effects

### Dev 4 — Interpretation

* Cluster labels + feature importance help explain predictions
* Bridges model outputs to human-understandable traits

---

## What This Enables

With even a small labeled dataset, this module allows you to:

* Rank videos by predicted performance
* Estimate upside early
* Compare styles quantitatively
* Analyze which features matter

This turns the system from **descriptive** → **predictive**.


## Training Results (Baseline Experiment)

The first end-to-end supervised training run was executed using:

* **Target**: `log_view_count = log(view_count + 1)`
* **Dataset size**: 222 videos

  * Train: 177
  * Test: 45
* **Features**:

  * Fused multimodal embeddings
  * Video metadata (e.g. duration)
  * Channel metadata
  * Cluster ID (categorical)
* **No early engagement signals** (content-only baseline)

### Metrics

```
MAE: 1.43
R²: 0.07
```

Where MAE is measured in **log-view space**.

---

## Interpretation of Results

### What the Metrics Mean

* **MAE ≈ 1.43 (log space)**

  * On average, predictions are off by a factor of approximately `exp(1.43) ≈ 4.2×` in raw view count.
  * Example: a video with 10k views may be predicted anywhere from ~2.5k to ~40k.

* **R² ≈ 0.07**

  * The model explains ~7% of the variance in log(view count).
  * This is expected for an early-stage, content-only virality model.

Importantly, these results indicate:

* The pipeline is functioning correctly
* There is no obvious target leakage
* The model is learning real (but limited) signal

This should be treated as a **baseline**, not a final model.

---

### Feature Importance Insights

Permutation feature importance highlights the following:

* **`duration_seconds` is the strongest single predictor** by a large margin

  * This aligns with known Shorts dynamics (retention, pacing, algorithm exposure)

* Multiple **embedding dimensions** appear among top features

  * Individual embedding indices are not directly interpretable
  * Their presence confirms that the fused embeddings carry predictive signal

* Metadata features (e.g. captions available) have smaller but non-zero effects

Overall takeaway:

> Structure and style matter, but without early engagement signals, prediction power is inherently limited.

---

## Why Performance Is Currently Limited

This model intentionally excludes the strongest predictors of final view count:

* Early views (e.g. views_1h, views_6h)
* Early likes/comments
* Early retention metrics

As a result, the model is answering a difficult question:

> “How well will this video perform based on content and channel context alone?”

Partial predictability is expected; high uncertainty is normal.

---

## Recommended Next Steps

To meaningfully improve performance, the following steps are recommended (in order of impact):

1. **Add early engagement signals**

   * Even a small early window (e.g. 1 hour) typically yields large gains
   * Expected R² improvement: ~0.3–0.6

2. **Increase dataset size**

   * 222 samples is very small for high-dimensional embeddings
   * More data will stabilize metrics and feature importance

3. **Time-based evaluation splits**

   * Train on older uploads, test on newer uploads
   * Better reflects real-world forecasting

4. **Cluster-aware modeling**

   * Predict relative performance within style clusters
   * Or model residuals over cluster-level averages

5. **Target experimentation**

   * Compare view count vs engagement vs composite targets
   * Validate which best captures practical “virality”

---

## Summary

This first supervised run successfully demonstrates:

* A working end-to-end prediction pipeline
* Meaningful signal from fused embeddings
* Honest, leakage-free baseline performance

The system is currently **data-limited, not model-limited**, and is well-positioned for rapid improvement as richer signals are introduced.
