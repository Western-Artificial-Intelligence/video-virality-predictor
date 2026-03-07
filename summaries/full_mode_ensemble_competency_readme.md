# Full-Mode Ensemble Competency README

## 0. Purpose of This Document

This document explains, in detail, why and how we evaluate the webapp `full` mode as an ensemble model.

It covers:
- foundational concepts (regression, ensembles, residuals, stacking)
- why we use the residual analysis dataset
- exact math used in the notebook
- what artifacts are created
- why this methodology works in practice
- what the current results mean
- known limitations and recommended next steps

Primary implementation notebook:
- `stacking/full_mode_ensemble_competency.ipynb`

Primary evaluation inputs:
- `EDA/analysis/s3_snapshots/reports/residual_long_selected_2026-03-06.csv`
- `EDA/analysis/s3_snapshots/reports/metrics_selected_2026-03-06.csv`

---

## 1. Concepts From First Principles

### 1.1 What problem are we solving?

We want to predict future YouTube Shorts view counts at fixed horizons:
- 7-day horizon
- 30-day horizon

This is a supervised regression problem: for each video, we have:
- features (or model predictions derived from features)
- a numeric target (`horizon_view_count`)

### 1.2 Why use log space?

View counts are heavy-tailed (a few videos are very large). Direct raw-scale training can become unstable.

Common solution:
- train/evaluate in log space
- target: `y_log = log1p(y_raw)`

Benefits:
- reduces skew
- makes errors more comparable across scales
- typically improves optimization behavior

### 1.3 What is an ensemble?

An ensemble combines multiple models to produce one final prediction.

Two common styles:
- **Blending**: fixed rules (mean, weighted mean, median)
- **Stacking**: learn a meta-model that combines base model outputs

### 1.4 What is a residual?

For one sample:
- `residual = prediction - truth`

In this repo's residual report, each row stores:
- model identity
- sample identity (`video_id`, `horizon`)
- `y_true_log`
- `pred_log`
- residual columns

Residuals are useful because they expose each model's error pattern.

---

## 2. Why We Use the Residual Dataset

### 2.1 Practical reason

The residual report already has aligned per-video predictions and true labels for all required full-mode base models. That means we can evaluate ensemble competence immediately without rerunning:
- video/audio/text embedding pipelines
- model inference across raw assets
- expensive data assembly steps

### 2.2 Methodological reason

Stacking needs exactly this shape:
- `X`: base model predictions for each sample
- `y`: true target for each sample

The residual report already gives this in long format; we pivot to wide format.

### 2.3 Important clarification

We do **not** train a stacker on `residual_log` as target.

We train on:
- input features: base `pred_log` columns
- target: `y_true_log`

Residuals are used for diagnostics and quality checks, not as the supervised label.

### 2.4 Why this is valid

If each sample has:
- true label
- predictions from all candidate base models

then training a meta-learner on those predictions is standard stacked generalization.

---

## 3. Which Base Models Define Webapp `full` Mode

From backend mode spec (see `virality-webapp/backend/app/constants.py`), full mode uses four model-family + fusion-strategy pairs:

1. `gbdt + concat`
2. `concat_mlp + max_pool`
3. `gated_fusion_mlp + concat`
4. `ridge + sum_pool`

The notebook evaluates exactly these four as base learners.

---

## 4. End-to-End Notebook Workflow

## 4.1 Load data and validate assumptions

The notebook checks:
- required columns exist in residual and metrics files
- no duplicate `(video_id, horizon, model_family, strategy)` rows
- all four base model pairs exist per horizon

This prevents silent corruption of ensemble training/evaluation.

## 4.2 Build horizon-specific matrices

The residual report is long format. For each horizon (`7`, `30`), we pivot to one row per `video_id`:
- feature columns: 4 base `pred_log` columns
- target column: `y_true_log`
- extra context: `text_present`

Resulting supervised matrix shape:
- horizon 7: 808 rows
- horizon 30: 733 rows

## 4.3 Exact log/raw transforms used

To stay consistent with training/backend math:

- `raw_from_log(log) = clip(expm1(clip(log, -20, 30)), lower=0)`
- `log_from_raw(raw) = log1p(max(raw, 0))`

This keeps comparisons numerically consistent.

## 4.4 Unified metric function (all candidates)

Every candidate (base, deterministic blend, stacker) is scored with the same function:

Log-space:
- `mae_log`
- `rmse_log`
- `r2_log`

Raw-space:
- `mae_raw`
- `rmse_raw`
- `mape_raw` (exclude zero denominator)

Diagnostic:
- `bias_log`
- `corr_true_pred_log`
- `p95_abs_residual_log`
- `n`

This makes "combination as one model" directly comparable to any base model.

## 4.5 Base-model leaderboard

Each of the four base models is evaluated as a standalone predictor.

Also included:
- rank-alignment check vs `metrics_selected_2026-03-06.csv` test MAE
- confirms directional consistency of recomputed metrics

## 4.6 Deterministic ensemble baselines

The notebook builds fixed-rule blends in log space:

1. `mean_log`
2. `median_log`
3. `trimmed_mean_log`
   - for 4 models, sort each row and average middle two values
4. `val_weighted_mean_log`
   - weights proportional to `1 / val_mae_log` from `metrics_selected`
5. `range_midpoint_log`
   - compute robust padded range in log space
   - convert to raw min/max
   - midpoint in raw
   - transform midpoint back to log for comparable scoring

These baselines answer: "Can simple ensemble rules beat single models?"

## 4.7 Robust range math (interval side)

Per sample, from four model log predictions:

1. sort logs
2. core = middle values (drop min and max for 4 models)
3. `spread = max(core_high - core_low, 0.08)`
4. `pad = 0.5 * spread + 0.10`
5. `min_log = core_low - pad`
6. `max_log = core_high + pad`
7. convert bounds to raw using log->raw transform

This is the same robust padded logic used by backend full-mode range behavior.

## 4.8 Learned stacker search with OOF protocol

Candidate meta-models:
- `LinearRegression`
- `RidgeCV`
- `LassoCV`
- `ElasticNetCV`
- `HuberRegressor`
- `RandomForestRegressor`
- `HistGradientBoostingRegressor`

Protocol:
- repeated K-fold OOF (`n_splits=5`, `n_repeats=3`, fixed seed)
- for each fold: train on train-fold, predict held-out fold
- average OOF predictions per row across repeats
- evaluate concatenated OOF predictions

Why this matters:
- avoids in-fold evaluation optimism
- produces fairer estimate of stacker quality

## 4.9 Combined comparison

For each horizon, notebook reports:
- best base model
- best deterministic blend
- best learned stacker

And outputs unified leaderboard files.

## 4.10 Range quality evaluation

Interval metrics are computed on robust ranges:
- `coverage_raw`
- `mean_width_raw`
- `median_width_raw`
- `p90_width_raw`
- `under_cover_rate`
- `over_cover_rate`

This evaluates uncertainty interval behavior separately from point prediction accuracy.

---

## 5. Artifact Inventory

The notebook writes CSV artifacts to:
- `stacking/artifacts/`

Generated files:

Global:
- `artifact_manifest.csv`: list of generated artifact files with timestamp
- `val_weights_by_horizon.csv`: inverse-MAE weights used for weighted blend
- `range_metrics_summary.csv`: interval quality summary per horizon
- `best_comparison_summary.csv`: best base vs best deterministic vs best stack per horizon

Per horizon (`h7`, `h30`):
- `h*_base_leaderboard.csv`: base-model metrics
- `h*_deterministic_leaderboard.csv`: deterministic blend metrics
- `h*_stack_leaderboard.csv`: stacker metrics (OOF)
- `h*_all_leaderboard.csv`: merged ranked leaderboard
- `h*_range_rows.csv`: row-level robust range bounds and midpoint
- `h*_stack_oof_predictions.csv`: OOF stacker predictions per sample

---

## 6. Why This Works in Practice

### 6.1 Error diversity

Different model families make different mistakes:
- trees may capture nonlinear interactions well
- linear models can be stable but biased
- neural models may capture signal patterns others miss

If error patterns are not identical, combining models can reduce total error.

### 6.2 Meta-learning can outperform static rules

Static blends (mean/median) assume equal contribution from all models.

Stacking learns context-free weights/interactions from data, allowing:
- upweight stronger base predictors
- downweight noisy base predictors
- model small corrections not captured by fixed averaging

### 6.3 OOF reduces over-optimism

If we fit and score on the same rows, stacker metrics look overly good.

OOF forces each prediction to come from a model that did not train on that row, giving a more realistic estimate.

### 6.4 Robust ranges for interval stability

Trim + pad in log space:
- reduces sensitivity to one extreme base prediction
- produces conservative but stable intervals
- suitable for uncertainty communication

---

## 7. What We Observed in the Current Run

From notebook execution on current residual snapshot:

### Horizon 7 (point accuracy)
- Best base: `gbdt|concat` (`rmse_log ~ 1.3695`)
- Best deterministic blend: `val_weighted_mean_log` (`rmse_log ~ 1.4426`)
- Best stacker: `stack::linear` (`rmse_log ~ 1.3584`)

Interpretation:
- learned stacker beat best single base on OOF metrics
- fixed deterministic blends did not beat best base on this horizon

### Horizon 30 (point accuracy)
- Best base: `gbdt|concat` (`rmse_log ~ 1.6212`)
- Best deterministic blend: `val_weighted_mean_log` (`rmse_log ~ 2.0124`)
- Best stacker: `stack::elasticnetcv` (`rmse_log ~ 1.6005`)

Interpretation:
- stacker produced measurable gain over best base
- deterministic blends were materially weaker for 30-day in this snapshot

### Range behavior (interval quality)
- h=7 coverage: ~0.213
- h=30 coverage: ~0.288

Interpretation:
- current robust padded range is relatively narrow vs true error dispersion
- gives informative intervals, but not high-coverage prediction intervals

---

## 8. Limitations and Caveats

1. **Dataset scope**
   - Evaluation is on one selected residual snapshot; results may vary with future runs/data.

2. **No untouched external holdout in this notebook**
   - OOF is strong for internal estimation, but final claims are strongest with an external locked holdout.

3. **Meta-feature scope**
   - Stacker uses only 4 base predictions. It does not use original metadata/modal features.

4. **Range not calibrated as probabilistic interval**
   - Robust padded range is heuristic and stability-focused, not quantile-calibrated uncertainty.

5. **Raw-scale metrics can be extreme**
   - Due to heavy-tailed targets, raw RMSE may be dominated by a few large outliers.

---

## 9. Why This Method Is Still the Right Step

For the question "How competent is full mode as an ensemble?", this notebook is appropriate because it:
- compares base, blends, and stackers under one metric system
- uses true labels (not synthetic proxies)
- isolates horizon-specific behavior
- includes both point prediction and interval diagnostics
- produces reproducible artifacts for auditing and iteration

---

## 10. Recommended Next Iterations

1. **Add external holdout test**
   - Train stacker using one period/split; score once on untouched holdout period.

2. **Calibrated interval modeling**
   - Add quantile models or conformal calibration for target coverage levels.

3. **Compare full-4 vs all-12 base space**
   - Evaluate whether adding all strategy outputs improves stacking stability/performance.

4. **Stability analysis across seeds/splits**
   - Quantify variance of stacker ranking and gain margins.

5. **Operational candidate selection rule**
   - Define deployment rule, e.g., choose best stacker only if gain over best base exceeds threshold and is stable.

---

## 11. Glossary

- **Base model**: a first-level model producing direct prediction.
- **Blend**: fixed formula combination of base predictions.
- **Stacking**: learned meta-model combining base predictions.
- **OOF**: out-of-fold prediction; each row predicted by model not trained on that row.
- **Residual**: prediction minus true value.
- **Coverage**: fraction of true values inside predicted interval.
- **Sharpness**: interval narrowness (smaller width is sharper).

---

## 12. Quick Reproduction Checklist

1. Open `stacking/full_mode_ensemble_competency.ipynb`.
2. Ensure these files exist:
   - `EDA/analysis/s3_snapshots/reports/residual_long_selected_2026-03-06.csv`
   - `EDA/analysis/s3_snapshots/reports/metrics_selected_2026-03-06.csv`
3. Run notebook top-to-bottom.
4. Inspect outputs in `stacking/artifacts/`.
5. Use `best_comparison_summary.csv` and horizon leaderboards as primary decision tables.
