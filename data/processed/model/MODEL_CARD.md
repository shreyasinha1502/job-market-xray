# Model card: seniority classifier (Job Market X-Ray, M5)

Generated 2026-10-01T17:22:01+00:00 from `data/processed/model/seniority_metrics.json`, code commit
`02279aa`.

## What it does

Classifies a job description as **below_senior vs senior**. `below_senior` pools the rule classes
mid, junior and intern. Those three can't be modelled separately yet: on the 2026-10-01 snapshot
the rules found 112 mid,
18 intern and
0 junior postings, against a minimum of
100 per class (`config/labeling.yaml`). No label was
invented to fill a class.

**Intended use:** an analytics demo of seniority mix in tech job postings.
**Not for** screening candidates or any decision about a person.

## Data

- Real postings from the public Greenhouse / Lever / Ashby boards of a fixed 55-employer panel,
  India locations only, first snapshot 2026-10-01. The latest version of each posting is used.
- **Labels are derived by rule from each posting's own fields** (`config/labeling.yaml`). Word
  cues ("senior", "staff", "lead", "associate", "intern"...) count in the title only. Experience
  ranges ("3+ years") count anywhere. The title decides. Two or more classes means excluded.
  Management titles (director, VP, head...) are excluded.

| outcome | postings |
|---|---|
| excluded:ambiguous_description | 4 |
| excluded:ambiguous_title | 10 |
| excluded:empty_model_input | 3 |
| excluded:management_title | 58 |
| excluded:no_signal | 509 |
| labeled_from_description_years | 119 |
| labeled_from_title | 612 |

Labelled: 731 (below_senior 130, senior 601).

**Splits** are stratified and grouped by identical input text, so the same job posted for several
cities never sits in both train and test. Seed 42.

| split | below_senior | senior |
|---|---|---|
| train | 91 | 421 |
| val | 13 | 60 |
| test | 26 | 120 |

## Leakage controls

- The **title is not in the input**, because it is where most labels come from.
- **Every labeling-rule pattern is deleted** from the description before training and prediction.
- Only role sections are used (requirements first, then responsibilities, then intro). Company
  About, benefits and legal boilerplate are dropped, so the model can't key on employer templates.
- Imbalance is handled with **class weights** (in the loss for DistilBERT, `class_weight=balanced`
  for logistic regression). There is no oversampling and no generated text.

## Results (held-out test set, used once per model)

Macro-F1 has a bootstrap 95% CI (1,000 resamples of the real test predictions).

| model | accuracy | macro-F1 (95% CI) | below_senior P / R / F1 | senior P / R / F1 |
|---|---|---|---|---|
| majority class (always "senior") | 0.822 | **0.451** (0.43–0.47) | 0.00 / 0.00 / 0.00 | 0.82 / 1.00 / 0.90 |
| TF-IDF + logistic regression | 0.843 | **0.686** (0.58–0.79) | 0.59 / 0.38 / 0.47 | 0.88 / 0.94 / 0.91 |
| DistilBERT fine-tuned | 0.849 | **0.694** (0.58–0.80) | 0.62 / 0.38 / 0.48 | 0.88 / 0.95 / 0.91 |
| DistilBERT int8 ONNX (served) | 0.856 | **0.689** (0.57–0.80) | 0.69 / 0.35 / 0.46 | 0.87 / 0.97 / 0.92 |

**Paired comparison on the same test postings** (exact McNemar on discordant predictions): both correct 116, only TF-IDF+LR correct 7, only DistilBERT correct 8, both wrong 15. p = 1.0, so **no significant difference at 0.05**. On this data, fine-tuning does not beat the linear baseline.

### DistilBERT confusion matrix (test)

| true \ predicted | below_senior | senior |
|---|---|---|
| below_senior | 10 | 16 |
| senior | 6 | 114 |

### TF-IDF + LR confusion matrix (test)

| true \ predicted | below_senior | senior |
|---|---|---|
| below_senior | 10 | 16 |
| senior | 7 | 113 |

### DistilBERT training (CPU)

| epoch | train loss | val macro-F1 | seconds |
|---|---|---|---|
| 1 | 0.6754 | 0.6191 | 527 |
| 2 | 0.6354 | 0.6789 | 515 |
| 3 | 0.4685 | 0.8053 | 463 |
| 4 | 0.3107 | 0.7678 | 466 |

Settings: `distilbert-base-uncased` @ `12040accade4e8a0f71eabdb258fecc2e7e948be`, max 256
tokens, batch 8, lr 5e-05, up to 4 epochs with
early stopping on validation macro-F1 (patience 1). The baseline's C was chosen
on validation macro-F1.

## Limitations

- **Small minority test set** (26
  below_senior postings), which is why the confidence intervals are wide. Read differences
  between models against those intervals.
- **Rule labels are noisy by construction.** "Lead" in operations titles (e.g. "Collections Team
  Lead") counts as senior. "Associate" counts as mid. A single seed was used and there was no
  hyperparameter search for the transformer.
- **One employer panel and one region.** The panel skews to product/SaaS companies in Bengaluru
  and may not transfer to other employers or markets.
- Inputs are truncated to 256 tokens, with requirements first.
- Labels say what the rules see in titles. The model learns textual correlates of those titles,
  not a ground truth of the job's real level.

## Files

- Weights: `models/seniority-distilbert/` (not in git; distributed separately). Baseline:
  `models/seniority-tfidf-logreg.joblib`.
- Labels with provenance and split: `data/processed/model/seniority_labels.parquet`.
- Metrics: `data/processed/model/seniority_metrics.json`.
- Load: `xray.classify.predict.SeniorityPredictor.load()`. CLI: `python -m xray predict --file jd.txt`.
