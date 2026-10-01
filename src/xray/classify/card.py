"""Model card for the seniority classifier, generated from the metrics file."""

from __future__ import annotations

import subprocess
from typing import Any

from xray.config import ROOT


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, check=True)  # fmt: skip
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _row(name: str, m: dict[str, Any], labels: list[str]) -> str:
    lo, hi = m["macro_f1_ci95_bootstrap"]
    per = " | ".join(
        f"{m['per_class'][c]['precision']:.2f} / {m['per_class'][c]['recall']:.2f} / "
        f"{m['per_class'][c]['f1-score']:.2f}"
        for c in labels
    )
    return f"| {name} | {m['accuracy']:.3f} | **{m['macro_f1']:.3f}** ({lo:.2f}–{hi:.2f}) | {per} |"


def _cm(m: dict[str, Any]) -> str:
    labels = m["confusion_matrix"]["labels"]
    rows = m["confusion_matrix"]["rows_true_cols_pred"]
    head = "| true \\ predicted | " + " | ".join(labels) + " |\n|---|" + "---|" * len(labels)
    body = "\n".join(
        f"| {lab} | " + " | ".join(map(str, r)) + " |" for lab, r in zip(labels, rows, strict=True)
    )
    return head + "\n" + body


def _comparison(c: dict[str, Any] | None) -> str:
    if not c:
        return ""
    return (
        "**Paired comparison on the same test postings** (exact McNemar on discordant "
        f"predictions): both correct {c['both_correct']}, only TF-IDF+LR correct "
        f"{c['only_baseline_correct']}, only DistilBERT correct {c['only_transformer_correct']}, "
        f"both wrong {c['both_wrong']}. p = {c['mcnemar_exact_p']}, so **{c['reading']}**. "
        "On this data, fine-tuning does not beat the linear baseline."
        if c["mcnemar_exact_p"] >= 0.05
        else f"p = {c['mcnemar_exact_p']}: {c['reading']}."
    )


def model_card(res: dict[str, Any]) -> str:
    lab = res["labels"]
    labels = sorted(lab["target_counts"])
    tr = res.get("transformer")
    models = [("majority class (always \"senior\")", res["majority_class_reference"]["test"]),
              ("TF-IDF + logistic regression", res["baseline"]["test"])]  # fmt: skip
    if tr:
        models.append(("DistilBERT fine-tuned", tr["test"]))
    per_head = " | ".join(f"{c} P / R / F1" for c in labels)
    table = "\n".join(_row(n, m, labels) for n, m in models)
    hist = ""
    if tr:
        hist = "\n".join(
            f"| {h['epoch']} | {h['train_loss']:.4f} | {h['val_macro_f1']:.4f} | {h['seconds']} |"
            for h in tr["training"]["history"]
        )
    reasons = "\n".join(f"| {k} | {v} |" for k, v in sorted(lab["by_reason"].items()))
    splits = "\n".join(
        f"| {s} | " + " | ".join(str(lab["split_counts"][s].get(c, 0)) for c in labels) + " |"
        for s in ("train", "val", "test")
    )
    cfg = res["config"]
    return f"""# Model card: seniority classifier (Job Market X-Ray, M5)

Generated {res["generated_at"]} from `data/processed/model/seniority_metrics.json`, code commit
`{_git_commit()}`.

## What it does

Classifies a job description as **{" vs ".join(labels)}**. `below_senior` pools the rule classes
mid, junior and intern. Those three can't be modelled separately yet: on the 2026-10-01 snapshot
the rules found {lab["fine_class_counts"].get("mid", 0)} mid,
{lab["fine_class_counts"].get("intern", 0)} intern and
{lab["fine_class_counts"].get("junior", 0)} junior postings, against a minimum of
{lab["min_confident_examples_per_class"]} per class (`config/labeling.yaml`). No label was
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
{reasons}

Labelled: {sum(lab["target_counts"].values())} ({", ".join(f"{c} {n}" for c, n in sorted(lab["target_counts"].items()))}).

**Splits** are stratified and grouped by identical input text, so the same job posted for several
cities never sits in both train and test. Seed {cfg["seed"]}.

| split | {" | ".join(labels)} |
|---|{"---|" * len(labels)}
{splits}

## Leakage controls

- The **title is not in the input**, because it is where most labels come from.
- **Every labeling-rule pattern is deleted** from the description before training and prediction.
- Only role sections are used (requirements first, then responsibilities, then intro). Company
  About, benefits and legal boilerplate are dropped, so the model can't key on employer templates.
- Imbalance is handled with **class weights** (in the loss for DistilBERT, `class_weight=balanced`
  for logistic regression). There is no oversampling and no generated text.

## Results (held-out test set, used once per model)

Macro-F1 has a bootstrap 95% CI (1,000 resamples of the real test predictions).

| model | accuracy | macro-F1 (95% CI) | {per_head} |
|---|---|---|{"---|" * len(labels)}
{table}

{_comparison(res.get("comparison"))}

{("### DistilBERT confusion matrix (test)" + chr(10) + chr(10) + _cm(tr["test"])) if tr else ""}

### TF-IDF + LR confusion matrix (test)

{_cm(res["baseline"]["test"])}

{("### DistilBERT training (CPU)" + chr(10) + chr(10) + "| epoch | train loss | val macro-F1 | seconds |" + chr(10) + "|---|---|---|---|" + chr(10) + hist) if tr else ""}

Settings: `{cfg["base_model"]}` @ `{cfg["base_model_revision"]}`, max {cfg["max_length"]}
tokens, batch {cfg["batch_size"]}, lr {cfg["learning_rate"]}, up to {cfg["epochs"]} epochs with
early stopping on validation macro-F1 (patience {cfg["patience"]}). The baseline's C was chosen
on validation macro-F1.

## Limitations

- **Small minority test set** ({lab["split_counts"]["test"].get("below_senior", 0)}
  below_senior postings), which is why the confidence intervals are wide. Read differences
  between models against those intervals.
- **Rule labels are noisy by construction.** "Lead" in operations titles (e.g. "Collections Team
  Lead") counts as senior. "Associate" counts as mid. A single seed was used and there was no
  hyperparameter search for the transformer.
- **One employer panel and one region.** The panel skews to product/SaaS companies in Bengaluru
  and may not transfer to other employers or markets.
- Inputs are truncated to {cfg["max_length"]} tokens, with requirements first.
- Labels say what the rules see in titles. The model learns textual correlates of those titles,
  not a ground truth of the job's real level.

## Files

- Weights: `models/seniority-distilbert/` (not in git; distributed separately). Baseline:
  `models/seniority-tfidf-logreg.joblib`.
- Labels with provenance and split: `data/processed/model/seniority_labels.parquet`.
- Metrics: `data/processed/model/seniority_metrics.json`.
- Load: `xray.classify.predict.SeniorityPredictor.load()`. CLI: `python -m xray predict --file jd.txt`.
"""
