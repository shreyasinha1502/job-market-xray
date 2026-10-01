"""M5: seniority classifier on rule-derived labels.

Baseline TF-IDF + logistic regression vs fine-tuned DistilBERT, same splits, same inputs.
- Splits are stratified and grouped by duplicate input text (the same job posted for several
  cities never straddles train/test).
- Imbalance is handled with class weights (loss and LR), never by oversampling or generating text.
- Model selection uses the validation split only; the test split is touched once per model.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from xray.classify.labels import (
    derive,
    dup_group,
    label_report,
    load_label_rules,
    model_input,
)
from xray.config import CONFIG_DIR, MODEL_DIR, ConfigError, load_yaml
from xray.log import kv
from xray.store import Store, write_parquet

log = logging.getLogger(__name__)

REPORT_DIR_NAME = "model"
LABEL_COLS = [
    ("posting_key", "VARCHAR"),
    ("content_hash", "VARCHAR"),
    ("board", "VARCHAR"),
    ("title", "VARCHAR"),
    ("label", "VARCHAR"),
    ("target", "VARCHAR"),
    ("reason", "VARCHAR"),
    ("evidence", "VARCHAR[]"),
    ("dup_group", "VARCHAR"),
    ("split", "VARCHAR"),
    ("input_chars", "INTEGER"),
]


@dataclass(frozen=True)
class TrainConfig:
    base_model: str
    base_model_revision: str | None
    test_size: float
    val_size: float
    max_length: int
    epochs: int
    learning_rate: float
    batch_size: int
    patience: int
    seed: int


def load_train_config(config_dir: Path = CONFIG_DIR) -> TrainConfig:
    c = load_yaml("project.yaml", config_dir)["classifier"]
    if c.get("baseline") != "tfidf_logreg":
        raise ConfigError("classifier.baseline must be 'tfidf_logreg'")
    return TrainConfig(
        base_model=c["base_model"],
        base_model_revision=c.get("base_model_revision"),
        test_size=float(c["test_size"]),
        val_size=float(c["val_size"]),
        max_length=int(c.get("max_length", 256)),
        epochs=int(c.get("epochs", 4)),
        learning_rate=float(c.get("learning_rate", 5e-5)),
        batch_size=int(c.get("batch_size", 8)),
        patience=int(c.get("patience", 1)),
        seed=int(c.get("seed", 42)),
    )


# ---------------------------------------------------------------- dataset


def build_dataset(store: Store, config_dir: Path = CONFIG_DIR) -> tuple[list[dict], dict]:
    """Latest version of every stored posting -> derived label + masked model input."""
    rules = load_label_rules(config_dir)
    con = store.connect()
    try:
        posts = con.execute(
            """SELECT posting_key, content_hash, board, title, description FROM postings
               QUALIFY row_number() OVER (PARTITION BY posting_key
                                          ORDER BY first_seen_date DESC) = 1
               ORDER BY posting_key"""
        ).fetchall()
    finally:
        con.close()
    rows = []
    for key, h, board, title, desc in posts:
        d = derive(title, desc, rules)
        text = model_input(desc, rules)
        reason = d.reason
        target = rules.target_of(d.label) if d.label else None
        if target is not None and not text:
            reason, target = "excluded:empty_model_input", None
        rows.append({
            "posting_key": key, "content_hash": h, "board": board, "title": title,
            "label": d.label, "target": target, "reason": reason, "evidence": list(d.evidence),
            "dup_group": dup_group(text), "text": text, "input_chars": len(text), "split": None,
        })  # fmt: skip
    # identical inputs that the rules labelled differently cannot be trusted either way
    targets_by_group: dict[str, set[str]] = {}
    for r in rows:
        if r["target"]:
            targets_by_group.setdefault(r["dup_group"], set()).add(r["target"])
    for r in rows:
        if r["target"] and len(targets_by_group[r["dup_group"]]) > 1:
            r["target"], r["reason"] = None, "excluded:conflicting_duplicates"
    return rows, label_report(rows, rules)


def assign_splits(rows: list[dict], cfg: TrainConfig) -> None:
    from sklearn.model_selection import StratifiedGroupKFold

    lab = [r for r in rows if r["target"]]
    y = np.array([r["target"] for r in lab])
    groups = np.array([r["dup_group"] for r in lab])
    n_test = round(1 / cfg.test_size)
    test_idx = next(StratifiedGroupKFold(n_test, shuffle=True, random_state=cfg.seed)
                    .split(np.zeros(len(lab)), y, groups))[1]  # fmt: skip
    rest = np.setdiff1d(np.arange(len(lab)), test_idx)
    n_val = round((1 - cfg.test_size) / cfg.val_size)
    val_rel = next(StratifiedGroupKFold(n_val, shuffle=True, random_state=cfg.seed)
                   .split(np.zeros(len(rest)), y[rest], groups[rest]))[1]  # fmt: skip
    test_set, val_set = set(test_idx.tolist()), set(rest[val_rel].tolist())
    for i, r in enumerate(lab):
        r["split"] = "test" if i in test_set else "val" if i in val_set else "train"
    leaks = {g for g in groups if len({r["split"] for r in lab if r["dup_group"] == g}) > 1}
    if leaks:
        raise AssertionError(f"duplicate groups straddle splits: {sorted(leaks)[:5]}")


# ---------------------------------------------------------------- metrics


def evaluate(y_true: list[str], y_pred: list[str], labels: list[str], seed: int) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    rep = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    rng = np.random.default_rng(seed)
    yt, yp = np.array(y_true), np.array(y_pred)
    boots = []
    for _ in range(1000):  # resamples the real test predictions; no new data
        idx = rng.integers(0, len(yt), len(yt))
        boots.append(f1_score(yt[idx], yp[idx], labels=labels, average="macro", zero_division=0))
    return {
        "n": len(y_true),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(rep["macro avg"]["f1-score"]), 4),
        "macro_f1_ci95_bootstrap": [round(float(np.percentile(boots, q)), 4) for q in (2.5, 97.5)],
        "weighted_f1": round(float(rep["weighted avg"]["f1-score"]), 4),
        "per_class": {
            c: {k: round(float(rep[c][k]), 4) for k in ("precision", "recall", "f1-score")}
            | {"support": int(rep[c]["support"])}
            for c in labels
        },
        "confusion_matrix": {
            "labels": labels,
            "rows_true_cols_pred": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        },
    }


# ---------------------------------------------------------------- baseline


def train_baseline(rows: list[dict], labels: list[str], cfg: TrainConfig, out: Path) -> dict:
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.pipeline import make_pipeline

    split = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    xt, yt = [r["text"] for r in split["train"]], [r["target"] for r in split["train"]]
    xv, yv = [r["text"] for r in split["val"]], [r["target"] for r in split["val"]]
    search = []
    for c in (0.1, 0.3, 1.0, 3.0, 10.0):
        pipe = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50000),
            LogisticRegression(C=c, class_weight="balanced", max_iter=5000),
        )
        pipe.fit(xt, yt)
        f1 = f1_score(yv, pipe.predict(xv), labels=labels, average="macro", zero_division=0)
        search.append({"C": c, "val_macro_f1": round(float(f1), 4)})
        log.info("baseline C", extra=kv(C=c, val_macro_f1=round(float(f1), 4)))
    best_c = max(search, key=lambda s: (s["val_macro_f1"], -s["C"]))["C"]
    pipe = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50000),
        LogisticRegression(C=best_c, class_weight="balanced", max_iter=5000),
    )
    pipe.fit(xt, yt)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out / "seniority-tfidf-logreg.joblib")
    preds = list(pipe.predict([r["text"] for r in split["test"]]))
    return {
        "model": "tfidf(1-2gram, min_df=2, sublinear) + logreg(class_weight=balanced)",
        "selection": {"grid": search, "chosen_C": best_c, "criterion": "val macro-F1"},
        "val": evaluate(yv, list(pipe.predict(xv)), labels, cfg.seed),
        "test": evaluate([r["target"] for r in split["test"]], preds, labels, cfg.seed),
    }


# ---------------------------------------------------------------- DistilBERT


def train_transformer(rows: list[dict], labels: list[str], cfg: TrainConfig, out: Path) -> dict:
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    torch.manual_seed(cfg.seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    split = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    lid = {c: i for i, c in enumerate(labels)}
    tok = AutoTokenizer.from_pretrained(cfg.base_model, revision=cfg.base_model_revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model, revision=cfg.base_model_revision, num_labels=len(labels),
        id2label=dict(enumerate(labels)), label2id=lid,
    )  # fmt: skip

    def encode(part: list[dict]):
        enc = tok([r["text"] for r in part], truncation=True, max_length=cfg.max_length,
                  padding="max_length", return_tensors="pt")  # fmt: skip
        y = torch.tensor([lid[r["target"]] for r in part])
        return enc["input_ids"], enc["attention_mask"], y

    data = {s: encode(p) for s, p in split.items()}
    counts = Counter(r["target"] for r in split["train"])
    n = sum(counts.values())
    weights = torch.tensor([n / (len(labels) * counts[c]) for c in labels], dtype=torch.float)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(split["train"]) / cfg.batch_size)
    sched = get_linear_schedule_with_warmup(
        opt, int(0.1 * steps_per_epoch * cfg.epochs), steps_per_epoch * cfg.epochs
    )

    def predict(part: str) -> list[str]:
        ids, mask, _ = data[part]
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(ids), 32):
                logits = model(input_ids=ids[i : i + 32], attention_mask=mask[i : i + 32]).logits
                preds += logits.argmax(-1).tolist()
        return [labels[p] for p in preds]

    from sklearn.metrics import f1_score

    history, best, best_f1, bad = [], None, -1.0, 0
    gen = torch.Generator().manual_seed(cfg.seed)
    ids, mask, y = data["train"]
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        order = torch.randperm(len(ids), generator=gen)
        for step, i in enumerate(range(0, len(ids), cfg.batch_size), 1):
            b = order[i : i + cfg.batch_size]
            logits = model(input_ids=ids[b], attention_mask=mask[b]).logits
            loss = loss_fn(logits, y[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            total += loss.item()
            if step % 10 == 0:
                log.info("train step", extra=kv(epoch=epoch, step=step, of=steps_per_epoch,
                                                loss=round(total / step, 4)))  # fmt: skip
        yv = [r["target"] for r in split["val"]]
        f1 = float(f1_score(yv, predict("val"), labels=labels, average="macro", zero_division=0))
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(total / steps_per_epoch, 4),
                "val_macro_f1": round(f1, 4),
                "seconds": round(time.time() - t0),
            }
        )
        log.info("epoch done", extra=kv(**history[-1]))  # fmt: skip
        if f1 > best_f1:
            best_f1, best, bad = f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad > cfg.patience:
                break
    model.load_state_dict(best)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    return {
        "model": f"{cfg.base_model}@{cfg.base_model_revision or 'unpinned'} fine-tuned",
        "training": {
            "max_length": cfg.max_length,
            "batch_size": cfg.batch_size,
            "lr": cfg.learning_rate,
            "epochs_max": cfg.epochs,
            "patience": cfg.patience,
            "class_weights": dict(zip(labels, [round(float(w), 4) for w in weights], strict=True)),
            "history": history,
            "best_val_macro_f1": round(best_f1, 4),
            "device": "cpu",
        },  # fmt: skip
        "val": evaluate([r["target"] for r in split["val"]], predict("val"), labels, cfg.seed),
        "test": evaluate([r["target"] for r in split["test"]], predict("test"), labels, cfg.seed),
    }


def _majority(rows: list[dict], labels: list[str], cfg: TrainConfig) -> dict:
    """Always predict the most frequent training class: the floor any model must beat."""
    top = Counter(r["target"] for r in rows if r["split"] == "train").most_common(1)[0][0]
    test = [r["target"] for r in rows if r["split"] == "test"]
    return {"predicts": top, "test": evaluate(test, [top] * len(test), labels, cfg.seed)}


# ---------------------------------------------------------------- orchestration


def run_training(
    store: Store | None = None,
    config_dir: Path = CONFIG_DIR,
    model_dir: Path = MODEL_DIR,
    skip_transformer: bool = False,
) -> dict[str, Any]:
    store = store or Store()
    cfg = load_train_config(config_dir)
    rows, report = build_dataset(store, config_dir)
    if not report["trainable"]:
        raise ConfigError(
            "refusing to train: confident examples per class below "
            f"{report['min_confident_examples_per_class']}: {report['target_counts']}. "
            "Scope down in labeling.yaml (class_groups) or wait for more real data."
        )
    assign_splits(rows, cfg)
    labels = sorted(report["target_counts"])
    report["split_counts"] = {
        s: dict(Counter(r["target"] for r in rows if r["split"] == s))
        for s in ("train", "val", "test")
    }
    rep_dir = store.root / REPORT_DIR_NAME
    write_parquet(rep_dir / "seniority_labels.parquet", LABEL_COLS, rows)
    results: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "task": "seniority: " + " vs ".join(labels),
        "config": asdict(cfg),
        "labels": report,
        "majority_class_reference": _majority([r for r in rows if r["split"]], labels, cfg),
        "baseline": train_baseline([r for r in rows if r["split"]], labels, cfg, model_dir),
    }
    if not skip_transformer:
        results["transformer"] = train_transformer(
            [r for r in rows if r["split"]], labels, cfg, model_dir / "seniority-distilbert"
        )
        results["comparison"] = compare_on_test(store, model_dir, config_dir, rows)
    (rep_dir / "seniority_metrics.json").write_text(
        json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    write_cards(results, rep_dir, model_dir)
    return results


PRED_COLS = [
    ("posting_key", "VARCHAR"),
    ("target", "VARCHAR"),
    ("baseline_pred", "VARCHAR"),
    ("transformer_pred", "VARCHAR"),
]


def compare_on_test(
    store: Store, model_dir: Path, config_dir: Path = CONFIG_DIR, rows: list[dict] | None = None
) -> dict[str, Any]:
    """Paired comparison on the same test postings: exact McNemar on discordant predictions."""
    from scipy.stats import binomtest

    from xray.classify.predict import SeniorityPredictor

    if rows is None:  # rebuild the identical, seeded split
        rows, _ = build_dataset(store, config_dir)
        assign_splits(rows, load_train_config(config_dir))
    test = [r for r in rows if r["split"] == "test"]
    bert = SeniorityPredictor.load(model_dir, config_dir, prefer="distilbert")
    base = SeniorityPredictor.load(model_dir, config_dir, prefer="tfidf_logreg")
    if bert.kind != "distilbert":
        raise ConfigError(f"no fine-tuned model in {model_dir / 'seniority-distilbert'}")
    preds = []
    for r in test:  # inputs are already prepared; bypass re-preparation
        preds.append({
            "posting_key": r["posting_key"], "target": r["target"],
            "baseline_pred": base.predict_prepared(r["text"])["prediction"],
            "transformer_pred": bert.predict_prepared(r["text"])["prediction"],
        })  # fmt: skip
    write_parquet(store.root / REPORT_DIR_NAME / "seniority_test_predictions.parquet",
                  PRED_COLS, preds)  # fmt: skip
    b_ok = [p["baseline_pred"] == p["target"] for p in preds]
    t_ok = [p["transformer_pred"] == p["target"] for p in preds]
    only_base = sum(b and not t for b, t in zip(b_ok, t_ok, strict=True))
    only_bert = sum(t and not b for b, t in zip(b_ok, t_ok, strict=True))
    p = binomtest(min(only_base, only_bert), only_base + only_bert, 0.5).pvalue if (
        only_base + only_bert) else 1.0  # fmt: skip
    return {
        "test_postings": len(preds),
        "both_correct": sum(b and t for b, t in zip(b_ok, t_ok, strict=True)),
        "only_baseline_correct": only_base,
        "only_transformer_correct": only_bert,
        "both_wrong": sum(not b and not t for b, t in zip(b_ok, t_ok, strict=True)),
        "mcnemar_exact_p": round(float(p), 4),
        "reading": "no significant difference at 0.05"
        if p >= 0.05
        else "significant difference at 0.05",
    }


def write_cards(results: dict[str, Any], rep_dir: Path, model_dir: Path) -> list[Path]:
    from xray.classify.card import model_card

    card = model_card(results)
    paths = [rep_dir / "MODEL_CARD.md"]
    if (model_dir / "seniority-distilbert").exists():
        paths.append(model_dir / "seniority-distilbert" / "README.md")
    for p in paths:
        p.write_text(card, encoding="utf-8")
    return paths
