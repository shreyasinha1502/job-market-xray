"""Build the single JSON document the web frontend renders, from committed files only."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from xray.app.data import DashboardData, load
from xray.config import PROCESSED_DIR, load_yaml

REPO = "https://github.com/shreyasinha1502/job-market-xray"

CATEGORY_LABELS = {
    "languages": "Languages",
    "ml_dl_nlp": "ML & AI",
    "data_eng": "Data & databases",
    "cloud_ops": "Cloud & DevOps",
    "viz_bi": "BI & productivity",
    "web_backend": "Web & backend",
    "security_net": "Security & networking",
    "enterprise_apps": "Enterprise apps",
    "practices": "Practices",
}


def _scope_id(s: dict[str, Any]) -> tuple[str, str, str]:
    if s.get("role"):
        return f"role:{s['role']}", "role", s["role"].capitalize()
    if s.get("city"):
        return f"city:{s['city']}", "city", s["city"]
    return "all", "all", "All India postings"


def _model(metrics: dict[str, Any] | None) -> dict[str, Any] | None:
    if not metrics:
        return None

    def row(name: str, key: str, m: dict[str, Any], note: str = "") -> dict[str, Any]:
        return {
            "name": name, "key": key, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
            "ci": m["macro_f1_ci95_bootstrap"], "per_class": m["per_class"],
            "confusion": m["confusion_matrix"], "note": note,
        }  # fmt: skip

    results = [
        row('Always "senior"', "majority", metrics["majority_class_reference"]["test"],
            "the floor any model must beat"),
        row("TF-IDF + logistic regression", "baseline", metrics["baseline"]["test"]),
    ]  # fmt: skip
    if metrics.get("transformer"):
        results.append(row("DistilBERT (fp32)", "transformer", metrics["transformer"]["test"]))
    if metrics.get("serving_int8"):
        results.append(row("DistilBERT int8 (served)", "int8",
                           metrics["serving_int8"]["int8_test_metrics"],
                           "the model behind this site"))  # fmt: skip
    lab = metrics["labels"]
    return {
        "task": metrics["task"],
        "results": results,
        "comparison": metrics.get("comparison"),
        "parity": {k: metrics["serving_int8"][k] for k in ("agreement_with_fp32", "disagreements")}
        if metrics.get("serving_int8")
        else None,
        "labels": {
            "fine": lab["fine_class_counts"],
            "target": lab["target_counts"],
            "by_reason": lab["by_reason"],
            "split": lab.get("split_counts"),
            "min_per_class": lab["min_confident_examples_per_class"],
        },
        "history": (metrics.get("transformer") or {}).get("training", {}).get("history"),
    }


def build(d: DashboardData | None = None, root: Path = PROCESSED_DIR) -> dict[str, Any]:
    d = d or load(root)
    rep = d.skills_report or {}
    q = d.quality
    skill_cat: dict[str, str] = {}
    scopes, trends = [], {}
    for s in d.trends["scopes"]:
        sid, kind, label = _scope_id(s)
        snap = s["snapshot"]
        for r in snap["skills"]:
            skill_cat[r["skill"]] = r["category"]
        scopes.append({
            "id": sid, "kind": kind, "label": label, "n": snap["postings"], "day": snap["day"],
            "skills": [{k: r[k] for k in ("skill", "category", "postings", "share", "ci95")}
                       for r in snap["skills"]],
        })  # fmt: skip
        trends[sid] = s["trend"]
    skill_meta = {
        r["skill"]: {k: r[k] for k in ("boards", "top_board", "top_board_share",
                                       "in_requirements_or_title")}
        for r in rep.get("skills", [])
    }  # fmt: skip
    last = q["days"][-1] if q["days"] else {}
    cov = (rep.get("coverage") or {}).get("all_in_region") or {}
    age = q.get("posting_age_latest") or {}
    ingest = {}
    cov_path = root / "coverage" / f"{d.as_of}.json"
    if cov_path.exists():
        ingest = json.loads(cov_path.read_text("utf-8")).get("jobs", {})
    examples_path = root / "model" / "examples.json"
    examples = (
        json.loads(examples_path.read_text("utf-8"))["examples"] if examples_path.exists() else []
    )
    vocab_size = sum(len(v) for k, v in load_yaml("skills.yaml").items() if k in CATEGORY_LABELS)
    return {
        "as_of": d.as_of,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "history": d.history,
        "kpis": {
            "postings": last.get("in_region_postings"),
            "employers_ok": last.get("boards_ok"),
            "employers": q["panel"]["boards"],
            "jobs_scanned": ingest.get("returned_all_regions"),
            "skills_tracked": vocab_size,
            "coverage": cov.get("coverage"),
            "median_days_open": age.get("median_days_open"),
            "share_over_90": age.get("share_open_over_90_days"),
            "snapshot_days": d.history["n_snapshots"],
        },
        "categories": CATEGORY_LABELS,
        "skill_category": skill_cat,
        "skill_meta": skill_meta,
        "scopes": scopes,
        "trends": trends,
        "quality": {
            "days": q["days"],
            "gaps": q["gaps"],
            "age_histogram": age.get("histogram", []),
            "classifier_labels": q.get("classifier_labels"),
            "coverage_by_role": (rep.get("coverage") or {}).get("by_role", {}),
            "coverage_scopes": {
                k: (rep.get("coverage") or {}).get(k)
                for k in ("all_in_region", "requirements_only", "anywhere_incl_boilerplate")
            },
            "vocab_gaps": (rep.get("vocab_gaps_observed") or {}).get("terms", []),
            "exclusions": (rep.get("exclusions") or {}).get("by_reason", {}),
            "headings_share": (rep.get("sections") or {}).get("postings_with_recognized_headings"),
            "extractor": q.get("extractor"),
        },
        "model": _model(d.model_metrics),
        "examples": examples,
        "links": {
            "repo": REPO,
            "model_card": f"{REPO}/blob/main/data/processed/model/MODEL_CARD.md",
            "release": f"{REPO}/releases/tag/model-v1",
            "skill_map": f"{REPO}/blob/main/config/skill_map.yaml",
            "workflow": f"{REPO}/actions/workflows/daily.yml",
        },
    }
