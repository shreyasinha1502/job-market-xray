"""M5 labels / splits / gate / predictor on real postings.

Title cases are real titles from tests/fixtures (postings_sample.jsonl + captured boards). Split
and predictor tests run on a temp copy of the committed real snapshot in data/processed.
"""

import json
import shutil
from pathlib import Path

import pytest
from helpers import BOARDS, CAPTURED, fixture_transport, make_config

from xray.classify.labels import derive, load_label_rules, model_input
from xray.classify.predict import SeniorityPredictor
from xray.classify.train import (
    assign_splits,
    build_dataset,
    evaluate,
    load_train_config,
    run_training,
)
from xray.config import PROCESSED_DIR, ConfigError
from xray.ingest import run_snapshot
from xray.skills.sections import tag_sections
from xray.store import Store

SAMPLE = Path(__file__).parent / "fixtures" / "postings_sample.jsonl"


def samples() -> dict[str, dict]:
    recs = [json.loads(line) for line in SAMPLE.read_text("utf-8").splitlines()]
    return {r["title"]: r for r in recs}


@pytest.mark.parametrize(
    ("title", "label", "reason"),
    [
        ("Sr. Staff Software Development Engineer - Golang + Control Plane", "senior",
         "labeled_from_title"),
        ("Staff Data Analyst, People Analytics", "senior", "labeled_from_title"),
        ("Trainee - Recruitment Coordinator", "intern", "labeled_from_title"),
        ("Account Director, Mid-Market", None, "excluded:management_title"),
    ],
)  # fmt: skip
def test_labels_from_real_titles(title, label, reason):
    rec = samples()[title]
    d = derive(rec["title"], rec["description"], load_label_rules())
    assert (d.label, d.reason) == (label, reason)


def test_word_cues_in_description_are_not_used():
    # okta's description talks about "senior" stakeholders etc.; only the title may decide
    rules = load_label_rules()
    rec = samples()["Renewals Operations Analyst II"]
    d = derive(rec["title"], rec["description"], rules)
    assert d.reason in {"excluded:no_signal", "labeled_from_description_years",
                        "excluded:ambiguous_description"}  # fmt: skip
    assert all("year" in e for e in d.evidence)


def test_model_input_hides_label_evidence_and_title():
    rules = load_label_rules()
    for rec in samples().values():
        text = model_input(rec["description"], rules)
        assert text, rec["posting_key"]
        assert rules.any_rule.search(text) is None, rec["posting_key"]


def test_model_input_puts_requirements_first():
    rules = load_label_rules()
    rec = samples()["Staff Data Analyst, People Analytics"]
    secs = tag_sections(rec["description"])
    first_req = secs.starts[secs.labels.index("requirements")]
    heading = rec["description"][first_req:].splitlines()[0].strip()
    assert model_input(rec["description"], rules).splitlines()[0].strip() == heading


def test_training_refuses_below_minimum_examples(tmp_path, fixtures_dir):
    store = Store(tmp_path / "processed")
    run_snapshot(
        snapshot_date=CAPTURED,
        store=store,
        raw_root=tmp_path / "raw",
        config_dir=make_config(tmp_path, BOARDS),
        transport=fixture_transport(fixtures_dir),
        sleep=lambda s: None,
    )
    with pytest.raises(ConfigError, match="refusing to train"):
        run_training(store=store, model_dir=tmp_path / "models", skip_transformer=True)


@pytest.fixture(scope="module")
def real_store(tmp_path_factory):
    """Temp copy of the committed real snapshot, so training outputs never touch the repo."""
    root = tmp_path_factory.mktemp("real") / "processed"
    for table in ("postings", "sightings", "board_runs"):
        shutil.copytree(PROCESSED_DIR / table, root / table)
    return Store(root)


def test_splits_never_share_a_duplicate_group(real_store):
    rows, report = build_dataset(real_store)
    assert report["trainable"]
    assign_splits(rows, load_train_config())
    lab = [r for r in rows if r["target"]]
    by_group: dict[str, set[str]] = {}
    for r in lab:
        by_group.setdefault(r["dup_group"], set()).add(r["split"])
    assert all(len(s) == 1 for s in by_group.values())
    share = sum(r["split"] == "test" for r in lab) / len(lab)
    assert 0.15 < share < 0.25


def test_baseline_trains_and_predictor_loads(real_store, tmp_path):
    res = run_training(store=real_store, model_dir=tmp_path, skip_transformer=True)
    assert res["baseline"]["test"]["n"] > 0
    assert res["majority_class_reference"]["predicts"] == "senior"
    pred = SeniorityPredictor.load(model_dir=tmp_path, prefer="tfidf_logreg")
    out = pred.predict(samples()["Staff Data Analyst, People Analytics"]["description"])
    assert out["prediction"] in {"senior", "below_senior"}
    assert abs(sum(out["probabilities"].values()) - 1) < 1e-6


def test_evaluate_matches_hand_computed_macro_f1():
    m = evaluate(["a", "a", "b"], ["a", "b", "b"], ["a", "b"], seed=0)
    assert m["macro_f1"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["confusion_matrix"]["rows_true_cols_pred"] == [[1, 1], [0, 1]]
