"""M3 normalization: lexical rules, real-embedding clustering behaviour, reviewed-map persistence.

Strings below are real surface forms observed in the 2026-10-01 snapshot (see
data/processed/skill_map/clusters_2026-10-01.json); the pairs are the ones the threshold choice
was argued from.
"""

from collections import Counter

import pytest
import yaml
from helpers import BOARDS, CAPTURED, fixture_transport, make_config

from xray.ingest import run_snapshot
from xray.skills.candidates import _list_items, lexical_key
from xray.skills.normalize import (
    acronym_of,
    dbscan,
    lexically_supported,
    load_norm_config,
)
from xray.skills.normalize_run import run_normalize
from xray.skills.pipeline import run_extraction
from xray.skills.vocab import load_vocab
from xray.store import Store


@pytest.mark.parametrize(
    ("a", "b", "same_key"),
    [
        ("React.js", "ReactJS", True),
        ("Node.js", "NodeJS", True),
        ("React", "ReactJS", False),  # merged only via embeddings + lexical support
        ("SAs", "SAS", False),  # Solutions Architects (plural) vs the SAS language
        ("PoCs", "POC", True),
        ("Next.js", "Next", False),
    ],
)
def test_lexical_key(a, b, same_key):
    assert (lexical_key(a) == lexical_key(b)) is same_key


@pytest.mark.parametrize(
    ("a", "b", "supported"),
    [
        ("React", "React.js", True),
        ("PostgreSQL", "Postgres", True),
        ("OAuth", "OAuth 2.0", True),
        ("MS Excel", "Excel", True),
        ("SASE", "SAST", False),
        ("Excel", "Google Sheets", False),
        ("Linux", "Linux systems", False),
    ],
)
def test_lexical_support(a, b, supported):
    assert lexically_supported(a, b) is supported


def test_acronyms():
    assert acronym_of("Amazon Web Services") == "aws"
    assert acronym_of("Google Cloud Platform") == "gcp"
    assert acronym_of("Python") is None


def test_list_items_keep_ab_and_split_languages():
    line = "- Coding experience in Python, R, Java, Apache Spark™ or Scala"  # databricks posting
    assert _list_items(line) == ["Python", "R", "Java", "Apache Spark™", "Scala"]
    assert "A/B testing" in _list_items("- Experience with A/B testing, SQL, Python/R")


@pytest.fixture(scope="module")
def embedder():
    from xray.skills.candidates import Term
    from xray.skills.normalize import embed

    cfg = load_norm_config()

    def run(strings):
        return embed([Term(lexical_key(s), surfaces=Counter({s: 1})) for s in strings], cfg)[0]

    return run, cfg


def test_threshold_merges_variants_but_not_lookalike_acronyms(embedder):
    run, cfg = embedder
    strings = ["Excel", "MS Excel", "Microsoft Excel", "Google Sheets", "PostgreSQL", "Postgres",
               "SASE", "SAST", "PyTorch", "TensorFlow"]  # fmt: skip
    labels = dict(zip(strings, dbscan(run(strings), cfg.eps, cfg.min_samples), strict=True))
    assert labels["Excel"] == labels["MS Excel"] == labels["Microsoft Excel"] != -1
    assert labels["PostgreSQL"] == labels["Postgres"] != -1
    assert labels["Google Sheets"] != labels["Excel"]
    assert labels["SASE"] == -1 or labels["SASE"] != labels["SAST"]
    assert labels["PyTorch"] == -1 or labels["PyTorch"] != labels["TensorFlow"]


def test_reviewed_decisions_survive_regeneration_and_only_accepted_aliases_apply(
    tmp_path, fixtures_dir
):
    store = Store(tmp_path / "processed")
    cfg = make_config(tmp_path, BOARDS)
    (cfg / "skill_map.yaml").unlink(missing_ok=True)
    run_snapshot(snapshot_date=CAPTURED, store=store, raw_root=tmp_path / "raw", config_dir=cfg,
                 transport=fixture_transport(fixtures_dir), sleep=lambda s: None)  # fmt: skip
    run_extraction(store=store, config_dir=cfg)
    first = run_normalize(store, CAPTURED, cfg)
    assert (cfg / "skill_map.yaml").exists()
    assert first["doc"]["conflicts"] == []

    doc = yaml.safe_load((cfg / "skill_map.yaml").read_text("utf-8"))
    doc["aliases"].append(
        {"variant": "Py Torch", "skill": "pytorch", "status": "rejected", "reviewed": True,
         "note": "test review"}
    )  # fmt: skip
    (cfg / "skill_map.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")
    run_normalize(store, CAPTURED, cfg)
    again = yaml.safe_load((cfg / "skill_map.yaml").read_text("utf-8"))
    kept = [a for a in again["aliases"] if a["variant"] == "Py Torch"]
    assert kept and kept[0]["status"] == "rejected" and kept[0]["note"] == "test review"
    assert kept[0].get("stale") is True  # not observed in this snapshot, but never silently lost
    assert "py torch" not in load_vocab(cfg).lower_forms["pytorch"]
