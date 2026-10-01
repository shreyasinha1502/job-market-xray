"""Extraction pipeline + snapshot report over a real snapshot built from captured responses."""

import pytest
from helpers import BOARDS, CAPTURED, IN_REGION, fixture_transport, make_config

from xray.config import ConfigError
from xray.ingest import run_snapshot
from xray.skills.pipeline import run_extraction
from xray.skills.report import build_report, ner_candidates
from xray.store import Store


@pytest.fixture
def snap(tmp_path, fixtures_dir):
    store = Store(tmp_path / "processed")
    cfg = make_config(tmp_path, BOARDS)
    run_snapshot(
        snapshot_date=CAPTURED,
        store=store,
        raw_root=tmp_path / "raw",
        config_dir=cfg,
        transport=fixture_transport(fixtures_dir),
        sleep=lambda s: None,
    )
    return store, cfg


def test_extraction_is_incremental_and_tracks_vocab_changes(snap):
    store, cfg = snap
    first, parts = run_extraction(store=store, config_dir=cfg)
    assert [(p.partition, p.status, p.n_postings) for p in parts] == [
        (CAPTURED.isoformat(), "extracted", IN_REGION)
    ]
    _, parts = run_extraction(store=store, config_dir=cfg)
    assert [p.status for p in parts] == ["up_to_date"]

    skills = cfg / "skills.yaml"
    skills.write_text(skills.read_text("utf-8") + "\n# edited\n", encoding="utf-8")
    second, parts = run_extraction(store=store, config_dir=cfg)
    assert [p.status for p in parts] == ["extracted"]
    assert second["extractor"]["vocab_sha256"] != first["extractor"]["vocab_sha256"]


def test_report_refuses_to_run_before_extraction(snap):
    store, cfg = snap
    with pytest.raises(ConfigError, match="run `xray extract`"):
        build_report(store, CAPTURED, cfg)


def test_report_numbers_are_consistent(snap):
    store, cfg = snap
    run_extraction(store=store, config_dir=cfg)
    report = build_report(store, CAPTURED, cfg)
    cov = report["coverage"]["all_in_region"]
    assert cov["postings"] == IN_REGION and 0 <= cov["with_skill"] <= IN_REGION
    assert all(0 < r["postings"] <= cov["with_skill"] for r in report["skills"])
    assert all(r["in_requirements_or_title"] <= r["postings"] for r in report["skills"])
    by_role = report["coverage"]["by_role"]
    assert by_role["data scientist"]["postings"] == 1  # epifi "DS/ML Intern"
    assert sum(v["postings"] for k, v in by_role.items() if k != "(no tracked role)") >= 1
    assert set(report["vocab"]["never_seen_in_snapshot"]).isdisjoint(
        {r["skill"] for r in report["skills"]}
    )
    for c in ner_candidates(store, CAPTURED, min_boards=1):
        assert c["postings"] >= 1 and c["boards"] >= 1


def test_coverage_scopes_are_nested(snap):
    store, cfg = snap
    run_extraction(store=store, config_dir=cfg)
    cov = build_report(store, CAPTURED, cfg)["coverage"]
    req, role, anywhere = (
        cov[k]["with_skill"]
        for k in ("requirements_only", "all_in_region", "anywhere_incl_boilerplate")
    )
    assert req <= role <= anywhere
