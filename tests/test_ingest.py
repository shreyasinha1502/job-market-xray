"""End-to-end snapshot over real captured board responses served from fixtures.

Only the transport is local; every body is a real response. The second-day test re-observes the
same real responses under a later snapshot label to exercise the unchanged-posting path.
"""

import hashlib
from datetime import date
from pathlib import Path

import pytest
from helpers import BOARDS, CAPTURED, IN_REGION, count, fixture_transport, make_config

from xray.fetch import read_raw
from xray.ingest import run_snapshot
from xray.store import Store


@pytest.fixture
def run(tmp_path, fixtures_dir):
    store = Store(tmp_path / "processed")

    def _run(day, boards=BOARDS):
        return run_snapshot(
            snapshot_date=day,
            store=store,
            raw_root=tmp_path / "raw",
            config_dir=make_config(tmp_path / f"cfg-{day}-{len(boards)}", boards),
            transport=fixture_transport(fixtures_dir),
            sleep=lambda s: None,
        )

    return store, _run


def test_snapshot_stores_in_region_postings_with_provenance(run, fixtures_dir):
    store, go = run
    res = go(CAPTURED)
    assert not res.failed
    assert count(store, "select count(*) from postings") == IN_REGION
    assert count(store, "select count(*) from sightings") == IN_REGION
    assert (
        count(store, "select count(*) from postings where not list_contains(countries, 'in')") == 0
    )
    con = store.connect()
    rows = con.execute(
        "select distinct raw_path, raw_sha256, source, board from postings"
    ).fetchall()
    con.close()
    for raw_path, sha, source, board in rows:
        assert raw_path.endswith(".json.gz")
        cached = read_raw(Path(raw_path))
        assert hashlib.sha256(cached).hexdigest() == sha
        assert cached == (fixtures_dir / source / f"{board}.json").read_bytes()
    cov = store.read_coverage(CAPTURED)
    assert cov["jobs"]["in_region"] == IN_REGION and cov["jobs"]["out_of_region"] == 2
    assert cov["history"]["n_days"] == 1


def test_second_observation_records_sightings_not_duplicate_postings(run):
    store, go = run
    go(CAPTURED)
    second = go(date(2026, 10, 2))
    assert second.coverage["stored_today"] == {"new": 0, "changed": 0, "unchanged": IN_REGION}
    assert count(store, "select count(*) from sightings") == 2 * IN_REGION
    assert count(store, "select count(*) from postings") == IN_REGION
    assert second.coverage["history"]["n_days"] == 2


def test_same_day_rerun_is_idempotent(run):
    store, go = run
    go(CAPTURED)
    res = go(CAPTURED)
    assert res.coverage["stored_today"]["new"] == IN_REGION
    assert count(store, "select count(*) from postings") == IN_REGION


def test_failed_board_is_recorded_and_others_still_stored(run):
    store, go = run
    res = go(CAPTURED, [*BOARDS, ("greenhouse", "xray-nonexistent-board")])
    (failed,) = res.failed
    assert failed.board == "xray-nonexistent-board" and failed.http_status == 404
    assert "Job not found" in failed.error
    assert res.coverage["panel"]["ok"] == len(BOARDS)
    assert res.coverage["panel"]["failed"][0]["board"] == failed.board
    assert count(store, "select count(*) from postings") == IN_REGION
