"""End-to-end snapshot over real captured board responses served from fixtures.

Only the transport is local; every body is a real response. The second-day test re-observes the
same real responses under a later snapshot label to exercise the unchanged-posting path.
"""

import hashlib
import json
import shutil
from datetime import date
from pathlib import Path

import httpx
import pytest

from xray.config import CONFIG_DIR
from xray.fetch import read_raw
from xray.ingest import run_snapshot
from xray.store import Store

BOARDS = [("greenhouse", "groww"), ("lever", "cred"), ("lever", "epifi"), ("ashby", "atlan")]
CAPTURED = date(2026, 10, 1)
IN_REGION = 7 + 8 + 2 + 4  # groww all, cred all, epifi all, atlan 4 of 6


def make_config(tmp_path: Path, boards) -> Path:
    cfg = tmp_path / "config"
    shutil.copytree(CONFIG_DIR, cfg, dirs_exist_ok=True)
    lines = ["boards:"] + [f"  - {{source: {s}, board: {b}, added: 2026-10-01}}" for s, b in boards]
    (cfg / "panel.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return cfg


def fixture_transport(fixtures_dir: Path) -> httpx.MockTransport:
    hosts = {"boards-api.greenhouse.io": "greenhouse", "api.lever.co": "lever"}

    def handler(req: httpx.Request) -> httpx.Response:
        source = hosts.get(req.url.host, "ashby")
        parts = req.url.path.strip("/").split("/")
        board = parts[2] if source == "greenhouse" else parts[-1]  # /v1/boards/{b}/jobs
        body = fixtures_dir / source / f"{board}.json"
        if not body.exists():
            nf = (fixtures_dir / "greenhouse" / "not_found_404.json").read_bytes()
            return httpx.Response(404, content=nf, headers={"Content-Type": "application/json"})
        meta = json.loads((fixtures_dir / source / f"{board}.meta.json").read_text("utf-8"))
        return httpx.Response(
            200, content=body.read_bytes(), headers={"Content-Type": meta["content_type"]}
        )

    return httpx.MockTransport(handler)


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


def count(store: Store, sql: str) -> int:
    con = store.connect()
    try:
        return con.execute(sql).fetchone()[0]
    finally:
        con.close()


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
