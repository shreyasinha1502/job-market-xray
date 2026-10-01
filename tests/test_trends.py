"""Trend engine + data-quality panel over real captured responses.

Multi-day tests re-observe the same real board responses under later snapshot labels, and the
failure case serves the real Greenhouse 404 body for one board on one day. The observation
schedule is test setup; every posting, skill and count comes from real fixtures.
"""

from datetime import date

import pytest
from helpers import BOARDS, fixture_transport, make_config

from xray.config import load_sources
from xray.ingest import run_snapshot
from xray.quality import build_quality
from xray.rules import RoleMatcher
from xray.skills.pipeline import run_extraction
from xray.store import Store
from xray.trends import (
    Scope,
    _fisher,
    load_frame,
    load_trend_config,
    snapshot_table,
    trend,
    wilson,
)

D1, D2, D3 = date(2026, 10, 1), date(2026, 10, 2), date(2026, 10, 3)


@pytest.fixture
def observe(tmp_path, fixtures_dir):
    store = Store(tmp_path / "processed")
    cfg = make_config(tmp_path, BOARDS)
    project = cfg / "project.yaml"
    # rank every skill in this 21-posting fixture panel (production default is 10)
    project.write_text(
        project.read_text("utf-8").replace("min_postings: 10 ", "min_postings: 1 "), "utf-8"
    )

    def run(*days, fail_on=None):
        for d in days:
            fail = frozenset(fail_on.get(d, ())) if fail_on else frozenset()
            run_snapshot(
                snapshot_date=d,
                store=store,
                raw_root=tmp_path / "raw",
                config_dir=cfg,
                transport=fixture_transport(fixtures_dir, fail),
                sleep=lambda s: None,
            )
        run_extraction(store=store, config_dir=cfg)
        frame = load_frame(store, cfg)
        roles = RoleMatcher.from_config(load_sources(cfg))
        return store, cfg, frame, roles, load_trend_config(cfg)

    return run


def test_single_snapshot_gives_snapshot_but_no_trend(observe):
    store, cfg, frame, roles, tcfg = observe(D1)
    t = trend(frame, Scope(), roles, tcfg)
    assert t["status"] == "insufficient_history" and t["history"]["n_snapshots"] == 1
    snap = snapshot_table(frame, D1, Scope(), roles)
    con = store.connect()
    expected = dict(
        con.execute(
            "SELECT skill, count(DISTINCT posting_key) FROM skill_mentions WHERE excluded_reason "
            "IS NULL AND section IN ('title','intro','responsibilities','requirements') GROUP BY 1"
        ).fetchall()
    )
    con.close()
    assert {r["skill"]: r["postings"] for r in snap["skills"]} == expected
    assert snap["postings"] == 21
    assert all(r["ci95"][0] <= r["share"] <= r["ci95"][1] for r in snap["skills"])


def test_identical_observations_show_zero_change_and_no_flow(observe):
    _, _, frame, roles, tcfg = observe(D1, D2)
    t = trend(frame, Scope(), roles, tcfg)
    assert t["status"] == "ok" and t["history"]["n_snapshots"] == 2
    assert t["skills"] and all(r["delta_pp"] == 0 for r in t["skills"])
    assert t["risers"] == [] and t["fallers"] == []
    # every posting was first seen on the backlog day, so there is no new-posting flow
    assert t["flow"]["new_postings"] == 0 and t["flow"]["status"] == "insufficient_history"


def test_failed_board_is_excluded_from_comparison_not_read_as_a_drop(observe):
    store, cfg, frame, roles, tcfg = observe(D1, D2, fail_on={D2: {"groww"}})
    t = trend(frame, Scope(), roles, tcfg)
    assert "groww" in t["panel"]["boards_excluded"]
    assert all(r["delta_pp"] == 0 for r in t["skills"])
    assert t["stock"]["first_day_postings"] == t["stock"]["last_day_postings"] == 21 - 7
    gaps = build_quality(store, cfg)["gaps"]
    assert any("groww" in g for g in gaps)


def test_missing_days_are_reported(observe):
    store, cfg, frame, roles, tcfg = observe(D1, D3)
    t = trend(frame, Scope(), roles, tcfg)
    assert t["history"]["missing_dates"] == ["2026-10-02"]
    assert any("2026-10-02" in g for g in build_quality(store, cfg)["gaps"])
    assert trend(frame, Scope(), roles, tcfg, window_days=1)["status"] == "insufficient_history"


def test_role_scope_filters_by_title_rules(observe):
    _, _, frame, roles, _ = observe(D1)
    snap = snapshot_table(frame, D1, Scope(role="data scientist"), roles)
    assert snap["postings"] == 1  # epifi "DS/ML Intern"


def test_statistics_helpers():
    # Fisher's lady-tasting-tea table: two-sided p = 34/70
    assert _fisher(3, 4, 1, 4) == pytest.approx(34 / 70)
    assert wilson(0, 0) is None
    lo, hi = wilson(22, 27)
    assert lo < 22 / 27 < hi and hi - lo > 0.25  # small n -> wide interval
