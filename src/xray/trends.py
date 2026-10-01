"""M4 trend engine.

Two views, never mixed:

  snapshot  point-in-time skill shares among postings open on one day (stock).
  trend     change across accumulated snapshots; only when >= `min_snapshots` real days exist.
    - stock change: share on the last vs first day of the window, over a balanced panel (boards
      fetched OK on every day of the window and in the panel since its start). Descriptive only:
      the same postings stay open across days, so day samples overlap and a test would be invalid.
    - flow test: skill share among postings first seen in the early vs late half of the window.
      These sets are disjoint, so Fisher's exact test + Benjamini-Hochberg apply. Postings first
      seen on the very first snapshot are the panel's backlog, not new demand, and are excluded.
      Too little history or too few new postings -> reported as insufficient, never guessed.

Every output carries the real history window it rests on.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from xray.config import CONFIG_DIR, load_panel, load_sources, load_yaml
from xray.rules import RoleMatcher
from xray.skills.report import ROLE_SECTIONS
from xray.store import Store

Key = tuple[str, str]  # (posting_key, content_hash)


@dataclass(frozen=True)
class TrendConfig:
    min_snapshots: int
    flow_min_days: int
    flow_min_new_per_half: int
    min_postings: int
    fdr_q: float
    top_k: int
    min_city_postings: int


def load_trend_config(config_dir: Path = CONFIG_DIR) -> TrendConfig:
    t = load_yaml("project.yaml", config_dir).get("trends") or {}
    return TrendConfig(
        min_snapshots=int(t.get("min_snapshots", 2)),
        flow_min_days=int(t.get("flow_min_days", 14)),
        flow_min_new_per_half=int(t.get("flow_min_new_per_half", 30)),
        min_postings=int(t.get("min_postings", 10)),
        fdr_q=float(t.get("fdr_q", 0.05)),
        top_k=int(t.get("top_k", 10)),
        min_city_postings=int(t.get("min_city_postings", 50)),
    )


@dataclass(frozen=True)
class PostMeta:
    board: str
    title: str
    cities: tuple[str, ...]
    first_seen: date
    published: datetime | None


@dataclass
class Frame:
    days: list[date]
    seen: dict[date, set[Key]]
    meta: dict[Key, PostMeta]
    skills: dict[Key, frozenset[str]]
    board_ok: dict[date, set[str]]
    board_failed: dict[date, set[str]]
    board_added: dict[str, date]
    category: dict[str, str]


def load_frame(store: Store, config_dir: Path = CONFIG_DIR) -> Frame:
    con = store.connect()
    try:
        sightings = con.execute(
            "SELECT snapshot_date, posting_key, content_hash FROM sightings"
        ).fetchall()
        posts = con.execute(
            """SELECT posting_key, content_hash, any_value(board), any_value(title),
                      any_value(cities), min(min(first_seen_date)) OVER (PARTITION BY posting_key),
                      any_value(published_at_utc)
               FROM postings GROUP BY posting_key, content_hash"""
        ).fetchall()
        sections = ", ".join(f"'{s}'" for s in sorted(ROLE_SECTIONS))
        mentions = con.execute(
            "SELECT posting_key, content_hash, skill, category FROM skill_mentions "
            f"WHERE excluded_reason IS NULL AND section IN ({sections})"
        ).fetchall()
        runs = con.execute("SELECT snapshot_date, board, status FROM board_runs").fetchall()
    finally:
        con.close()
    seen: dict[date, set[Key]] = defaultdict(set)
    for d, k, h in sightings:
        seen[d].add((k, h))
    meta = {(k, h): PostMeta(b, t, tuple(c or ()), fs, pub) for k, h, b, t, c, fs, pub in posts}
    skills: dict[Key, set[str]] = defaultdict(set)
    category: dict[str, str] = {}
    for k, h, s, cat in mentions:
        skills[(k, h)].add(s)
        category[s] = cat
    ok: dict[date, set[str]] = defaultdict(set)
    failed: dict[date, set[str]] = defaultdict(set)
    for d, b, st in runs:
        (ok if st == "ok" else failed)[d].add(b)
    added = {b.board: b.added for b in load_panel(load_sources(config_dir), config_dir)}
    return Frame(
        days=sorted(seen),
        seen=dict(seen),
        meta=meta,
        skills={k: frozenset(v) for k, v in skills.items()},
        board_ok=dict(ok),
        board_failed=dict(failed),
        board_added=added,
        category=category,
    )


# ---------------------------------------------------------------- scopes


@dataclass(frozen=True)
class Scope:
    role: str | None = None
    city: str | None = None

    @property
    def label(self) -> str:
        parts = [f"role={self.role}" if self.role else "", f"city={self.city}" if self.city else ""]
        return ", ".join(p for p in parts if p) or "all in-region postings"


def _in_scope(m: PostMeta, scope: Scope, roles: RoleMatcher) -> bool:
    if scope.city and scope.city not in m.cities:
        return False
    return not scope.role or scope.role in roles.match(m.title)


def history(frame: Frame, days: list[date]) -> dict[str, Any]:
    if not days:
        return {"n_snapshots": 0, "first": None, "last": None, "span_days": 0, "missing_dates": []}
    have = set(days)
    span = (days[-1] - days[0]).days
    missing = [
        (days[0] + timedelta(i)).isoformat()
        for i in range(1, span)
        if days[0] + timedelta(i) not in have
    ]
    return {
        "n_snapshots": len(days),
        "first": days[0].isoformat(),
        "last": days[-1].isoformat(),
        "span_days": span,
        "missing_dates": missing,
    }


def balanced_boards(frame: Frame, days: list[date]) -> tuple[set[str], dict[str, str]]:
    """Boards usable for comparing `days`: fetched OK on every one of them, in panel from start."""
    boards = set().union(*(frame.board_ok.get(d, set()) | frame.board_failed.get(d, set())
                           for d in days))  # fmt: skip
    included, excluded = set(), {}
    for b in sorted(boards):
        missed = [d.isoformat() for d in days if b not in frame.board_ok.get(d, set())]
        added = frame.board_added.get(b)
        if missed:
            excluded[b] = f"not fetched OK on {', '.join(missed)}"
        elif added is not None and added > days[0]:
            excluded[b] = f"added to panel on {added.isoformat()}, after window start"
        else:
            included.add(b)
    return included, excluded


def _counts(frame: Frame, keys: set[Key]) -> tuple[int, Counter[str]]:
    c: Counter[str] = Counter()
    for k in keys:
        c.update(frame.skills.get(k, ()))
    return len(keys), c


def _open_on(frame: Frame, day: date, boards: set[str] | None, scope: Scope,
             roles: RoleMatcher) -> set[Key]:  # fmt: skip
    return {
        k
        for k in frame.seen.get(day, set())
        if (boards is None or frame.meta[k].board in boards)
        and _in_scope(frame.meta[k], scope, roles)
    }


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% Wilson interval for a share: honest width for small n (16 postings is not 1,315)."""
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)


def snapshot_table(frame: Frame, day: date, scope: Scope, roles: RoleMatcher) -> dict[str, Any]:
    boards = frame.board_ok.get(day, set())
    n, c = _counts(frame, _open_on(frame, day, boards, scope, roles))
    rows = [
        {"skill": s, "category": frame.category.get(s), "postings": v,
         "share": round(v / n, 4) if n else None, "ci95": wilson(v, n)}
        for s, v in c.most_common()
    ]  # fmt: skip
    return {
        "day": day.isoformat(),
        "scope": scope.label,
        "postings": n,
        "view": "point-in-time snapshot (one day); not a trend",
        "ci_note": "Wilson 95% interval over this panel's postings, not the whole market",
        "skills": rows,
    }


def _fisher(a: int, n1: int, b: int, n2: int) -> float:
    from scipy.stats import fisher_exact

    return float(fisher_exact([[a, n1 - a], [b, n2 - b]]).pvalue)


def _bh(pvals: list[float]) -> list[float]:
    if not pvals:
        return []
    from scipy.stats import false_discovery_control

    return [float(q) for q in false_discovery_control(np.asarray(pvals), method="bh")]


def flow_test(frame: Frame, days: list[date], boards: set[str], scope: Scope,
              roles: RoleMatcher, cfg: TrendConfig) -> dict[str, Any]:  # fmt: skip
    backlog_day = frame.days[0]  # everything "new" on the first snapshot is pre-existing stock
    span = (days[-1] - days[0]).days
    new = {
        k: m.first_seen
        for k, m in frame.meta.items()
        if m.board in boards
        and days[0] <= m.first_seen <= days[-1]
        and m.first_seen != backlog_day
        and _in_scope(m, scope, roles)
    }
    mid = days[0] + timedelta(days=span / 2)
    early = {k for k, d in new.items() if d <= mid}
    late = {k for k, d in new.items() if d > mid}
    base = {"new_postings": len(new), "early": len(early), "late": len(late),
            "split_at": mid.isoformat()}  # fmt: skip
    if span < cfg.flow_min_days:
        return {
            **base,
            "status": "insufficient_history",
            "reason": f"window spans {span} days; flow test needs >= {cfg.flow_min_days}",
        }
    if min(len(early), len(late)) < cfg.flow_min_new_per_half:
        return {
            **base,
            "status": "insufficient_new_postings",
            "reason": f"needs >= {cfg.flow_min_new_per_half} new postings in each half",
        }
    n1, c1 = _counts(frame, early)
    n2, c2 = _counts(frame, late)
    tested = [s for s in set(c1) | set(c2) if c1[s] + c2[s] >= cfg.min_postings]
    pvals = [_fisher(c1[s], n1, c2[s], n2) for s in tested]
    rows = []
    for s, p, q in zip(tested, pvals, _bh(pvals), strict=True):
        rows.append({
            "skill": s, "early_share": round(c1[s] / n1, 4), "late_share": round(c2[s] / n2, 4),
            "delta_pp": round(100 * (c2[s] / n2 - c1[s] / n1), 2), "p": round(p, 5),
            "q_bh": round(q, 5), "significant": q < cfg.fdr_q,
        })  # fmt: skip
    return {**base, "status": "ok", "tested_skills": len(tested), "fdr_q": cfg.fdr_q,
            "skills": sorted(rows, key=lambda r: r["q_bh"])}  # fmt: skip


def trend(frame: Frame, scope: Scope, roles: RoleMatcher, cfg: TrendConfig,
          window_days: int | None = None) -> dict[str, Any]:  # fmt: skip
    days = frame.days
    if window_days is not None and days:
        days = [d for d in days if d > days[-1] - timedelta(days=window_days)]
    hist = history(frame, days)
    out: dict[str, Any] = {"scope": scope.label, "history": hist}
    if len(days) < cfg.min_snapshots:
        return {**out, "status": "insufficient_history",
                "reason": f"{len(days)} snapshot day(s); a trend needs >= {cfg.min_snapshots}. "
                          "Showing nothing rather than a guess."}  # fmt: skip

    boards, excluded = balanced_boards(frame, days)
    series = {d: _counts(frame, _open_on(frame, d, boards, scope, roles)) for d in days}
    (n0, c0), (n1, c1) = series[days[0]], series[days[-1]]
    x = np.array([(d - days[0]).days for d in days], dtype=float)
    rows = []
    for s in sorted(set(c0) | set(c1)):
        if max(c0[s], c1[s]) < cfg.min_postings:
            continue
        sh = np.array([series[d][1][s] / series[d][0] if series[d][0] else np.nan for d in days])
        slope = (
            float(np.polyfit(x, sh * 100, 1)[0])
            if len(days) >= 3 and not np.isnan(sh).any()
            else None
        )
        rows.append({
            "skill": s, "category": frame.category.get(s),
            "first_share": round(c0[s] / n0, 4) if n0 else None,
            "last_share": round(c1[s] / n1, 4) if n1 else None,
            "delta_pp": round(100 * (c1[s] / n1 - c0[s] / n0), 2) if n0 and n1 else None,
            "first_postings": c0[s], "last_postings": c1[s],
            "slope_pp_per_day": None if slope is None else round(slope, 3),
            "series": [None if np.isnan(v) else round(float(v), 4) for v in sh],
        })  # fmt: skip
    flow = flow_test(frame, days, boards, scope, roles, cfg)
    sig = {r["skill"]: r for r in flow.get("skills", [])}
    for r in rows:
        f = sig.get(r["skill"])
        r["flow_significant"] = bool(f and f["significant"])
    ranked = [r for r in rows if r["delta_pp"] is not None]
    risers = sorted((r for r in ranked if r["delta_pp"] > 0), key=lambda r: -r["delta_pp"])
    fallers = sorted((r for r in ranked if r["delta_pp"] < 0), key=lambda r: r["delta_pp"])
    return {
        **out,
        "status": "ok",
        "panel": {"boards_compared": len(boards), "boards_excluded": excluded},
        "stock": {"first_day_postings": n0, "last_day_postings": n1,
                  "days": [d.isoformat() for d in days],
                  "postings_per_day": [series[d][0] for d in days],
                  "note": "descriptive: same postings stay open across days, so no p-values"},
        "risers": risers[: cfg.top_k],
        "fallers": fallers[: cfg.top_k],
        "skills": rows,
        "flow": flow,
    }  # fmt: skip


def scopes(frame: Frame, roles: RoleMatcher, cfg: TrendConfig) -> list[Scope]:
    out = [Scope()] + [Scope(role=r) for r in roles.names]
    if frame.days:
        cities = Counter(c for k in frame.seen[frame.days[-1]] for c in frame.meta[k].cities)
        out += [Scope(city=c) for c, n in cities.most_common() if n >= cfg.min_city_postings]
    return out


def build_trends(store: Store | None = None, config_dir: Path = CONFIG_DIR,
                 window_days: int | None = None) -> dict[str, Any]:  # fmt: skip
    store = store or Store()
    frame = load_frame(store, config_dir)
    cfg = load_trend_config(config_dir)
    roles = RoleMatcher.from_config(load_sources(config_dir))
    if not frame.days:
        raise ValueError("no snapshots stored yet")
    as_of = frame.days[-1]
    return {
        "as_of": as_of.isoformat(),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "history": history(frame, frame.days),
        "config": cfg.__dict__,
        "scopes": [
            {"scope": sc.label, "role": sc.role, "city": sc.city,
             "snapshot": snapshot_table(frame, as_of, sc, roles),
             "trend": trend(frame, sc, roles, cfg, window_days)}
            for sc in scopes(frame, roles, cfg)
        ],
    }  # fmt: skip


def write_json(store: Store, folder: str, day: str, doc: dict[str, Any]) -> Path:
    out = store.root / folder / f"{day}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return out
