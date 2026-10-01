"""Command-line entry point: `python -m xray <command>`."""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from collections import Counter
from datetime import date

from xray.config import (
    LOG_DIR,
    RAW_DIR,
    ConfigError,
    adzuna_credentials,
    enforce_no_synthetic_gates,
    load_sources,
)
from xray.fetch import FetchError, cache_raw, new_run_id
from xray.ingest import run_snapshot
from xray.log import setup_logging
from xray.rules import RoleMatcher
from xray.sources.adzuna import AdzunaClient, raw_page_path
from xray.store import Store

log = logging.getLogger("xray")


def _trunc(s: object, n: int) -> str:
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def cmd_ingest(args: argparse.Namespace) -> int:
    res = run_snapshot(snapshot_date=args.date)
    print(f"snapshot {res.snapshot_date}  run_id={res.run_id}")
    print(f"{'source':<10} {'board':<14} {'status':<6} {'http':>4} {'all':>5} {'IN':>4} {'new':>4}")
    for r in res.board_runs:
        print(
            f"{r.source:<10} {r.board:<14} {r.status:<6} {r.http_status or '-':>4} "
            f"{r.n_jobs_total:>5} {r.n_in_region:>4} {r.n_new:>4}"
        )
    cov = res.coverage
    print(f"\nboards ok {cov['panel']['ok']}/{cov['panel']['boards']}  jobs {cov['jobs']}")
    print(f"stored today: {cov['stored_today']}")
    for name, path in res.written.items():
        print(f"wrote {name:<10} {path}")
    if res.failed:
        print(
            f"\nFAILED boards: {[(r.source, r.board, r.error) for r in res.failed]}",
            file=sys.stderr,
        )
        return 1
    return 0


def _resolve_day(store: Store, day: date | None) -> date:
    dates = store.dates("sightings")
    if not dates:
        raise ConfigError("no snapshots stored yet; run `xray ingest` first")
    if day is None:
        return dates[-1]
    if day not in dates:
        raise ConfigError(f"no snapshot for {day}; have {[d.isoformat() for d in dates]}")
    return day


def cmd_show(args: argparse.Namespace) -> int:
    store = Store()
    day = _resolve_day(store, args.date)
    roles = RoleMatcher.from_config(load_sources())
    con = store.connect()
    rows = con.execute(
        """
        WITH latest AS (
            SELECT * FROM postings
            QUALIFY row_number() OVER (PARTITION BY posting_key ORDER BY first_seen_date DESC) = 1
        )
        SELECT p.source, p.board, p.title, p.cities, p.locations, p.published_at_utc,
               length(p.description) AS desc_chars, p.first_seen_date, p.raw_path, p.posting_key
        FROM sightings s JOIN latest p USING (posting_key, content_hash)
        WHERE s.snapshot_date = ?
        ORDER BY p.published_at_utc DESC NULLS LAST
        """,
        [day],
    ).fetchall()
    con.close()
    tagged = [(r, roles.match(r[2])) for r in rows]
    if args.role:
        tagged = [(r, rs) for r, rs in tagged if args.role in rs]
    if args.city:
        tagged = [(r, rs) for r, rs in tagged if args.city in r[3]]
    print(f"snapshot {day}: {len(rows)} in-region postings; {len(tagged)} after filters\n")
    print(f"{'board':<12} {'published':<10} {'title':<48} {'city':<14} {'roles':<26} {'desc':>5}")
    for r, rs in tagged[: args.limit]:
        city = ",".join(r[3]) or f"({_trunc(r[4][0] if r[4] else '', 12)})"
        pub = r[5].date().isoformat() if r[5] else "None"
        print(
            f"{_trunc(r[1], 12):<12} {pub:<10} {_trunc(r[2], 48):<48} {_trunc(city, 14):<14} "
            f"{_trunc(','.join(rs) or '-', 26):<26} {r[6] or 0:>5}"
        )
    if tagged:
        r = tagged[0][0]
        print(f"\nprovenance of first row: key={r[9]}  first_seen={r[7]}  raw={r[8]}")
    return 0


def cmd_coverage(args: argparse.Namespace) -> int:
    store = Store()
    day = _resolve_day(store, args.date)
    cov = store.read_coverage(day)
    if not args.full:
        cov.pop("skipped_records", None)
    print(json.dumps(cov, indent=2, ensure_ascii=False))
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    """Adzuna (dormant): one live call. Needs ADZUNA_APP_ID / ADZUNA_APP_KEY in .env."""
    app_id, app_key = adzuna_credentials()
    run_id = f"probe-{new_run_id()}"
    params: dict[str, str | int] = {args.mode: args.role, "results_per_page": args.n}
    if args.sort_by:
        params["sort_by"] = args.sort_by
    with AdzunaClient(app_id, app_key, max_retries=2) as client:
        raw = client.search(args.country, 1, params)
    cached = cache_raw(raw, raw_page_path(RAW_DIR, args.country, run_id, args.role, 1), "adzuna")
    print(f"HTTP {raw.status}  attempts={raw.attempts}  bytes={len(raw.body)}")
    print(f"endpoint   {raw.endpoint}")
    print(f"raw cached {cached.rel()}  sha256={cached.sha256[:16]}...")
    raw.raise_for_status()
    doc = json.loads(raw.body)
    results = doc.get("results") or []
    print(f"count={doc.get('count')}  len(results)={len(results)}")
    if results:
        keys = Counter(k for r in results for k, v in r.items() if v not in (None, "", [], {}))
        print("field presence:", dict(keys))
        lens = [len(r.get("description") or "") for r in results]
        print(f"description chars: median={statistics.median(lens)} max={max(lens)}")
        print(json.dumps(results[0], indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="xray", description="Job Market X-Ray")
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="command", required=True)

    ing = sub.add_parser("ingest", help="fetch today's snapshot for the whole panel")
    ing.add_argument("--date", type=date.fromisoformat, default=None, help="default: today UTC")
    ing.set_defaults(func=cmd_ingest)

    show = sub.add_parser("show", help="print stored postings for a snapshot")
    show.add_argument("--date", type=date.fromisoformat, default=None)
    show.add_argument("--role", default=None)
    show.add_argument("--city", default=None)
    show.add_argument("--limit", type=int, default=25)
    show.set_defaults(func=cmd_show)

    cov = sub.add_parser("coverage", help="print the coverage / gaps report for a snapshot")
    cov.add_argument("--date", type=date.fromisoformat, default=None)
    cov.add_argument("--full", action="store_true", help="include every skipped record")
    cov.set_defaults(func=cmd_coverage)

    probe = sub.add_parser("probe", help="Adzuna (dormant, needs keys): one live call")
    probe.add_argument("--role", default="data scientist")
    probe.add_argument("--country", default="in")
    probe.add_argument("--mode", choices=("what", "what_phrase", "title_only"), default="what")
    probe.add_argument("--n", type=int, default=50)
    probe.add_argument("--sort-by", choices=("date", "relevance", "salary"), default=None)
    probe.set_defaults(func=cmd_probe)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level, LOG_DIR / "xray.jsonl")
    try:
        enforce_no_synthetic_gates()
        return args.func(args)
    except (ConfigError, FetchError) as e:
        log.error(str(e))
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
