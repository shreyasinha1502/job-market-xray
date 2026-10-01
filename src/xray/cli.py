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


def cmd_extract(args: argparse.Namespace) -> int:
    from xray.skills.pipeline import run_extraction
    from xray.skills.report import build_report, ner_candidates, write_report

    store = Store()
    manifest, parts = run_extraction(store=store, rebuild=args.rebuild)
    print(f"extractor {manifest['extractor']}")
    for p in parts:
        print(
            f"  postings/{p.partition}: {p.status:<10} "
            f"postings={p.n_postings} mentions={p.n_mentions}"
        )
    day = _resolve_day(store, None)
    report = build_report(store, day)
    rp, cp = write_report(store, day, report, ner_candidates(store, day))
    c = report["coverage"]
    print(f"\nsnapshot {day}: coverage (>=1 skill) {c['all_in_region']}")
    for role, v in c["by_role"].items():
        print(f"  {role:<28} {v['with_skill']:>4}/{v['postings']:<4} = {v['coverage']}")
    print(f"exclusions: {report['exclusions']['by_reason']}")
    print(f"wrote {rp}\nwrote {cp}")
    return 0


def cmd_skills(args: argparse.Namespace) -> int:
    from xray.skills.report import load_snapshot, skill_table

    store = Store()
    day = _resolve_day(store, args.date)
    posts, mentions, _ = load_snapshot(store, day)
    if args.role:
        roles = RoleMatcher.from_config(load_sources())
        posts = [p for p in posts if args.role in roles.match(p.title)]
    if args.city:
        posts = [p for p in posts if args.city in p.cities]
    table = skill_table(posts, mentions, scope=args.scope)
    scope = {
        "role": "in title/intro/responsibilities/requirements",
        "requirements": "in title/requirements",
        "anywhere": "anywhere incl. company boilerplate",
    }[args.scope]
    print(f"snapshot {day} ({day} only; not a trend): {len(posts)} postings, skill {scope}\n")
    print(
        f"{'skill':<16} {'category':<10} {'postings':>8} {'share':>6} {'boards':>6}  "
        "top board (share)"
    )
    for r in table[: args.top]:
        print(
            f"{r['skill']:<16} {r['category']:<10} {r['postings']:>8} {r['share']:>6.1%} "
            f"{r['boards']:>6}  {r['top_board']} ({r['top_board_share']:.0%})"
        )
    return 0


def cmd_mentions(args: argparse.Namespace) -> int:
    from xray.skills.report import _contexts, load_snapshot

    store = Store()
    day = _resolve_day(store, args.date)
    _, mentions, _ = load_snapshot(store, day)
    picked = [
        m
        for m in mentions
        if m.skill == args.skill.lower() and (m.excluded_reason is not None) == args.excluded
    ]
    print(
        f"{len(picked)} {'excluded' if args.excluded else 'counted'} mentions of {args.skill!r}\n"
    )
    for m, ctx in zip(picked[: args.limit], _contexts(store, picked[: args.limit]), strict=True):
        print(f"[{m.section}{'/' + m.excluded_reason if m.excluded_reason else ''}] …{ctx}…\n")
    return 0


def cmd_normalize(args: argparse.Namespace) -> int:
    from xray.skills.normalize_run import run_normalize

    store = Store()
    res = run_normalize(store, _resolve_day(store, args.date))
    meta, doc = res["meta"], res["doc"]
    print(
        f"snapshot {meta['snapshot']}: {meta['candidate_terms']} candidate terms, {meta['model']}"
    )
    cols = ("eps", "sim>=", "clusters", "terms", "conflicts", "no-lexical")
    print("\n" + " ".join(f"{c:>10}" for c in cols))
    for r in res["sensitivity"]:
        mark = "  <- chosen" if r["eps"] == meta["eps"] else ""
        vals = (
            r["eps"], r["min_similarity"], r["clusters"], r["terms_in_clusters"],
            r["multi_skill_conflicts"], r["merges_without_lexical_support"],
        )  # fmt: skip
        print(" ".join(f"{v:>10}" for v in vals) + mark)
    status = Counter(a["status"] for a in doc["aliases"])
    print(f"\nalias proposals: {dict(status)}  conflicts: {len(doc['conflicts'])}  "
          f"new skill groups proposed: {len(doc['new_skill_groups'])}")  # fmt: skip
    for f in res["files"]:
        print(f"wrote {f}")
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

    ext = sub.add_parser("extract", help="extract skills (incremental) + write skill report")
    ext.add_argument("--rebuild", action="store_true", help="re-extract every partition")
    ext.set_defaults(func=cmd_extract)

    sk = sub.add_parser("skills", help="top skills in one snapshot")
    sk.add_argument("--date", type=date.fromisoformat, default=None)
    sk.add_argument("--role", default=None)
    sk.add_argument("--city", default=None)
    sk.add_argument("--scope", choices=("role", "requirements", "anywhere"), default="role")
    sk.add_argument("--top", type=int, default=40)
    sk.set_defaults(func=cmd_skills)

    mn = sub.add_parser("mentions", help="show the text around each match of one skill (audit)")
    mn.add_argument("--skill", required=True)
    mn.add_argument("--excluded", action="store_true", help="show flagged/rejected matches")
    mn.add_argument("--date", type=date.fromisoformat, default=None)
    mn.add_argument("--limit", type=int, default=15)
    mn.set_defaults(func=cmd_mentions)

    nm = sub.add_parser("normalize", help="M3: embed + cluster skill terms -> skill_map.yaml")
    nm.add_argument("--date", type=date.fromisoformat, default=None)
    nm.set_defaults(func=cmd_normalize)

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
