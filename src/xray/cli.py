"""Command-line entry point: `python -m xray <command>`."""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from collections import Counter

from xray.config import (
    LOG_DIR,
    RAW_DIR,
    ConfigError,
    adzuna_credentials,
    enforce_no_synthetic_gates,
)
from xray.log import setup_logging
from xray.sources.adzuna import AdzunaClient, AdzunaError, cache_raw, new_run_id, raw_page_path

log = logging.getLogger("xray")

QUERY_MODES = ("what", "what_phrase", "title_only")


def cmd_probe(args: argparse.Namespace) -> int:
    """One live call. Caches the raw response and prints what actually came back."""
    app_id, app_key = adzuna_credentials()
    run_id = f"probe-{new_run_id()}"
    params: dict[str, str | int] = {args.mode: args.role, "results_per_page": args.n}
    if args.sort_by:
        params["sort_by"] = args.sort_by
    with AdzunaClient(app_id, app_key, max_retries=2) as client:
        raw = client.search(args.country, 1, params)
    cached = cache_raw(raw, raw_page_path(RAW_DIR, args.country, run_id, args.role, 1))

    print(f"HTTP {raw.status}  attempts={raw.attempts}  bytes={len(raw.body)}")
    print(f"endpoint   {raw.endpoint}")
    print(f"raw cached {cached.rel()}  sha256={cached.sha256[:16]}...")
    raw.raise_for_status()

    doc = json.loads(raw.body)
    results = doc.get("results") or []
    print(f"top-level keys: {sorted(doc)}")
    print(f"count={doc.get('count')}  mean={doc.get('mean')}  len(results)={len(results)}")
    if not results:
        return 0

    keys = Counter(k for r in results for k, v in r.items() if v not in (None, "", [], {}))
    print("\nfield presence (non-empty) across results:")
    for k in sorted({k for r in results for k in r}):
        print(f"  {k:<22} {keys[k]:>3}/{len(results)}")

    lens = [len(r.get("description") or "") for r in results]
    print(f"\ndescription chars: min={min(lens)} median={statistics.median(lens)} max={max(lens)}")

    print("\nrank | created              | title | company | location")
    for i, r in enumerate(results, 1):
        print(
            f"{i:>4} | {r.get('created', '?'):<20} | {r.get('title')} | "
            f"{(r.get('company') or {}).get('display_name')} | "
            f"{(r.get('location') or {}).get('display_name')}"
        )
    print("\nfirst result, raw JSON exactly as returned:")
    print(json.dumps(results[0], indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="xray", description="Job Market X-Ray")
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="command", required=True)

    probe = sub.add_parser("probe", help="one live API call; print the real raw sample")
    probe.add_argument("--role", default="data scientist")
    probe.add_argument("--country", default="in")
    probe.add_argument("--mode", choices=QUERY_MODES, default="what")
    probe.add_argument("--n", type=int, default=50, help="results_per_page")
    probe.add_argument("--sort-by", choices=("date", "relevance", "salary"), default=None)
    probe.set_defaults(func=cmd_probe)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level, LOG_DIR / "xray.jsonl")
    try:
        enforce_no_synthetic_gates()
        return args.func(args)
    except (ConfigError, AdzunaError) as e:
        log.error(str(e))
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
