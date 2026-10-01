"""Daily snapshot: fetch every panel board, cache raw, normalize, region-filter, store, report."""

from __future__ import annotations

import hashlib
import json
import logging
import statistics
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import ValidationError

from xray.config import (
    CONFIG_DIR,
    RAW_DIR,
    PanelBoard,
    load_panel,
    load_regions,
    load_sources,
)
from xray.fetch import Fetcher, FetchError, cache_raw, new_run_id
from xray.log import kv
from xray.rules import RegionMatcher, RoleMatcher
from xray.schemas import BoardRun, Posting, Provenance, Sighting, SkippedRecord
from xray.sources.ats import ADAPTERS, ParsedJob
from xray.store import Store, board_run_row, posting_row, sighting_row

log = logging.getLogger(__name__)

HASHED_FIELDS = (
    "title", "description", "company_name", "locations", "department", "team",
    "employment_type", "workplace_type", "published_at", "url",
)  # fmt: skip
COMPLETENESS_FIELDS = (
    "description", "company_name", "cities", "department", "team", "employment_type",
    "workplace_type", "published_at", "updated_at", "url",
)  # fmt: skip


def content_hash(job: ParsedJob) -> str:
    payload = {f: getattr(job, f) for f in HASHED_FIELDS}
    payload["published_at"] = job.published_at.isoformat() if job.published_at else None
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


@dataclass
class _BoardOutcome:
    run: BoardRun
    observed: list[tuple[Posting, str]] = field(default_factory=list)  # (posting, new|changed|...)
    skipped: list[SkippedRecord] = field(default_factory=list)


@dataclass(frozen=True)
class SnapshotResult:
    snapshot_date: date
    run_id: str
    board_runs: list[BoardRun]
    coverage: dict[str, Any]
    written: dict[str, Path]

    @property
    def failed(self) -> list[BoardRun]:
        return [r for r in self.board_runs if r.status == "failed"]


def _ingest_board(
    b: PanelBoard,
    fetcher: Fetcher,
    *,
    run_id: str,
    day: date,
    raw_root: Path,
    matcher: RegionMatcher,
    prev: dict[str, str],
) -> _BoardOutcome:
    adapter = ADAPTERS[b.source]
    url = adapter.url(b.board)
    base = dict(run_id=run_id, snapshot_date=day, source=b.source, board=b.board)
    zero = dict(
        n_jobs_total=0, n_skipped=0, n_in_region=0, n_out_of_region=0,
        n_new=0, n_changed=0, n_unchanged=0,
    )  # fmt: skip

    def failed(error: str, raw=None, cached=None) -> _BoardOutcome:
        log.error("board failed", extra=kv(source=b.source, board=b.board, error=error))
        return _BoardOutcome(
            BoardRun(
                **base,
                endpoint=raw.endpoint if raw else str(httpx.URL(url, params=adapter.params)),
                fetched_at=raw.fetched_at if raw else None,
                http_status=raw.status if raw else None,
                attempts=raw.attempts if raw else None,
                raw_path=cached.rel() if cached else None,
                raw_sha256=cached.sha256 if cached else None,
                status="failed",
                error=error,
                **zero,
            )
        )

    try:
        raw = fetcher.get(url, adapter.params)
    except FetchError as e:
        return failed(str(e))
    cached = cache_raw(raw, raw_root / b.source / run_id / b.board, b.source)
    if not raw.ok:
        return failed(
            f"HTTP {raw.status}: {raw.body[:200].decode('utf-8', 'replace')}", raw, cached
        )
    try:
        jobs, skips = adapter.parse(json.loads(raw.body))
    except ValueError as e:  # includes JSONDecodeError
        return failed(f"parse error: {e}", raw, cached)

    prov = Provenance(
        source=b.source,
        run_id=run_id,
        query=b.board,
        endpoint=raw.endpoint,
        page=None,
        fetched_at=raw.fetched_at,
        raw_path=cached.rel(),
        raw_sha256=cached.sha256,
    )
    observed: list[tuple[Posting, str]] = []
    seen: set[str] = set()
    out_of_region = 0
    for job in jobs:
        if job.source_job_id in seen:
            skips.append((job.source_job_id, "duplicate_id_in_response"))
            continue
        seen.add(job.source_job_id)
        matches = matcher.match(job.locations, job.structured_countries)
        if not matches:
            if not job.locations and not job.structured_countries:
                skips.append((job.source_job_id, "no_location"))
            else:
                out_of_region += 1
            continue
        key = f"{b.source}:{b.board}:{job.source_job_id}"
        h = content_hash(job)
        try:
            posting = Posting(
                posting_key=key,
                source=b.source,
                board=b.board,
                source_job_id=job.source_job_id,
                title=job.title,
                description=job.description,
                company_name=job.company_name,
                locations=job.locations,
                countries=[m.country for m in matches],
                country_evidence=[m.evidence for m in matches],
                cities=sorted({c for m in matches for c in m.cities}),
                department=job.department,
                team=job.team,
                employment_type=job.employment_type,
                workplace_type=job.workplace_type,
                published_at=job.published_at,
                updated_at=job.updated_at,
                url=job.url,
                content_hash=h,
                provenance=prov,
            )
        except ValidationError as e:
            skips.append((job.source_job_id, f"invalid: {e.errors()[0]['msg']}"))
            continue
        status = "new" if key not in prev else ("unchanged" if prev[key] == h else "changed")
        observed.append((posting, status))

    skipped = [
        SkippedRecord(run_id=run_id, source=b.source, board=b.board, source_job_id=jid, reason=r)
        for jid, r in skips
    ]
    counts = Counter(s for _, s in observed)
    run = BoardRun(
        **base,
        endpoint=raw.endpoint,
        fetched_at=raw.fetched_at,
        http_status=raw.status,
        attempts=raw.attempts,
        raw_path=cached.rel(),
        raw_sha256=cached.sha256,
        status="ok",
        error=None,
        n_jobs_total=len(jobs) + sum(1 for _, r in skips if r in ("missing_id", "missing_title")),
        n_skipped=len(skips),
        n_in_region=len(observed),
        n_out_of_region=out_of_region,
        n_new=counts["new"],
        n_changed=counts["changed"],
        n_unchanged=counts["unchanged"],
    )
    log.info(
        "board ingested",
        extra=kv(source=b.source, board=b.board, in_region=len(observed), new=counts["new"]),
    )
    return _BoardOutcome(run, observed, skipped)


def _coverage(
    *,
    day: date,
    run_id: str,
    panel: list[PanelBoard],
    outcomes: list[_BoardOutcome],
    roles: RoleMatcher,
    history_dates: list[date],
    config_files: dict[str, str],
) -> dict[str, Any]:
    runs = [o.run for o in outcomes]
    observed = [p for o in outcomes for p, _ in o.observed]
    statuses = Counter(s for o in outcomes for _, s in o.observed)
    skipped = [s for o in outcomes for s in o.skipped]
    n = len(observed)

    def filled(p: Posting, f: str) -> bool:
        v = getattr(p, f)
        return v not in (None, "", [])

    desc_lens = sorted(len(p.description) for p in observed if p.description)
    role_hits = [roles.match(p.title) for p in observed]
    role_counts = Counter(r for hits in role_hits for r in hits)
    city_counts = Counter(c for p in observed for c in p.cities)
    return {
        "snapshot_date": day.isoformat(),
        "run_id": run_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "panel": {
            "boards": len(panel),
            "ok": sum(r.status == "ok" for r in runs),
            "failed": [
                {"source": r.source, "board": r.board, "error": r.error}
                for r in runs
                if r.status == "failed"
            ],
        },
        "jobs": {
            "returned_all_regions": sum(r.n_jobs_total for r in runs),
            "in_region": n,
            "out_of_region": sum(r.n_out_of_region for r in runs),
            "skipped": len(skipped),
            "skip_reasons": dict(Counter(s.reason for s in skipped)),
        },
        "stored_today": {k: statuses[k] for k in ("new", "changed", "unchanged")},
        "field_completeness_in_region": {
            f: round(sum(filled(p, f) for p in observed) / n, 4) if n else None
            for f in COMPLETENESS_FIELDS
        },
        "description_chars": (
            {
                "min": desc_lens[0],
                "median": statistics.median(desc_lens),
                "p10": desc_lens[len(desc_lens) // 10],
                "max": desc_lens[-1],
            }
            if desc_lens
            else None
        ),
        "roles": {
            "counts": dict(role_counts),
            "untagged": sum(not h for h in role_hits),
            "multi_tagged": sum(len(h) > 1 for h in role_hits),
            "tagged_share": round(sum(bool(h) for h in role_hits) / n, 4) if n else None,
        },
        "cities": {
            "counts": dict(city_counts.most_common()),
            "country_level_only": sum(not p.cities for p in observed),
        },
        "config_sha256": config_files,
        "history": {
            "snapshot_dates": [d.isoformat() for d in history_dates],
            "n_days": len(history_dates),
        },
        "skipped_records": [s.model_dump() for s in skipped],
        "notes": [
            "Sampling frame is the fixed employer panel in config/panel.yaml, "
            "not the whole market.",
            "first_seen_date on the first snapshot marks the panel start, not when a job was "
            "posted; use published_at for posting dates.",
        ],
    }


def run_snapshot(
    *,
    snapshot_date: date | None = None,
    store: Store | None = None,
    raw_root: Path = RAW_DIR,
    config_dir: Path = CONFIG_DIR,
    transport: httpx.BaseTransport | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> SnapshotResult:
    sources = load_sources(config_dir)
    panel = load_panel(sources, config_dir)
    matcher = RegionMatcher(load_regions(sources, config_dir))
    roles = RoleMatcher.from_config(sources)
    store = store or Store()
    now = datetime.now(UTC)
    run_id = new_run_id(now)
    day = snapshot_date or now.date()
    prev = store.latest_hashes(before=day)
    log.info("snapshot start", extra=kv(run_id=run_id, day=day, boards=len(panel)))

    fetchers = {
        src: Fetcher(
            min_interval_sec=sources.rate_limit_sleep_sec,
            max_retries=4,
            transport=transport,
            sleep=sleep,
        )
        for src in sorted({b.source for b in panel})
    }
    outcomes: list[_BoardOutcome] = []
    try:
        for b in panel:
            outcomes.append(
                _ingest_board(
                    b,
                    fetchers[b.source],
                    run_id=run_id,
                    day=day,
                    raw_root=raw_root,
                    matcher=matcher,
                    prev=prev,
                )  # fmt: skip
            )
    finally:
        for f in fetchers.values():
            f.close()

    observed = [(p, s) for o in outcomes for p, s in o.observed]
    written = {
        "postings": store.write_day(
            "postings", day, [posting_row(p, day) for p, s in observed if s != "unchanged"]
        ),
        "sightings": store.write_day(
            "sightings",
            day,
            [
                sighting_row(
                    Sighting(
                        snapshot_date=day,
                        posting_key=p.posting_key,
                        content_hash=p.content_hash,
                        run_id=run_id,
                    )
                )
                for p, _ in observed
            ],
        ),
        "board_runs": store.write_day("board_runs", day, [board_run_row(o.run) for o in outcomes]),
    }
    report = _coverage(
        day=day,
        run_id=run_id,
        panel=panel,
        outcomes=outcomes,
        roles=roles,
        history_dates=store.dates("sightings"),
        config_files={
            f: hashlib.sha256((config_dir / f).read_bytes()).hexdigest()
            for f in ("sources.yaml", "regions.yaml", sources.panel_file)
        },
    )
    written["coverage"] = store.write_coverage(day, report)
    result = SnapshotResult(day, run_id, [o.run for o in outcomes], report, written)
    log.info(
        "snapshot done",
        extra=kv(run_id=run_id, in_region=len(observed), failed_boards=len(result.failed)),
    )
    return result
