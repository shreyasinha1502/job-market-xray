"""Data-quality panel: what the numbers rest on, per snapshot day, and every known gap."""

from __future__ import annotations

import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from xray.config import CONFIG_DIR, load_panel, load_sources
from xray.skills.pipeline import read_manifest
from xray.skills.vocab import load_vocab
from xray.store import Store
from xray.trends import history, load_frame


def build_quality(store: Store | None = None, config_dir: Path = CONFIG_DIR) -> dict[str, Any]:
    store = store or Store()
    frame = load_frame(store, config_dir)
    panel = load_panel(load_sources(config_dir), config_dir)
    days = []
    for d in frame.days:
        keys = frame.seen[d]
        ingest = store.read_coverage(d) if store.path("coverage", d).exists() else {}
        with_skill = sum(1 for k in keys if frame.skills.get(k))
        days.append({
            "day": d.isoformat(),
            "boards_ok": len(frame.board_ok.get(d, ())),
            "boards_failed": sorted(frame.board_failed.get(d, ())),
            "in_region_postings": len(keys),
            "stored": ingest.get("stored_today"),
            "skill_coverage": round(with_skill / len(keys), 4) if keys else None,
            "description_completeness": (ingest.get("field_completeness_in_region") or {}).get(
                "description"
            ),
        })  # fmt: skip

    gaps: list[str] = []
    hist = history(frame, frame.days)
    if hist["n_snapshots"] < 2:
        gaps.append(f"only {hist['n_snapshots']} snapshot day(s): no trend can be computed yet")
    if hist["missing_dates"]:
        gaps.append(f"{len(hist['missing_dates'])} calendar day(s) without a snapshot inside the "
                    f"history window: {', '.join(hist['missing_dates'])}")  # fmt: skip
    for row in days:
        if row["boards_failed"]:
            gaps.append(f"{row['day']}: boards failed {row['boards_failed']}; they are excluded "
                        "from trend comparisons that include this day")  # fmt: skip
    manifest = read_manifest(store)
    current = load_vocab(config_dir).sha256
    extractor = manifest.get("extractor") or {}
    if extractor.get("vocab_sha256") != current:
        gaps.append("skill extraction is stale for the current vocabulary: run `xray extract`")
    never_ok = sorted(
        {b.board for b in panel} - set().union(*frame.board_ok.values())
        if frame.board_ok
        else {b.board for b in panel}
    )
    if never_ok:
        gaps.append(f"panel boards never fetched OK: {never_ok}")

    age = None
    if frame.days:
        last = frame.days[-1]
        published = [frame.meta[k].published for k in frame.seen[last] if frame.meta[k].published]
        ages = [(datetime.combine(last, datetime.min.time()) - p).days for p in published]
        if ages:
            age = {
                "postings_with_publish_date": len(ages),
                "median_days_open": statistics.median(ages),
                "share_open_over_90_days": round(sum(a > 90 for a in ages) / len(ages), 4),
                "note": "long-open postings dominate stock counts; flow (new postings) is the "
                "cleaner demand signal once enough days accumulate",
            }
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "history": hist,
        "panel": {
            "boards": len(panel),
            "added_dates": sorted({b.added.isoformat() for b in panel}),
        },
        "extractor": extractor,
        "days": days,
        "posting_age_latest": age,
        "gaps": gaps,
    }
