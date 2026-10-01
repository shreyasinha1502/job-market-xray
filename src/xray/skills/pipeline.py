"""Run skill extraction over stored postings, one postings partition at a time.

A partition is re-extracted when its postings file changed or the extractor changed (vocab,
spaCy/model version, or EXTRACTOR_VERSION). `_manifest.json` records what produced each output.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from xray.config import CONFIG_DIR, load_regions, load_sources
from xray.log import kv
from xray.skills.extract import PostingText, SkillExtractor
from xray.skills.vocab import load_vocab
from xray.store import Store

log = logging.getLogger(__name__)


def location_terms(config_dir: Path = CONFIG_DIR) -> list[str]:
    sources = load_sources(config_dir)
    terms: list[str] = []
    for rule in load_regions(sources, config_dir).values():
        terms += rule.country_aliases + [rule.name, rule.iso2]
        for city, aliases in rule.cities.items():
            terms += [city, *aliases]
    return terms


@dataclass(frozen=True)
class PartitionResult:
    partition: str
    status: str  # extracted | up_to_date
    n_postings: int
    n_mentions: int


def _manifest_path(store: Store) -> Path:
    return store.root / "skill_mentions" / "_manifest.json"


def read_manifest(store: Store) -> dict:
    p = _manifest_path(store)
    return json.loads(p.read_text("utf-8")) if p.exists() else {}


def run_extraction(
    *,
    store: Store | None = None,
    config_dir: Path = CONFIG_DIR,
    rebuild: bool = False,
    extractor: SkillExtractor | None = None,
) -> tuple[dict, list[PartitionResult]]:
    store = store or Store()
    extractor = extractor or SkillExtractor(
        load_vocab(config_dir), location_terms=location_terms(config_dir)
    )
    xid = extractor.extractor_id
    manifest = read_manifest(store)
    same_extractor = manifest.get("extractor") == xid
    parts_meta: dict = manifest.get("partitions", {}) if same_extractor else {}
    results: list[PartitionResult] = []

    for part in store.files("postings"):
        day = date.fromisoformat(part.stem)
        sha = hashlib.sha256(part.read_bytes()).hexdigest()
        prev = parts_meta.get(part.stem)
        outputs = [
            store.path(t, day) for t in ("skill_mentions", "entity_mentions", "extraction_meta")
        ]
        if (
            not rebuild
            and prev
            and prev["postings_sha256"] == sha
            and all(o.exists() for o in outputs)
        ):
            results.append(
                PartitionResult(part.stem, "up_to_date", prev["n_postings"], prev["n_mentions"])
            )
            continue

        con = store.connect()
        try:
            rows = con.execute(
                "SELECT posting_key, content_hash, title, description, board, company_name "
                "FROM read_parquet(?)",
                [part.as_posix()],
            ).fetchall()
        finally:
            con.close()
        mentions, entities, meta = [], [], []
        for ex in extractor.extract(PostingText(*r) for r in rows):
            key = {"posting_key": ex.posting.posting_key, "content_hash": ex.posting.content_hash}
            mentions += [{**key, **asdict(m)} for m in ex.mentions]
            entities += [{**key, **asdict(e)} for e in ex.entities]
            meta.append({**key, "n_headings": ex.n_headings})
        store.write_day("skill_mentions", day, mentions)
        store.write_day("entity_mentions", day, entities)
        store.write_day("extraction_meta", day, meta)
        parts_meta[part.stem] = {
            "postings_sha256": sha,
            "n_postings": len(rows),
            "n_mentions": len(mentions),
            "extracted_at": datetime.now(UTC).isoformat(),
        }
        results.append(PartitionResult(part.stem, "extracted", len(rows), len(mentions)))
        log.info(
            "partition extracted",
            extra=kv(partition=part.stem, postings=len(rows), mentions=len(mentions)),
        )

    manifest = {"extractor": xid, "partitions": parts_meta}
    path = _manifest_path(store)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest, results
