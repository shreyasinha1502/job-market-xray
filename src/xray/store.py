"""Day-partitioned parquet tables under data/processed, written and read with DuckDB.

    postings/<date>.parquet    postings first seen, or whose content changed, on <date>
    sightings/<date>.parquet   every in-region posting observed on <date> (stock)
    board_runs/<date>.parquet  one row per board fetch, including failures
    coverage/<date>.json       data-quality report for the snapshot

Small append-only daily files keep git history cheap: an unchanged posting costs one sighting row
per day, not a re-stored description. Timestamps are stored as naive UTC (`*_utc` columns).
"""

from __future__ import annotations

import json
import os
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import duckdb

from xray.config import PROCESSED_DIR
from xray.schemas import BoardRun, Posting, Sighting

SCHEMAS: dict[str, list[tuple[str, str]]] = {
    "postings": [
        ("posting_key", "VARCHAR"),
        ("first_seen_date", "DATE"),
        ("source", "VARCHAR"),
        ("board", "VARCHAR"),
        ("source_job_id", "VARCHAR"),
        ("title", "VARCHAR"),
        ("description", "VARCHAR"),
        ("company_name", "VARCHAR"),
        ("locations", "VARCHAR[]"),
        ("countries", "VARCHAR[]"),
        ("country_evidence", "VARCHAR[]"),
        ("cities", "VARCHAR[]"),
        ("department", "VARCHAR"),
        ("team", "VARCHAR"),
        ("employment_type", "VARCHAR"),
        ("workplace_type", "VARCHAR"),
        ("published_at_utc", "TIMESTAMP"),
        ("updated_at_utc", "TIMESTAMP"),
        ("url", "VARCHAR"),
        ("content_hash", "VARCHAR"),
        ("run_id", "VARCHAR"),
        ("endpoint", "VARCHAR"),
        ("fetched_at_utc", "TIMESTAMP"),
        ("raw_path", "VARCHAR"),
        ("raw_sha256", "VARCHAR"),
    ],
    "sightings": [
        ("snapshot_date", "DATE"),
        ("posting_key", "VARCHAR"),
        ("content_hash", "VARCHAR"),
        ("run_id", "VARCHAR"),
    ],
    "board_runs": [
        ("run_id", "VARCHAR"),
        ("snapshot_date", "DATE"),
        ("source", "VARCHAR"),
        ("board", "VARCHAR"),
        ("endpoint", "VARCHAR"),
        ("fetched_at_utc", "TIMESTAMP"),
        ("http_status", "INTEGER"),
        ("attempts", "INTEGER"),
        ("raw_path", "VARCHAR"),
        ("raw_sha256", "VARCHAR"),
        ("status", "VARCHAR"),
        ("error", "VARCHAR"),
        ("n_jobs_total", "INTEGER"),
        ("n_skipped", "INTEGER"),
        ("n_in_region", "INTEGER"),
        ("n_out_of_region", "INTEGER"),
        ("n_new", "INTEGER"),
        ("n_changed", "INTEGER"),
        ("n_unchanged", "INTEGER"),
    ],
}


def utc_naive(dt: datetime | None) -> datetime | None:
    return None if dt is None else dt.astimezone(UTC).replace(tzinfo=None)


def posting_row(p: Posting, first_seen: date) -> dict[str, Any]:
    d = p.model_dump(exclude={"provenance", "published_at", "updated_at"})
    pv = p.provenance
    return {
        **d,
        "first_seen_date": first_seen,
        "published_at_utc": utc_naive(p.published_at),
        "updated_at_utc": utc_naive(p.updated_at),
        "run_id": pv.run_id,
        "endpoint": pv.endpoint,
        "fetched_at_utc": utc_naive(pv.fetched_at),
        "raw_path": pv.raw_path,
        "raw_sha256": pv.raw_sha256,
    }


def sighting_row(s: Sighting) -> dict[str, Any]:
    return s.model_dump()


def board_run_row(r: BoardRun) -> dict[str, Any]:
    d = r.model_dump(exclude={"fetched_at"})
    return {**d, "fetched_at_utc": utc_naive(r.fetched_at)}


def _sql_str(path: Path) -> str:
    return "'" + path.as_posix().replace("'", "''") + "'"


class Store:
    def __init__(self, root: Path = PROCESSED_DIR) -> None:
        self.root = root

    def path(self, table: str, day: date) -> Path:
        ext = "json" if table == "coverage" else "parquet"
        return self.root / table / f"{day.isoformat()}.{ext}"

    def files(self, table: str) -> list[Path]:
        return sorted((self.root / table).glob("*.parquet"))

    def dates(self, table: str) -> list[date]:
        return [date.fromisoformat(p.stem) for p in self.files(table)]

    def write_day(self, table: str, day: date, rows: list[dict[str, Any]]) -> Path:
        cols = SCHEMAS[table]
        path = self.path(table, day)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        con = duckdb.connect()
        try:
            con.execute(f"CREATE TABLE t ({', '.join(f'{c} {t}' for c, t in cols)})")
            if rows:
                con.executemany(
                    f"INSERT INTO t VALUES ({', '.join('?' * len(cols))})",
                    [tuple(r[c] for c, _ in cols) for r in rows],
                )
            con.execute(f"COPY t TO {_sql_str(tmp)} (FORMAT parquet, COMPRESSION zstd)")
        finally:
            con.close()
        os.replace(tmp, path)
        return path

    def connect(self) -> duckdb.DuckDBPyConnection:
        """In-memory connection with one view per table over all its day files."""
        con = duckdb.connect()
        for table, cols in SCHEMAS.items():
            files = self.files(table)
            if files:
                file_list = "[" + ", ".join(_sql_str(f) for f in files) + "]"
                con.execute(
                    f"CREATE VIEW {table} AS SELECT * FROM read_parquet({file_list}, "
                    "union_by_name = true)"
                )
            else:
                con.execute(f"CREATE TABLE {table} ({', '.join(f'{c} {t}' for c, t in cols)})")
        return con

    def latest_hashes(self, before: date) -> dict[str, str]:
        """posting_key -> content_hash of its latest stored version first seen before `before`."""
        con = self.connect()
        try:
            rows = con.execute(
                "SELECT posting_key, arg_max(content_hash, first_seen_date) FROM postings "
                "WHERE first_seen_date < ? GROUP BY 1",
                [before],
            ).fetchall()
        finally:
            con.close()
        return dict(rows)

    def write_coverage(self, day: date, report: dict[str, Any]) -> Path:
        path = self.path("coverage", day)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, default=str, ensure_ascii=False), "utf-8")
        return path

    def read_coverage(self, day: date) -> dict[str, Any]:
        return json.loads(self.path("coverage", day).read_text(encoding="utf-8"))
