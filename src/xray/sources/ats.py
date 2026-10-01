"""Public employer job-board APIs (no auth): Greenhouse, Lever, Ashby.

Each adapter turns one board's raw JSON into ParsedJob records plus explicit skip reasons.
Fields a source does not carry are None; nothing is inferred across fields.
"""

from __future__ import annotations

import html
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from xray.fetch import Params
from xray.log import kv
from xray.text import clean_inline, html_to_text

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParsedJob:
    source_job_id: str
    title: str
    description: str | None
    company_name: str | None
    locations: list[str]
    structured_countries: list[str]
    department: str | None
    team: str | None
    employment_type: str | None
    workplace_type: str | None
    published_at: datetime | None
    updated_at: datetime | None
    url: str | None


Skip = tuple[str | None, str]  # (source_job_id, reason)


def _iso(value: Any, job_id: str, field: str) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        dt = datetime.fromisoformat(str(value))
    except ValueError:
        log.warning("unparseable timestamp", extra=kv(job_id=job_id, field=field, value=value))
        return None
    if dt.tzinfo is None:
        log.warning("timestamp without zone", extra=kv(job_id=job_id, field=field, value=value))
        return None
    return dt


def _epoch_ms(value: Any, job_id: str, field: str) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        return datetime.fromtimestamp(int(value) / 1000, UTC)
    except (TypeError, ValueError, OverflowError):
        log.warning("unparseable epoch", extra=kv(job_id=job_id, field=field, value=value))
        return None


def _dedup(items: list[str | None]) -> list[str]:
    seen: dict[str, None] = {}
    for it in items:
        c = clean_inline(it)
        if c and c.lower() not in {s.lower() for s in seen}:
            seen[c] = None
    return list(seen)


def _join(parts: list[str | None]) -> str | None:
    kept = [p for p in parts if p]
    return "\n\n".join(kept) if kept else None


def _base(job: dict, id_key: str, title_key: str, skips: list[Skip]) -> tuple[str, str] | None:
    jid = clean_inline(job.get(id_key))
    title = clean_inline(job.get(title_key))
    if not jid:
        skips.append((None, "missing_id"))
        return None
    if not title:
        skips.append((jid, "missing_title"))
        return None
    return jid, title


# ---------------------------------------------------------------- greenhouse


def parse_greenhouse(doc: Any) -> tuple[list[ParsedJob], list[Skip]]:
    if not isinstance(doc, dict) or not isinstance(doc.get("jobs"), list):
        raise ValueError("greenhouse response has no 'jobs' list")
    jobs, skips = [], []
    for j in doc["jobs"]:
        base = _base(j, "id", "title", skips)
        if base is None:
            continue
        jid, title = base
        content = j.get("content")
        offices = j.get("offices") or []
        depts = [clean_inline(d.get("name")) for d in j.get("departments") or []]
        jobs.append(
            ParsedJob(
                source_job_id=jid,
                title=title,
                description=html_to_text(html.unescape(content)) if content else None,
                company_name=clean_inline(j.get("company_name")),
                locations=_dedup(
                    [(j.get("location") or {}).get("name")]
                    + [o.get("location") or o.get("name") for o in offices]
                ),
                structured_countries=[],
                department=" / ".join(d for d in depts if d) or None,
                team=None,
                employment_type=None,
                workplace_type=None,
                published_at=_iso(j.get("first_published"), jid, "first_published"),
                updated_at=_iso(j.get("updated_at"), jid, "updated_at"),
                url=clean_inline(j.get("absolute_url")),
            )
        )
    return jobs, skips


# ---------------------------------------------------------------- lever


def parse_lever(doc: Any) -> tuple[list[ParsedJob], list[Skip]]:
    if not isinstance(doc, list):
        raise ValueError("lever response is not a list of postings")
    jobs, skips = [], []
    for j in doc:
        base = _base(j, "id", "text", skips)
        if base is None:
            continue
        jid, title = base
        cats = j.get("categories") or {}
        lists = [
            _join([clean_inline(li.get("text")), html_to_text(li.get("content"))])
            for li in j.get("lists") or []
        ]
        jobs.append(
            ParsedJob(
                source_job_id=jid,
                title=title,
                description=_join(
                    [
                        (j.get("openingPlain") or "").strip(),
                        (j.get("descriptionPlain") or "").strip()
                        or html_to_text(j.get("description")),
                        *lists,
                        (j.get("additionalPlain") or "").strip()
                        or html_to_text(j.get("additional")),
                    ]
                ),
                company_name=None,
                locations=_dedup([*(cats.get("allLocations") or []), cats.get("location")]),
                structured_countries=[c for c in [clean_inline(j.get("country"))] if c],
                department=clean_inline(cats.get("department")),
                team=clean_inline(cats.get("team")),
                employment_type=clean_inline(cats.get("commitment")),
                workplace_type=clean_inline(j.get("workplaceType")),
                published_at=_epoch_ms(j.get("createdAt"), jid, "createdAt"),
                updated_at=None,
                url=clean_inline(j.get("hostedUrl")),
            )
        )
    return jobs, skips


# ---------------------------------------------------------------- ashby


def _ashby_address(a: Any) -> tuple[list[str | None], str | None]:
    pa = ((a or {}).get("postalAddress") or {}) if isinstance(a, dict) else {}
    parts = [pa.get("addressLocality"), pa.get("addressRegion"), pa.get("addressCountry")]
    loc = ", ".join(p for p in parts if p) or None
    return [loc], clean_inline(pa.get("addressCountry"))


def parse_ashby(doc: Any) -> tuple[list[ParsedJob], list[Skip]]:
    if not isinstance(doc, dict) or not isinstance(doc.get("jobs"), list):
        raise ValueError("ashby response has no 'jobs' list")
    jobs, skips = [], []
    for j in doc["jobs"]:
        base = _base(j, "id", "title", skips)
        if base is None:
            continue
        jid, title = base
        if j.get("isListed") is False:
            skips.append((jid, "unlisted"))
            continue
        locs: list[str | None] = [j.get("location")]
        countries: list[str | None] = []
        addr_locs, addr_country = _ashby_address(j.get("address"))
        locs += addr_locs
        countries.append(addr_country)
        for s in j.get("secondaryLocations") or []:
            locs.append(s.get("location"))
            s_locs, s_country = _ashby_address(s.get("address"))
            locs += s_locs
            countries.append(s_country)
        jobs.append(
            ParsedJob(
                source_job_id=jid,
                title=title,
                description=(j.get("descriptionPlain") or "").strip()
                or html_to_text(j.get("descriptionHtml")),
                company_name=None,
                locations=_dedup(locs),
                structured_countries=_dedup(countries),
                department=clean_inline(j.get("department")),
                team=clean_inline(j.get("team")),
                employment_type=clean_inline(j.get("employmentType")),
                workplace_type=clean_inline(j.get("workplaceType")),
                published_at=_iso(j.get("publishedAt"), jid, "publishedAt"),
                updated_at=None,
                url=clean_inline(j.get("jobUrl")),
            )
        )
    return jobs, skips


# ---------------------------------------------------------------- registry


@dataclass(frozen=True)
class Adapter:
    url_template: str
    params: Params
    parse: Callable[[Any], tuple[list[ParsedJob], list[Skip]]]

    def url(self, board: str) -> str:
        return self.url_template.format(board=board)


ADAPTERS: dict[str, Adapter] = {
    "greenhouse": Adapter(
        "https://boards-api.greenhouse.io/v1/boards/{board}/jobs",
        {"content": "true"},
        parse_greenhouse,
    ),
    "lever": Adapter("https://api.lever.co/v0/postings/{board}", {"mode": "json"}, parse_lever),
    "ashby": Adapter("https://api.ashbyhq.com/posting-api/job-board/{board}", {}, parse_ashby),
}
