"""Point-in-time skill views over one snapshot: coverage report, skill tables, mention contexts.

Coverage is computed over every in-region posting observed that day and broken down by tracked
role. It is never filtered to make the headline number look better.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from xray.config import CONFIG_DIR, ConfigError, load_sources
from xray.rules import RoleMatcher
from xray.skills.extract import analysis_text
from xray.skills.pipeline import read_manifest
from xray.skills.vocab import load_vocab
from xray.store import Store

REQUIREMENT_SECTIONS = frozenset({"title", "requirements"})
ROLE_SECTIONS = frozenset({"title", "intro", "responsibilities", "requirements"})
SCOPES = ("role", "requirements", "anywhere")

_SNAPSHOT = """
    WITH snap AS (
        SELECT DISTINCT p.posting_key, p.content_hash, p.title, p.board, p.cities
        FROM sightings s JOIN postings p USING (posting_key, content_hash)
        WHERE s.snapshot_date = ?
    )
"""


@dataclass(frozen=True)
class SnapPosting:
    key: str
    content_hash: str
    title: str
    board: str
    cities: list[str]


@dataclass(frozen=True)
class SnapMention:
    key: str
    skill: str
    category: str
    section: str
    surface: str
    excluded_reason: str | None
    start_char: int
    end_char: int


def load_snapshot(store: Store, day: date) -> tuple[list[SnapPosting], list[SnapMention], dict]:
    con = store.connect()
    try:
        posts = [
            SnapPosting(*r) for r in con.execute(_SNAPSHOT + "SELECT * FROM snap", [day]).fetchall()
        ]
        mentions = [
            SnapMention(*r)
            for r in con.execute(
                _SNAPSHOT
                + """SELECT m.posting_key, m.skill, m.category, m.section, m.surface,
                            m.excluded_reason, m.start_char, m.end_char
                     FROM skill_mentions m JOIN snap USING (posting_key, content_hash)""",
                [day],
            ).fetchall()
        ]
        headings = dict(
            con.execute(
                _SNAPSHOT + "SELECT e.posting_key, e.n_headings FROM extraction_meta e "
                "JOIN snap USING (posting_key, content_hash)",
                [day],
            ).fetchall()
        )
    finally:
        con.close()
    missing = len(posts) - len(headings)
    if missing:
        raise ConfigError(
            f"{missing} postings in snapshot {day} have no extraction; run `xray extract`"
        )
    return posts, mentions, headings


def _contexts(store: Store, wanted: list[SnapMention], pad: int = 60) -> list[str]:
    if not wanted:
        return []
    con = store.connect()
    try:
        texts = dict(
            (k, analysis_text(t, d))
            for k, t, d in con.execute(
                "SELECT posting_key, title, description FROM postings WHERE posting_key IN "
                f"({', '.join('?' * len({m.key for m in wanted}))})",
                sorted({m.key for m in wanted}),
            ).fetchall()
        )
    finally:
        con.close()
    out = []
    for m in wanted:
        t = texts[m.key]
        a, b = max(0, m.start_char - pad), m.end_char + pad
        parts = (t[a : m.start_char], t[m.start_char : m.end_char], t[m.end_char : b])
        before, hit, after = (x.replace("\n", " | ") for x in parts)
        out.append(f"{before}[[{hit}]]{after}")
    return out


def _scoped(m: SnapMention, scope: str) -> bool:
    if scope == "anywhere":
        return True
    return m.section in (REQUIREMENT_SECTIONS if scope == "requirements" else ROLE_SECTIONS)


def counted_skills(mentions: list[SnapMention], scope: str = "role") -> dict[str, set[str]]:
    """posting_key -> skills counted under `scope` (valid, in-vocab mentions only)."""
    out: dict[str, set[str]] = defaultdict(set)
    for m in mentions:
        if m.excluded_reason is None and _scoped(m, scope):
            out[m.key].add(m.skill)
    return out


def skill_table(
    posts: list[SnapPosting], mentions: list[SnapMention], *, scope: str = "role"
) -> list[dict[str, Any]]:
    """Per skill: postings mentioning it, share, boards, and the single largest board's share.

    scope: "role" (title/intro/responsibilities/requirements, the default), "requirements"
    (title/requirements only) or "anywhere" (includes company boilerplate sections).
    """
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}")
    by_key = {p.key: p for p in posts}
    holders: dict[str, set[str]] = defaultdict(set)
    req_holders: dict[str, set[str]] = defaultdict(set)
    category: dict[str, str] = {}
    for m in mentions:
        if m.excluded_reason is not None or m.key not in by_key:
            continue
        category[m.skill] = m.category
        if m.section in REQUIREMENT_SECTIONS:
            req_holders[m.skill].add(m.key)
        if _scoped(m, scope):
            holders[m.skill].add(m.key)
    n = len(posts)
    rows = []
    for skill, keys in holders.items():
        boards = Counter(by_key[k].board for k in keys)
        top_board, top_n = boards.most_common(1)[0]
        rows.append(
            {
                "skill": skill,
                "category": category[skill],
                "postings": len(keys),
                "share": round(len(keys) / n, 4) if n else None,
                "in_requirements_or_title": len(req_holders[skill] & keys),
                "boards": len(boards),
                "top_board": top_board,
                "top_board_share": round(top_n / len(keys), 3),
            }
        )
    return sorted(rows, key=lambda r: (-r["postings"], r["skill"]))


def build_report(store: Store, day: date, config_dir: Path = CONFIG_DIR) -> dict[str, Any]:
    posts, mentions, headings = load_snapshot(store, day)
    vocab = load_vocab(config_dir)
    roles = RoleMatcher.from_config(load_sources(config_dir))
    manifest = read_manifest(store)
    by_scope = {sc: counted_skills(mentions, sc) for sc in SCOPES}
    n = len(posts)

    def cov(group: list[SnapPosting], scope: str = "role") -> dict[str, Any]:
        k = sum(bool(by_scope[scope][p.key]) for p in group)
        return {
            "postings": len(group),
            "with_skill": k,
            "coverage": round(k / len(group), 4) if group else None,
        }

    role_of = {p.key: roles.match(p.title) for p in posts}
    by_role = {r: cov([p for p in posts if r in role_of[p.key]]) for r in roles.names}
    by_role["(no tracked role)"] = cov([p for p in posts if not role_of[p.key]])
    table = skill_table(posts, mentions)
    seen = {r["skill"] for r in table}

    flagged = [m for m in mentions if m.excluded_reason not in (None, "not_in_vocab")]
    samples: list[dict[str, Any]] = []
    for reason in sorted({m.excluded_reason for m in flagged}):
        picks = [m for m in flagged if m.excluded_reason == reason][:4]
        samples += [
            {"reason": reason, "surface": m.surface, "context": c}
            for m, c in zip(picks, _contexts(store, picks), strict=True)
        ]
    valid = [m for m in mentions if m.excluded_reason is None]
    accepted_ambiguous = [m for m in valid if m.surface in vocab.needs_context.forms][:6]

    gap_holders: dict[str, set[str]] = defaultdict(set)
    gap_boards: dict[str, set[str]] = defaultdict(set)
    board_of = {p.key: p.board for p in posts}
    for m in mentions:
        if m.excluded_reason == "not_in_vocab" and m.section in ROLE_SECTIONS:
            gap_holders[m.skill].add(m.key)
            gap_boards[m.skill].add(board_of[m.key])
    gaps = sorted(
        (
            {"term": t, "postings": len(k), "boards": len(gap_boards[t])}
            for t, k in gap_holders.items()
        ),
        key=lambda r: -r["postings"],
    )

    role_counted = by_scope["role"]
    tracked_zero = sorted({p.title for p in posts if role_of[p.key] and not role_counted[p.key]})
    return {
        "snapshot_date": day.isoformat(),
        "generated_at": datetime.now(UTC).isoformat(),
        "extractor": manifest.get("extractor"),
        "vocab": {
            "skills": len(vocab.skills),
            "surface_forms": vocab.n_surface_forms(),
            "never_seen_in_snapshot": sorted(set(vocab.skills) - seen),
        },
        "coverage": {
            "definition": "posting has >=1 valid vocabulary skill in title, intro, "
            "responsibilities or requirements (company about/benefits/legal sections excluded)",
            "all_in_region": cov(posts),
            "by_role": by_role,
            "requirements_only": cov(posts, "requirements"),
            "anywhere_incl_boilerplate": cov(posts, "anywhere"),
        },
        "skills": table,
        "vocab_gaps_observed": {
            "note": "known tech terms seen in role sections that the seed vocab does not contain; "
            "observed only, never counted",
            "terms": gaps,
        },
        "sections": {
            "postings_with_recognized_headings": round(
                sum(1 for p in posts if headings.get(p.key, 0) > 0) / n, 4
            )
            if n
            else None,
            "valid_mentions_by_section": dict(Counter(m.section for m in valid).most_common()),
        },
        "exclusions": {
            "by_reason": dict(Counter(m.excluded_reason for m in flagged)),
            "by_surface": dict(
                Counter(f"{m.surface}:{m.excluded_reason}" for m in flagged).most_common(15)
            ),
            "samples": samples,
            "accepted_ambiguous_samples": [
                {"surface": m.surface, "context": c}
                for m, c in zip(
                    accepted_ambiguous, _contexts(store, accepted_ambiguous), strict=True
                )
            ],
        },
        "tracked_role_postings_without_skill": tracked_zero[:25],
        "notes": [
            "Point-in-time snapshot over the fixed employer panel; not a trend.",
            "Skill counts use role sections only, so an employer's product blurb in its "
            "About section (e.g. 'runs on AWS, GCP and Azure') is not counted as demand.",
            "NER candidates are vocabulary-gap suggestions for review, not counted skills.",
        ],
    }


def ner_candidates(store: Store, day: date, min_boards: int = 3, limit: int = 100) -> list[dict]:
    con = store.connect()
    try:
        rows = con.execute(
            _SNAPSHOT
            + """SELECT e.text_norm, string_agg(DISTINCT e.label, '|') AS labels,
                        count(DISTINCT e.posting_key) AS postings,
                        count(DISTINCT snap.board) AS boards, any_value(e.surface) AS example
                 FROM entity_mentions e JOIN snap USING (posting_key, content_hash)
                 GROUP BY 1 HAVING count(DISTINCT snap.board) >= ?
                 ORDER BY postings DESC, boards DESC, 1 LIMIT ?""",
            [day, min_boards, limit],
        ).fetchall()
    finally:
        con.close()
    return [
        dict(zip(("candidate", "labels", "postings", "boards", "example"), r, strict=True))
        for r in rows
    ]


def write_report(
    store: Store, day: date, report: dict, candidates: list[dict]
) -> tuple[Path, Path]:
    out = store.root / "skill_reports"
    out.mkdir(parents=True, exist_ok=True)
    rp = out / f"{day.isoformat()}.json"
    rp.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    cp = out / f"ner_candidates_{day.isoformat()}.csv"
    with cp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["candidate", "labels", "postings", "boards", "example"])
        w.writeheader()
        w.writerows(candidates)
    return rp, cp
