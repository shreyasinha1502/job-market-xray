"""Harvest candidate skill terms from the role sections of real postings in one snapshot.

Sources, all from real text:
  vocab   surfaces of valid vocabulary matches (these become the anchors)
  ner     spaCy entities recorded at extraction time (non-boilerplate sections only)
  shape   tech-shaped tokens: CamelCase, dotted/symbol names, ACRONYMS, letters+digits
  list    short items of comma/slash-separated lists ("Hadoop, Spark, Redis")

Spelling variants are grouped by a deterministic lexical key before any embedding.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date

import spacy

from xray.skills.extract import norm_key
from xray.skills.report import ROLE_SECTIONS, load_snapshot
from xray.skills.sections import tag_sections
from xray.skills.vocab import SkillVocab
from xray.store import Store

_SHAPE = re.compile(
    r"^(?=.*[A-Za-z])(?:"
    r"[A-Za-z]*[a-z][A-Z][A-Za-z0-9]*"  # CamelCase: PostgreSQL, TypeScript, GitHub
    r"|[A-Za-z][A-Za-z0-9]*[.+#][A-Za-z0-9.+#]*"  # Node.js, C++, C#, ASP.NET
    r"|[A-Z]{2,6}[0-9]*"  # ACRONYMS: AWS, ETL, EC2
    r"|[A-Za-z]+[0-9]+[A-Za-z0-9]*"  # letters+digits: S3, GPT4, k8s
    r")$"
)
# "/" separates list items ("Python/R/SQL") except between single letters ("A/B", "I/O")
_LIST_SEP = re.compile(r"\s*(?:,|(?<!\b[A-Za-z])/|(?<=\w\w)/|\||·|;|\band\b|\bor\b)\s*")
_ACRONYM_PLURAL = re.compile(r"^[A-Z][A-Za-z]*[A-Z]s$")
_LEAD_IN = re.compile(
    r"^.*\b(?:experience (?:with|in)|such as|e\.g\.?|including|like|knowledge of|"
    r"familiarity with|proficiency (?:with|in)|using|tools?:|stack:)\s*",
    re.I,
)
_STOP = {
    "a", "an", "the", "and", "or", "of", "in", "on", "with", "to", "for", "etc", "other",
    "similar", "related", "tools", "experience", "skills", "knowledge", "strong", "good",
}  # fmt: skip
MAX_LIST_ITEM_WORDS = 3
MAX_TERM_CHARS = 30


def lexical_key(surface: str) -> str:
    """Spelling-insensitive key: 'React.js' == 'ReactJS' == 'React JS' (but not 'React': dropping
    a "js" suffix here pulled ordinary words like "Next" into Next.js)."""
    s = surface.strip()
    if _ACRONYM_PLURAL.match(s):  # "SAs" (architects) is not "SAS"; "PoCs" is "PoC"
        s = s[:-1]
    return re.sub(r"[\s.\-_/]+", "", s.lower())


@dataclass
class Term:
    key: str
    surfaces: Counter[str] = field(default_factory=Counter)
    postings: set[str] = field(default_factory=set)
    boards: set[str] = field(default_factory=set)
    sources: set[str] = field(default_factory=set)
    vocab_surfaces: set[str] = field(default_factory=set)  # spellings the matcher already counts
    anchor: str | None = None  # vocabulary skill this term IS (lexically), if any
    ordinary: bool = False  # every word is written in lowercase at least as often as not

    @property
    def display(self) -> str:
        return self.surfaces.most_common(1)[0][0]


def _list_items(line: str) -> list[str]:
    if len(_LIST_SEP.findall(line)) < 2:
        return []
    out = []
    for i, seg in enumerate(_LIST_SEP.split(line.strip(" -•○*\t"))):
        if i == 0:
            seg = _LEAD_IN.sub("", seg)
        seg = re.sub(r"\(.*?\)|[()\[\]\"“”'’:.!]+$", "", seg).strip(" .,:;-")
        words = seg.split()
        if (
            not words
            or len(words) > MAX_LIST_ITEM_WORDS
            or len(seg) > MAX_TERM_CHARS
            or not seg[0].isalnum()
            or all(w.lower() in _STOP for w in words)
            or not any(c.isupper() or c in "+#" for c in seg)
        ):
            continue
        out.append(seg)
    return out


def harvest(
    store: Store, day: date, vocab: SkillVocab, *, exclude_terms: set[str]
) -> dict[str, Term]:
    """lexical key -> Term, over role sections of every posting in the snapshot."""
    posts, mentions, _ = load_snapshot(store, day)
    board_of = {p.key: p.board for p in posts}
    anchor_keys: dict[str, str] = {}
    for skill in vocab.skills:
        for f in [skill, *vocab.lower_forms.get(skill, []), *vocab.cased_forms.get(skill, [])]:
            anchor_keys[lexical_key(f)] = skill

    terms: dict[str, Term] = {}

    def add(surface: str, key_posting: str, source: str) -> Term | None:
        surface = surface.strip()
        k = lexical_key(surface)
        if len(k) < 2 or k.isdigit() or norm_key(surface) in exclude_terms:
            return None
        own = norm_key(board_of[key_posting])
        if source != "vocab" and own and own in norm_key(surface):
            return None  # the posting's own employer / product, not a skill
        t = terms.setdefault(k, Term(k, anchor=anchor_keys.get(k)))
        t.surfaces[surface] += 1
        t.postings.add(key_posting)
        t.boards.add(board_of[key_posting])
        t.sources.add(source)
        if source == "vocab":
            t.vocab_surfaces.add(surface)
        return t

    for m in mentions:
        if m.excluded_reason is None and m.section in ROLE_SECTIONS:
            t = add(m.surface, m.key, "vocab")
            if t is not None:
                t.anchor = m.skill

    con = store.connect()
    try:
        snap = (
            "SELECT DISTINCT p.posting_key, p.content_hash, p.title, p.description "
            "FROM sightings s JOIN postings p USING (posting_key, content_hash) "
            "WHERE s.snapshot_date = ?"
        )
        rows = con.execute(snap, [day]).fetchall()
        ents = con.execute(
            f"WITH snap AS ({snap}) SELECT e.posting_key, e.surface FROM entity_mentions e "
            "JOIN snap USING (posting_key, content_hash)",
            [day],
        ).fetchall()
    finally:
        con.close()
    for key, surface in ents:
        if len(surface) <= MAX_TERM_CHARS:
            add(surface, key, "ner")

    tok = spacy.blank("en").tokenizer
    lower_n: Counter[str] = Counter()  # word written all-lowercase
    other_n: Counter[str] = Counter()  # same word Capitalized / UPPER
    for key, _, title, desc in rows:
        secs = tag_sections(desc or "")
        offset = len(title) + 2
        lines = [(0, title)]
        pos = offset
        for ln in (desc or "").splitlines(keepends=True):
            lines.append((pos, ln))
            pos += len(ln)
        for start, line in lines:
            section = "title" if start < offset else secs.at(start - offset)
            if section not in ROLE_SECTIONS or not line.strip():
                continue
            for t in tok(line):
                (lower_n if t.text.islower() else other_n)[t.text.lower()] += 1
                if _SHAPE.match(t.text) and len(t.text) <= MAX_TERM_CHARS:
                    add(t.text, key, "shape")
            for item in _list_items(line):
                add(item, key, "list")

    # "Design", "Build", "WHAT", "In this role": capitalized only because they start a bullet or
    # heading. Real names (React, AWS, Linux, C++) are rarely written in lowercase.
    for t in terms.values():
        words = [w.lower() for w in t.display.split() if any(c.isalpha() for c in w)]
        t.ordinary = (
            t.anchor is None and bool(words) and all(lower_n[w] >= other_n[w] for w in words)
        )
    return terms


def frequent(terms: dict[str, Term], min_postings: int, min_boards: int) -> list[Term]:
    """Anchors are always kept; other terms must not be ordinary words and need real support
    across postings and employers."""
    keep = [
        t
        for t in terms.values()
        if t.anchor is not None
        or (not t.ordinary and len(t.postings) >= min_postings and len(t.boards) >= min_boards)
    ]
    return sorted(keep, key=lambda t: (-len(t.postings), t.key))


def source_counts(terms: list[Term]) -> dict[str, int]:
    c: dict[str, int] = defaultdict(int)
    for t in terms:
        for s in t.sources:
            c[s] += 1
    return dict(c)
