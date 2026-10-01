"""spaCy skill extraction.

PhraseMatcher over the seed vocabulary (case-insensitive forms + exact-spelling forms), with
context rules for ambiguous forms ("Go", "R"). Every match is kept as a mention with the rule that
produced it; matches that fail a rule are kept too, flagged with `excluded_reason`, so nothing is
silently dropped. spaCy NER entities that are not vocabulary matches are recorded separately as
vocabulary-gap candidates for review. They are never counted as skills.
"""

from __future__ import annotations

import bisect
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span
from spacy.util import filter_spans

from xray.skills.sections import tag_sections
from xray.skills.vocab import SkillVocab

EXTRACTOR_VERSION = "m2.3"
NOT_IN_VOCAB = "(not in vocab)"
MODEL = "en_core_web_sm"
NER_LABELS = frozenset({"ORG", "PRODUCT", "GPE"})
# Entities in company boilerplate are mostly the employer's own names, offices and policies.
NER_SKIP_SECTIONS = frozenset({"legal", "benefits", "about"})
SYMBOL_AFTER = frozenset({"-", "+", "&", "'", "’"})


def norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9+#]+", "", s.lower())


@dataclass(frozen=True)
class PostingText:
    posting_key: str
    content_hash: str
    title: str
    description: str | None
    board: str
    company_name: str | None


@dataclass(frozen=True)
class Mention:
    skill: str
    category: str
    surface: str
    start_char: int  # offsets into f"{title}\n\n{description}"
    end_char: int
    section: str
    rule: str  # phrase | cased | cased+context
    excluded_reason: str | None


@dataclass(frozen=True)
class EntityCandidate:
    text_norm: str
    label: str
    n: int
    surface: str


@dataclass(frozen=True)
class Extraction:
    posting: PostingText
    mentions: list[Mention]
    entities: list[EntityCandidate]
    n_headings: int


def analysis_text(title: str, description: str | None) -> str:
    return f"{title}\n\n{description or ''}"


class SkillExtractor:
    def __init__(
        self, vocab: SkillVocab, *, location_terms: Iterable[str] = (), model: str = MODEL
    ) -> None:
        self.vocab = vocab
        self.nlp = spacy.load(model, exclude=["parser", "lemmatizer", "tagger", "attribute_ruler"])
        mk = self.nlp.make_doc
        self._lower = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        for skill, forms in vocab.lower_forms.items():
            self._lower.add(skill, [mk(f) for f in forms])
        self._cased = PhraseMatcher(self.nlp.vocab, attr="ORTH")
        for skill, forms in vocab.cased_forms.items():
            self._cased.add(skill, [mk(f) for f in forms])
        nc = vocab.needs_context
        self._ctx_only = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self._ctx_only.add("context_only", [mk(w) for w in nc.context_only_words])
        self._needs_ctx = set(nc.forms)
        self._reject_next = {w.lower() for w in nc.reject_if_next}
        self._cues_before = {w.lower() for w in nc.cue_words_before}
        self._cues_after = {w.lower() for w in nc.cue_words_after}
        self._window = nc.window
        self._locations = {norm_key(t) for t in location_terms}

    @property
    def extractor_id(self) -> dict[str, str]:
        m = self.nlp.meta
        return {
            "version": EXTRACTOR_VERSION,
            "vocab_sha256": self.vocab.sha256,
            "spacy": spacy.__version__,
            "model": f"{m['lang']}_{m['name']}-{m['version']}",
        }

    def extract(
        self, postings: Iterable[PostingText], batch_size: int = 16
    ) -> Iterator[Extraction]:
        stream = ((analysis_text(p.title, p.description), p) for p in postings)
        for doc, p in self.nlp.pipe(stream, as_tuples=True, batch_size=batch_size):
            yield self._one(doc, p)

    def _has_anchor(self, anchors: list[int], start: int, end: int) -> bool:
        i = bisect.bisect_left(anchors, start - self._window)
        return i < len(anchors) and anchors[i] < end + self._window

    def _ambiguity(self, doc: Doc, sp: Span, anchors: list[int]) -> str | None:
        """None if an ambiguous form ("Go", "R") reads as the skill here, else the reject reason."""
        nxt = doc[sp.end] if sp.end < len(doc) else None
        prev = doc[sp.start - 1] if sp.start > 0 else None
        if nxt is not None and nxt.text in SYMBOL_AFTER:
            return "ambiguous_followed_by_symbol"
        if nxt is not None and nxt.lower_ in self._reject_next:
            return "ambiguous_negative_pattern"
        if (
            (nxt is not None and nxt.lower_ in self._cues_after)
            or (prev is not None and prev.lower_ in self._cues_before)
            or self._has_anchor(anchors, sp.start, sp.end)
        ):
            return None
        return "ambiguous_no_context"

    def _one(self, doc: Doc, p: PostingText) -> Extraction:
        title_end = len(p.title)
        desc_offset = title_end + 2
        secs = tag_sections(p.description or "")

        def section_at(c: int) -> str:
            return "title" if c < title_end else secs.at(max(0, c - desc_offset))

        strings = self.nlp.vocab.strings
        lower = [Span(doc, s, e, label=strings[m]) for m, s, e in self._lower(doc)]
        cased = [Span(doc, s, e, label=strings[m]) for m, s, e in self._cased(doc)]
        cased_ids = {(sp.start, sp.end) for sp in cased}
        kept = filter_spans(lower + cased)
        taken = {t for sp in kept for t in range(sp.start, sp.end)}
        ctx_only = [
            Span(doc, s, e) for _, s, e in self._ctx_only(doc) if not taken & set(range(s, e))
        ]
        anchors = sorted(
            {t for sp in kept if sp.text not in self._needs_ctx for t in range(sp.start, sp.end)}
            | {t for sp in ctx_only for t in range(sp.start, sp.end)}
        )
        employer = {norm_key(p.board)} | ({norm_key(p.company_name)} if p.company_name else set())

        mentions: list[Mention] = []
        for sp in kept:
            skill = sp.label_
            rule = "cased" if (sp.start, sp.end) in cased_ids else "phrase"
            excluded = None
            if sp.text in self._needs_ctx:
                rule = "cased+context"
                excluded = self._ambiguity(doc, sp, anchors)
            if (
                excluded is None
                and self.vocab.exclude_employer_self_mentions
                and norm_key(skill) in employer
            ):
                excluded = "employer_self_mention"
            mentions.append(
                Mention(
                    skill=skill,
                    category=self.vocab.category[skill],
                    surface=sp.text,
                    start_char=sp.start_char,
                    end_char=sp.end_char,
                    section=section_at(sp.start_char),
                    rule=rule,
                    excluded_reason=excluded,
                )
            )
        # Known tech words outside the vocabulary: recorded (never counted) so the report can show
        # what the seed vocab misses. Single letters ("C") are too ambiguous to report.
        for sp in ctx_only:
            if len(sp.text) > 1:
                mentions.append(
                    Mention(
                        skill=sp.text.lower(),
                        category=NOT_IN_VOCAB,
                        surface=sp.text,
                        start_char=sp.start_char,
                        end_char=sp.end_char,
                        section=section_at(sp.start_char),
                        rule="context_only",
                        excluded_reason="not_in_vocab",
                    )
                )

        skill_chars = sorted((sp.start_char, sp.end_char) for sp in kept)
        ents: Counter[tuple[str, str]] = Counter()
        surface: dict[tuple[str, str], str] = {}
        for ent in doc.ents:
            if ent.label_ not in NER_LABELS or section_at(ent.start_char) in NER_SKIP_SECTIONS:
                continue
            if any(s < ent.end_char and ent.start_char < e for s, e in skill_chars):
                continue
            text = re.sub(r"\s+", " ", ent.text).strip(" .,;:()[]\"'™®")
            key = norm_key(text)
            if len(key) < 2 or key.isdigit() or key in self._locations:
                continue
            if any(emp and emp in key for emp in employer):  # employer boilerplate
                continue
            k = (text.lower(), ent.label_)
            ents[k] += 1
            surface.setdefault(k, text)
        entities = [EntityCandidate(t, lbl, n, surface[(t, lbl)]) for (t, lbl), n in ents.items()]
        return Extraction(p, mentions, entities, secs.n_headings)
