"""Rules for the vocabulary added on 2026-10-02, on verbatim real postings that exercise them:
"Excel at tracking" (verb) vs "MS Excel/Google Sheets" (tool), "swift solutions" (adjective) vs
"Swift/Objective-C" (language)."""

import json
from pathlib import Path

import pytest

from xray.skills.extract import PostingText, SkillExtractor
from xray.skills.pipeline import location_terms
from xray.skills.vocab import load_vocab

CASES = Path(__file__).parent / "fixtures" / "postings_vocab_cases.jsonl"


@pytest.fixture(scope="module")
def by_title():
    recs = [json.loads(line) for line in CASES.read_text("utf-8").splitlines()]
    ex = SkillExtractor(load_vocab(), location_terms=location_terms())
    texts = [
        PostingText(r["posting_key"], r["content_hash"], r["title"], r["description"], r["board"],
                    r["company_name"])
        for r in recs
    ]  # fmt: skip
    return {r["title"]: e for r, e in zip(recs, ex.extract(texts), strict=True)}


def valid(e, skill):
    return [m for m in e.mentions if m.skill == skill and m.excluded_reason is None]


def test_excel_as_a_verb_is_rejected(by_title):
    e = by_title["Staff Technical Program Manager, GTM Tech"]
    flagged = [m for m in e.mentions if m.skill == "excel"]
    assert flagged and all(m.excluded_reason == "negative_next_word" for m in flagged)


def test_excel_as_a_tool_is_counted(by_title):
    assert valid(by_title["Internship - Talent Acquisition"], "excel")
    assert valid(by_title["Internship - Talent Acquisition"], "google sheets")


def test_swift_adjective_is_not_the_language(by_title):
    assert not valid(by_title["Financial Representative, Accounts Payable"], "swift")


def test_swift_language_is_counted(by_title):
    assert valid(by_title["Mobile Architect"], "swift")


def test_employer_products_are_self_mentions():
    v = load_vocab()
    assert "elasticsearch" in v.employer_products["elastic"]
    assert "claude code" in v.employer_products["anthropic"]
