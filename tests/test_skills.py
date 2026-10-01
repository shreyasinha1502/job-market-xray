"""Skill extraction on real postings.

tests/fixtures/postings_sample.jsonl holds six postings copied verbatim from the 2026-10-01
snapshot (each with its raw_path / raw_sha256 provenance), chosen because they exercise the
ambiguity rules: "R+E", "a spark of inspiration", "Go-live", "(C and Go)", "Python and/or R", and
an employer whose name is also a vocabulary skill (Databricks).
"""

import json
import shutil
from pathlib import Path

import pytest

from xray.config import CONFIG_DIR, ConfigError
from xray.skills.extract import PostingText, SkillExtractor, analysis_text
from xray.skills.pipeline import location_terms
from xray.skills.vocab import load_vocab

SAMPLE = Path(__file__).parent / "fixtures" / "postings_sample.jsonl"


def records() -> dict[str, dict]:
    recs = [json.loads(line) for line in SAMPLE.read_text("utf-8").splitlines()]
    return {r["board"]: r for r in recs}


def as_text(r: dict) -> PostingText:
    return PostingText(
        r["posting_key"], r["content_hash"], r["title"], r["description"], r["board"],
        r["company_name"],
    )  # fmt: skip


@pytest.fixture(scope="module")
def extracted():
    ex = SkillExtractor(load_vocab(), location_terms=location_terms())
    recs = records()
    return {
        board: e for board, e in zip(recs, ex.extract(map(as_text, recs.values())), strict=True)
    }


def counted(e, skill=None, surface=None):
    return [
        m
        for m in e.mentions
        if m.excluded_reason is None
        and (skill is None or m.skill == skill)
        and (surface is None or m.surface == surface)
    ]


def test_vocab_aliases_and_case_rules():
    v = load_vocab()
    assert "k8s" in v.lower_forms["kubernetes"]
    assert "spark" not in v.lower_forms["spark"] and "Spark" in v.cased_forms["spark"]
    assert set(v.needs_context.forms) == {"Go", "GO", "R"}


def test_vocab_rejects_alias_for_unknown_skill(tmp_path):
    cfg = tmp_path / "config"
    shutil.copytree(CONFIG_DIR, cfg)
    p = cfg / "skills.yaml"
    p.write_text(p.read_text("utf-8").replace("aliases:\n", "aliases:\n  rust: [\"rustlang\"]\n"),
                 encoding="utf-8")  # fmt: skip
    with pytest.raises(ConfigError, match="unknown skills"):
        load_vocab(cfg)


def test_r_in_a_language_list_is_counted(extracted):
    e = extracted["databricks"]
    assert counted(e, "r", "R")
    assert {"python", "java", "scala", "spark", "sql"} <= {m.skill for m in counted(e)}


def test_employer_name_is_not_counted_as_a_skill(extracted):
    dbx = [m for m in extracted["databricks"].mentions if m.skill == "databricks"]
    assert dbx and all(m.excluded_reason == "employer_self_mention" for m in dbx)


def test_r_plus_e_is_not_r(extracted):
    assert "R+E" in records()["rubrik"]["description"]
    assert not [m for m in extracted["rubrik"].mentions if m.skill == "r"]


def test_lowercase_spark_in_boilerplate_is_not_spark(extracted):
    assert "a spark of inspiration" in records()["meesho"]["description"]
    assert not counted(extracted["meesho"], "spark")


def test_go_live_is_rejected_and_kept_flagged(extracted):
    go = [m for m in extracted["highradius"].mentions if m.skill == "go"]
    assert go and all(m.excluded_reason == "ambiguous_followed_by_symbol" for m in go)


def test_go_next_to_c_is_counted(extracted):
    assert "(C and Go)" in records()["zscaler"]["description"]
    assert counted(extracted["zscaler"], "go", "Go")
    assert counted(extracted["zscaler"], "go", "Golang")[0].section == "title"


def test_python_and_or_r_is_counted(extracted):
    assert counted(extracted["okta"], "r", "R")


def test_offsets_point_at_the_surface_text(extracted):
    recs = records()
    for board, e in extracted.items():
        text = analysis_text(recs[board]["title"], recs[board]["description"])
        for m in e.mentions:
            assert text[m.start_char : m.end_char] == m.surface


def test_sections_are_tagged_from_real_headings(extracted):
    e = extracted["okta"]
    assert e.n_headings > 0
    assert {"requirements", "responsibilities"} <= {m.section for m in e.mentions}


def test_ner_candidates_never_include_vocab_matches_or_employer(extracted):
    vocab = set(load_vocab().skills)
    for board, e in extracted.items():
        for ent in e.entities:
            assert ent.text_norm not in vocab
            assert board not in ent.text_norm.replace(" ", "")


def test_extraction_is_deterministic(extracted):
    ex = SkillExtractor(load_vocab(), location_terms=location_terms())
    again = dict(zip(records(), ex.extract(map(as_text, records().values())), strict=True))
    assert all(again[b].mentions == extracted[b].mentions for b in extracted)


def test_go_after_cue_word_is_counted(extracted):
    assert "rewriting it in Go." in records()["sarvam"]["description"]
    assert counted(extracted["sarvam"], "go", "Go")


def test_go_to_market_and_go_beyond_are_rejected(extracted):
    assert "our Go To Market (GTM) team" in records()["openai"]["description"]
    assert "Go beyond the rulebook" in records()["groww"]["description"]
    for board in ("openai", "groww"):
        go = [m for m in extracted[board].mentions if m.skill == "go"]
        assert go and not counted(extracted[board], "go")
        assert {m.excluded_reason for m in go} <= {
            "ambiguous_negative_pattern",
            "ambiguous_followed_by_symbol",
            "ambiguous_no_context",
        }


def test_out_of_vocab_terms_are_observed_never_counted(extracted):
    gaps = [m for e in extracted.values() for m in e.mentions if m.rule == "context_only"]
    assert gaps
    assert all(m.excluded_reason == "not_in_vocab" and len(m.surface) > 1 for m in gaps)
    assert not {m.skill for m in gaps} & set(load_vocab().skills)
