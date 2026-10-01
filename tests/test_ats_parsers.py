"""ATS parsers against real board responses captured from the live APIs on 2026-10-01."""

import json
import re

import pytest

from xray.sources.ats import ADAPTERS
from xray.text import html_to_text

BOARDS = [("greenhouse", "groww"), ("lever", "cred"), ("lever", "epifi"), ("ashby", "atlan")]


def load(fixtures_dir, source, board):
    return json.loads((fixtures_dir / source / f"{board}.json").read_bytes())


def raw_jobs(source, doc):
    return doc if source == "lever" else doc["jobs"]


@pytest.mark.parametrize(("source", "board"), BOARDS)
def test_every_raw_job_is_parsed_or_skipped_with_a_reason(fixtures_dir, source, board):
    doc = load(fixtures_dir, source, board)
    jobs, skips = ADAPTERS[source].parse(doc)
    assert len(jobs) + len(skips) == len(raw_jobs(source, doc))
    assert all(reason for _, reason in skips)
    assert {j.source_job_id for j in jobs} <= {str(r["id"]) for r in raw_jobs(source, doc)}


@pytest.mark.parametrize(("source", "board"), BOARDS)
def test_parsed_fields_are_clean_and_typed(fixtures_dir, source, board):
    jobs, _ = ADAPTERS[source].parse(load(fixtures_dir, source, board))
    assert jobs
    for j in jobs:
        assert j.title and j.title == j.title.strip()
        assert j.description, j.source_job_id
        assert not re.search(r"</?[a-zA-Z][^>]*>", j.description), "HTML tags left in description"
        assert "&lt;" not in j.description and "&amp;" not in j.description
        assert j.published_at is not None and j.published_at.tzinfo is not None
        assert j.locations or j.structured_countries


def test_lever_description_is_assembled_from_lists_when_plain_is_empty(fixtures_dir):
    doc = load(fixtures_dir, "lever", "cred")
    raw = next(r for r in doc if not r["descriptionPlain"] and r["lists"])
    job = next(j for j in ADAPTERS["lever"].parse(doc)[0] if j.source_job_id == raw["id"])
    first_line = html_to_text(raw["lists"][0]["content"]).splitlines()[0]
    assert first_line in job.description


def test_lever_country_field_is_kept_as_structured_evidence(fixtures_dir):
    doc = load(fixtures_dir, "lever", "cred")
    jobs, _ = ADAPTERS["lever"].parse(doc)
    by_id = {r["id"]: r for r in doc}
    assert all(j.structured_countries == [by_id[j.source_job_id]["country"]] for j in jobs)


def test_unexpected_shape_raises_instead_of_returning_nothing(fixtures_dir):
    not_found = json.loads((fixtures_dir / "greenhouse" / "not_found_404.json").read_bytes())
    for source in ADAPTERS:
        with pytest.raises(ValueError):
            ADAPTERS[source].parse(not_found)
