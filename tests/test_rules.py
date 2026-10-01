"""Region / role rules on real postings from the captured fixtures."""

import json

from xray.config import load_regions, load_sources
from xray.rules import RegionMatcher, RoleMatcher
from xray.sources.ats import ADAPTERS


def parsed(fixtures_dir, source, board):
    doc = json.loads((fixtures_dir / source / f"{board}.json").read_bytes())
    return ADAPTERS[source].parse(doc)[0]


def matcher():
    return RegionMatcher(load_regions(load_sources()))


def test_location_text_match_with_city(fixtures_dir):
    jobs = parsed(fixtures_dir, "greenhouse", "groww")
    m = matcher().match(jobs[0].locations, jobs[0].structured_countries)
    assert [(x.country, x.cities) for x in m] == [("in", ("Bengaluru",))]
    assert m[0].evidence == f"location~{jobs[0].locations[0]}"


def test_structured_country_takes_precedence(fixtures_dir):
    for j in parsed(fixtures_dir, "lever", "cred"):
        (m,) = matcher().match(j.locations, j.structured_countries)
        assert m.evidence == "structured_country=IN"


def test_country_only_posting_has_no_city(fixtures_dir):
    j = next(j for j in parsed(fixtures_dir, "ashby", "atlan") if j.locations == ["India"])
    (m,) = matcher().match(j.locations, j.structured_countries)
    assert m.cities == ()


def test_real_non_india_postings_are_rejected(fixtures_dir):
    jobs = parsed(fixtures_dir, "ashby", "atlan")
    us = [j for j in jobs if "United States" in j.structured_countries]
    assert us and all(matcher().match(j.locations, j.structured_countries) == [] for j in us)


def test_roles_on_real_titles(fixtures_dir):
    roles = RoleMatcher.from_config(load_sources())
    titles = {j.title: roles.match(j.title) for j in parsed(fixtures_dir, "lever", "epifi")}
    assert titles["DS/ML Intern"] == ["data scientist", "machine learning engineer"]
    assert titles["GTM - sales"] == []  # excluded function
    assert all(roles.match(j.title) == [] for j in parsed(fixtures_dir, "lever", "cred"))
