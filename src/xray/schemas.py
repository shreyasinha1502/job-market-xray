"""Pydantic schemas. Fields the source does not provide are None — never defaulted or guessed."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

Source = Literal["greenhouse", "lever", "ashby", "adzuna"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Provenance(_Strict):
    source: Source
    run_id: str
    query: str  # board slug (ATS sources) or search query (Adzuna)
    endpoint: str  # request URL with credentials stripped
    page: int | None = Field(ge=1)
    fetched_at: AwareDatetime
    raw_path: str  # repo-relative path of the cached raw response
    raw_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Posting(_Strict):
    """One real in-region posting, normalized from one cached raw response.

    Nullable fields have no default on purpose: the normalizer must state each one explicitly.
    """

    posting_key: str  # "{source}:{board}:{source_job_id}"
    source: Source
    board: str
    source_job_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str | None  # plain text from the source's HTML/plain fields
    company_name: str | None
    locations: list[str]  # every location string the source lists, whitespace-normalized
    countries: list[str] = Field(min_length=1)  # region keys matched (regions.yaml)
    country_evidence: list[str] = Field(min_length=1)  # which real field matched, per country
    cities: list[str]  # canonical cities matched; empty when only the country is named
    department: str | None
    team: str | None
    employment_type: str | None
    workplace_type: str | None
    published_at: AwareDatetime | None  # the source's own publish timestamp
    updated_at: AwareDatetime | None
    url: str | None
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    provenance: Provenance


class Sighting(_Strict):
    snapshot_date: date
    posting_key: str
    content_hash: str
    run_id: str


class SkippedRecord(_Strict):
    run_id: str
    source: Source
    board: str
    source_job_id: str | None
    reason: str


class BoardRun(_Strict):
    """One board fetched in one snapshot run: what was asked, what came back, what was kept."""

    run_id: str
    snapshot_date: date
    source: Source
    board: str
    endpoint: str
    fetched_at: AwareDatetime | None
    http_status: int | None
    attempts: int | None
    raw_path: str | None
    raw_sha256: str | None
    status: Literal["ok", "failed"]
    error: str | None
    n_jobs_total: int  # every job in the response, all regions
    n_skipped: int
    n_in_region: int
    n_out_of_region: int
    n_new: int
    n_changed: int
    n_unchanged: int
