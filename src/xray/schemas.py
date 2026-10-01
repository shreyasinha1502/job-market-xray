"""Pydantic schemas. Fields the source does not provide are None — never defaulted or guessed."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Provenance(_Strict):
    source: Literal["adzuna"]
    run_id: str
    country: str
    role_query: str
    endpoint: str  # request URL with credentials stripped
    page: int = Field(ge=1)
    fetched_at: AwareDatetime
    raw_path: str  # repo-relative path of the cached raw response
    raw_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Posting(_Strict):
    """One real posting as returned for one role query in one snapshot.

    Nullable fields have no default on purpose: the normalizer must state each one explicitly.
    """

    snapshot_date: date
    source_id: str = Field(min_length=1)
    rank: int = Field(ge=1)  # 1-based position within the query's results
    title: str = Field(min_length=1)
    description: str | None
    company: str | None
    location_display: str | None
    location_area: list[str] | None
    category_label: str | None
    category_tag: str | None
    created: AwareDatetime
    salary_min: float | None
    salary_max: float | None
    salary_is_predicted: bool | None
    contract_type: str | None
    contract_time: str | None
    redirect_url: str | None
    latitude: float | None
    longitude: float | None
    provenance: Provenance


class SkippedRecord(_Strict):
    run_id: str
    country: str
    role_query: str
    page: int
    source_id: str | None
    reason: str


class QueryRun(_Strict):
    """One role query in one snapshot run: what was asked, what came back, what was kept."""

    run_id: str
    snapshot_date: date
    source: Literal["adzuna"]
    country: str
    role_query: str
    params: str  # JSON of non-secret request params
    started_at: datetime
    finished_at: datetime
    pages_fetched: int
    n_results_raw: int
    n_postings_kept: int
    n_skipped: int
    total_count: int | None  # source-reported total matches for the query; None if unknown
    mean_salary: float | None
    status: Literal["ok", "partial", "failed"]
    error: str | None
