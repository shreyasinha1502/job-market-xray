# Job Market X-Ray

Live job postings as an economic signal. This repo pulls real postings from public job APIs, extracts and
semantically normalizes skills, and tracks which skills are rising or falling in demand per role and
region, using only real accumulated snapshots.

## Hard rule: no synthetic data

- Nothing is fabricated, mocked or sampled to stand in for postings, skills, counts, dates or labels.
- A missing source or field is logged, skipped with a reason, and recorded in the daily coverage report.
  It stays missing (`None`), never defaulted.
- Every raw API response is cached to disk before parsing. Every stored posting carries provenance:
  source, board, endpoint (credentials stripped), fetch timestamp, raw file path, and the sha256 of the
  exact bytes received.
- `config/project.yaml:allow_synthetic` and `config/labeling.yaml:allow_synthetic_labels` must be present
  and `false`. The CLI refuses to start otherwise.
- Tests use real responses captured from the live APIs (`tests/fixtures/`). Only the HTTP transport is
  local, plus status sequences the API didn't happen to produce (e.g. a 429 before the final response).

## Data source: employers' public job-board APIs

Adzuna was the original plan, but it needs an account and API keys. The pipeline instead reads the
**public, no-auth job-board APIs** that employers publish through their applicant-tracking systems:

| ATS        | Endpoint                                                     | Gives                                      |
|------------|--------------------------------------------------------------|--------------------------------------------|
| Greenhouse | `boards-api.greenhouse.io/v1/boards/{board}/jobs?content=true` | full HTML description, offices, first_published |
| Lever      | `api.lever.co/v0/postings/{board}?mode=json`                 | full description, ISO country, createdAt   |
| Ashby      | `api.ashbyhq.com/posting-api/job-board/{board}`              | full description, postal country, publishedAt |

One call per board returns every open posting. Calls are throttled to one per 1.5 s per host and
retried with backoff on 429/5xx. Each stored row links back to the original posting (`url`).

Compared to Adzuna this gives **full descriptions instead of snippets** and real publish timestamps.
The cost is the sampling frame:

### Sampling frame (read this before interpreting any number)

The data covers a **fixed panel of 55 employers** (`config/panel.yaml`), not the whole Indian job
market. Membership rule: a public Greenhouse/Lever/Ashby board that had at least one India posting when
scanned on 2026-10-01. The candidate slugs were guesses from domain knowledge; the scan evidence (per
slug: HTTP status, total jobs, India jobs) is in `data/processed/panel_discovery/2026-10-01.json`.
The panel skews towards product/SaaS companies with Bengaluru offices. Indian IT-services firms and
many large employers don't use these ATSs, so they are absent. A fixed panel does make trends
comparable over time, because coverage doesn't drift the way an aggregator's index does. Boards added
later carry an `added:` date so trend windows can restrict to boards present throughout.

Adzuna support (`xray probe`, `src/xray/sources/adzuna.py`) is kept dormant. Put keys in `.env` to
use it as a second source later.

## Region and role rules

- **Region** (`config/regions.yaml`): a posting counts as India only if a structured country field says
  so (Lever `country`, Ashby postal address) or a location string names India or an Indian city.
  "Remote" or "APAC" alone doesn't count. Each row stores the evidence that matched
  (`country_evidence`), e.g. `structured_country=IN` or `location~Bengaluru, Karnataka, India`.
- **Role** (`config/sources.yaml: roles_to_track`): whole-phrase rules on the job title, with a list of
  non-engineering title phrases excluded (`role_title_exclude`). A title can match several roles. Roles
  are computed at read time from the stored title, so improving the rules never requires re-ingesting.

## Storage

```
data/raw/<source>/<run_id>/<board>.json.gz      exact response bytes (gzip-wrapped) + .meta.json  [gitignored]
data/processed/postings/<date>.parquet          postings first seen, or whose content changed, that day
data/processed/sightings/<date>.parquet         every in-region posting observed that day (stock)
data/processed/board_runs/<date>.parquet        one row per board fetch, failures included
data/processed/coverage/<date>.json             coverage / gaps report
```

Unchanged postings cost one sightings row per day, not a re-stored description, so the committed
history stays small. Read everything through `xray.store.Store().connect()` (DuckDB views over all
days). Timestamps are naive UTC in `*_utc` columns.

## Snapshots vs trends

Each run is a point-in-time snapshot. Trends need accumulated daily snapshots and are never
backfilled. On the first snapshot every posting is "new" because that is when the panel started
observing it, so `first_seen_date` on day 1 is not a posting date. Use `published_at_utc` for when
a job was posted. Every trend output states how many real days of history it rests on.

## Setup

```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -r requirements-dev.txt -e .
```

No credentials are needed for the default sources.

## Commands

```bash
python -m xray ingest                                 # today's snapshot for the whole panel
python -m xray show --role "data scientist" --city Bengaluru
python -m xray coverage                               # coverage / gaps report (add --full for skips)
python scripts/discover_panel.py                      # re-scan candidate boards (writes evidence)
python -m pytest && python -m ruff check .
```

`ingest` exits non-zero if any board failed. The other boards are still stored, and the failure is
listed in the coverage report.

## Status

- [x] M0: scaffold, config + hard gates, structured logging, schemas, polite fetcher + raw cache
- [x] M1: panel ingestion from public ATS APIs, provenance-stamped postings, coverage report
- [ ] M2: spaCy skill extraction
- [ ] M3: embedding-based skill normalization
- [ ] M4: trend engine
- [ ] M5: seniority classifier on rule-derived labels
- [ ] M6: Render deployment + scheduled ingestion
