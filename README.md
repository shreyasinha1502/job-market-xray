# Job Market X-Ray

Live job postings as an economic signal. This repo pulls real postings from a job API, extracts and
semantically normalizes skills, and tracks which skills are rising or falling in demand per role and
region, using only real accumulated snapshots.

## Hard rule: no synthetic data

- Nothing is fabricated, mocked or sampled to stand in for postings, skills, counts, dates or labels.
- A missing source or field is logged, skipped, and recorded in a coverage report. It stays missing.
- Every raw API response is cached to disk byte-for-byte before parsing. Every stored posting carries
  provenance: source, query, endpoint (credentials stripped), page, fetch timestamp, raw file path and
  sha256.
- `config/project.yaml:allow_synthetic` and `config/labeling.yaml:allow_synthetic_labels` must be present
  and `false`. The CLI refuses to start otherwise.
- Tests use real responses captured from the live API (`tests/fixtures/`). Where a test needs an HTTP
  status sequence the API didn't happen to produce (e.g. a 429 before the final response), only the
  status is simulated. The bodies are still real.

## Data source: Adzuna

`GET https://api.adzuna.com/v1/api/jobs/{country}/search/{page}`. Auth is `app_id` + `app_key` query
params, read from `.env`.

Known limits, which matter for the analysis:

- **Descriptions are snippets.** Adzuna's docs say the search response returns only a snippet of the
  job description, not the full text. Skill extraction and label derivation therefore see partial
  text, so expect coverage below what full postings would give.
- **Default key caps:** 25 calls/min, 250/day, 1,000/week, 2,500/month. Config in this repo makes 4
  calls/day (4 roles × 1 country × 1 page of 50), about 124/month.

## Setup

```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -r requirements-dev.txt -e .
cp .env.example .env   # then fill ADZUNA_APP_ID / ADZUNA_APP_KEY
```

## Commands

```bash
python -m xray probe --role "data scientist"   # one live call; caches raw + prints the real sample
python -m pytest
python -m ruff check .
```

## Layout

```
config/            sources / skills / labeling / project YAML (domain config, not data)
data/raw/          cached raw API responses + .meta.json sidecars (gitignored)
data/processed/    normalized, provenance-stamped tables (committed: history must persist)
src/xray/          package: config, log (JSON lines + secret redaction), schemas, sources/
tests/fixtures/    real captured API responses
logs/              JSON-lines run logs (gitignored)
```

## Snapshots vs trends

Day 1 is a cross-sectional snapshot only. Trends need accumulated daily snapshots and are never
backfilled. Every trend output states how many real days of history it rests on.

## Status

- [x] M0: scaffold, config + hard gates, structured logging, schemas, Adzuna client + live probe
- [ ] M1: ingestion: pagination, provenance-stamped postings table, coverage report
- [ ] M2: spaCy skill extraction
- [ ] M3: embedding-based skill normalization
- [ ] M4: trend engine
- [ ] M5: seniority classifier on rule-derived labels
- [ ] M6: Render deployment + scheduled ingestion
