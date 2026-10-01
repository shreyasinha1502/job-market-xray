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

## Skill extraction (M2)

`xray extract` runs spaCy (`en_core_web_sm` 3.8.0) over `title + description` of every stored
posting version:

- **PhraseMatcher** over the seed vocabulary in `config/skills.yaml` (35 skills). Aliases map other
  spellings to the same skill (`k8s` → kubernetes, `LLMs` → llm). They never add new skills.
- **Exact-spelling forms** for skills whose lowercase is ordinary English. "Spark" counts; Meesho's
  "a spark of inspiration" does not.
- **Context rules for "Go" and "R"**: rejected when followed by a symbol or a known non-language word
  (`Go-live`, `Go To Market`, `Go beyond`). Counted when a cue word is adjacent (`in Go`,
  `Go services`) or another language/tool is within 6 tokens (`Python, R, SQL`, `(C and Go)`).
- **Employer self-mentions** are flagged, not counted: "Databricks" inside a Databricks posting names
  the employer (1,183 such mentions).
- **Section tags**: rule-based headings split each posting into title / intro / responsibilities /
  requirements / about / benefits / legal (99.5% of postings have recognized headings). Skill
  counts use role sections only, so an employer's About-section blurb ("runs on AWS, GCP and Azure")
  is not counted as demand.
- **NER** (ORG/PRODUCT/GPE) entities that are not vocabulary matches are written to
  `skill_reports/ner_candidates_<date>.csv` as vocabulary-gap suggestions. The small model often
  mislabels tech terms (e.g. "Databricks" as GPE), so NER output is never counted as a skill.

Every match is stored with its character offsets, section and the rule that produced it. Rejected
matches are kept and flagged (`excluded_reason`), never dropped. `xray mentions --skill go
--excluded` prints the text around each one.

### Coverage, 2026-10-01 snapshot (1,315 India postings)

| scope | postings with ≥1 skill |
|---|---|
| role sections (headline) | 753 / 1,315 = **57.3%** |
| title + requirements only | 691 = 52.6% |
| anywhere incl. company boilerplate | 794 = 60.4% |

By tracked role (role sections): data scientist 16/16, ML engineer 22/26, data analyst 22/27, data
engineer 11/12, backend 33/42, software engineer 226/245. Postings with no tracked role: 452/983
(46%), mostly sales, finance, operations and support roles that name no tech skill.

These numbers are not inflated. Known gaps, as reported in `skill_reports/<date>.json`:

- **Seed vocabulary misses common terms.** Seen in role sections but not in the vocab: Excel (112
  postings), C++ (90), Node.js (34), Rust (33), Ruby (28), Bash (24), C# (21), Kotlin (20). These are
  reported, not counted. Add them to `skills.yaml` deliberately if wanted.
- `spacy` and `xgboost` appear in no posting in this snapshot.
- **Precision spot check** (40 random counted mentions, seed 20261001): 39 correct. The one miss was
  "SQL" meaning *Sales Qualified Lead* next to "MQL". It occurs in 2 of 225 SQL postings.
- **Board concentration** is shown per skill (`top_board_share`). Some skills are dominated by one
  employer, e.g. scala 57% Databricks and mlflow 73% Databricks.

## Storage

```
data/raw/<source>/<run_id>/<board>.json.gz      exact response bytes (gzip-wrapped) + .meta.json  [gitignored]
data/processed/postings/<date>.parquet          postings first seen, or whose content changed, that day
data/processed/sightings/<date>.parquet         every in-region posting observed that day (stock)
data/processed/board_runs/<date>.parquet        one row per board fetch, failures included
data/processed/coverage/<date>.json             coverage / gaps report
data/processed/skill_mentions/<date>.parquet    every vocab match: offsets, section, rule, excluded_reason
data/processed/entity_mentions/<date>.parquet   NER entities that are not vocab matches (review only)
data/processed/extraction_meta/<date>.parquet   per posting: recognized section headings
data/processed/skill_mentions/_manifest.json    extractor id (version, vocab sha256, spaCy/model)
data/processed/skill_reports/<date>.json        skill coverage report for that snapshot
```

Derived skill tables are partitioned like `postings` and re-extracted automatically when the postings
file, the vocabulary, the spaCy/model version or the extractor version changes.

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
python -m xray extract                                # skill extraction (incremental) + skill report
python -m xray skills --role "data scientist"         # top skills in a snapshot (--scope role|requirements|anywhere)
python -m xray mentions --skill go --excluded         # audit: text around each (rejected) match
python scripts/discover_panel.py                      # re-scan candidate boards (writes evidence)
python -m pytest && python -m ruff check .
```

`ingest` exits non-zero if any board failed. The other boards are still stored, and the failure is
listed in the coverage report.

## Status

- [x] M0: scaffold, config + hard gates, structured logging, schemas, polite fetcher + raw cache
- [x] M1: panel ingestion from public ATS APIs, provenance-stamped postings, coverage report
- [x] M2: spaCy skill extraction, section tags, ambiguity rules, honest coverage report
- [ ] M3: embedding-based skill normalization
- [ ] M4: trend engine
- [ ] M5: seniority classifier on rule-derived labels
- [ ] M6: Render deployment + scheduled ingestion
