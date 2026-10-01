"""Find which candidate employers publish a public Greenhouse / Lever / Ashby job board with
India postings. Candidate slugs are guesses from domain knowledge; only boards that actually
respond with real India postings are kept. Writes the observed evidence to
data/processed/panel_discovery/<date>.json so the panel choice is auditable."""

from __future__ import annotations

import json
import re
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
UA = {"User-Agent": "job-market-xray/0.1 (research; panel discovery)"}
INDIA = re.compile(
    r"\b(india|bengaluru|bangalore|mumbai|hyderabad|pune|delhi|new delhi|gurgaon|gurugram|noida|"
    r"chennai|kolkata|ahmedabad|jaipur|kochi|coimbatore|thiruvananthapuram|indore|chandigarh)\b",
    re.I,
)

CANDIDATES = [
    s
    for line in (Path(__file__).parent / "panel_candidates.txt").read_text("utf-8").splitlines()
    if not line.startswith("#")
    for s in line.split()
]

ATS = {
    "greenhouse": ("https://boards-api.greenhouse.io/v1/boards/{s}/jobs", {}),
    "lever": ("https://api.lever.co/v0/postings/{s}", {"mode": "json"}),
    "ashby": ("https://api.ashbyhq.com/posting-api/job-board/{s}", {}),
}


def india_jobs(ats: str, doc) -> tuple[int, int]:
    if ats == "greenhouse":
        jobs = doc.get("jobs", [])
        locs = [(j.get("location") or {}).get("name", "") for j in jobs]
    elif ats == "lever":
        jobs = doc if isinstance(doc, list) else []
        locs = [
            " ".join(
                [
                    j.get("country") == "IN" and "india" or "",
                    *((j.get("categories") or {}).get("allLocations") or []),
                    (j.get("categories") or {}).get("location") or "",
                ]
            )
            for j in jobs
        ]
    else:
        jobs = doc.get("jobs", [])
        locs = [
            " ".join(
                [
                    j.get("location") or "",
                    *[s.get("location", "") for s in j.get("secondaryLocations") or []],
                    str(
                        ((j.get("address") or {}).get("postalAddress") or {}).get("addressCountry")
                        or ""
                    ),
                ]
            )
            for j in jobs
        ]
    return len(jobs), sum(bool(INDIA.search(x or "")) for x in locs)


def scan(ats: str, out: list) -> None:
    url_t, params = ATS[ats]
    with httpx.Client(timeout=30, headers=UA, follow_redirects=False) as c:
        for s in CANDIDATES:
            url = url_t.format(s=s)
            rec = {
                "ats": ats,
                "slug": s,
                "endpoint": str(httpx.URL(url, params=params)),
                "fetched_at": datetime.now(UTC).isoformat(),
            }
            try:
                r = c.get(url, params=params)
                rec["status"] = r.status_code
                if r.status_code == 200:
                    rec["total_jobs"], rec["india_jobs"] = india_jobs(ats, r.json())
            except Exception as e:  # noqa: BLE001 - record and move on, never invent
                rec["status"], rec["error"] = None, repr(e)
            out.append(rec)
            time.sleep(0.5)


def main() -> int:
    results: list = []
    threads = [threading.Thread(target=scan, args=(a, results)) for a in ATS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    day = datetime.now(UTC).date().isoformat()
    out = ROOT / "data" / "processed" / "panel_discovery" / f"{day}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    results.sort(key=lambda r: (r["ats"], r["slug"]))
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    hits = [r for r in results if r.get("india_jobs")]
    for r in sorted(hits, key=lambda r: -r["india_jobs"]):
        print(f"{r['ats']:<10} {r['slug']:<16} total={r['total_jobs']:<5} india={r['india_jobs']}")
    found = [r for r in results if r.get("status") == 200]
    errs = [r for r in results if r.get("status") is None]
    print(
        f"\nboards found: {len(found)}  with India jobs: {len(hits)}  "
        f"India postings: {sum(r['india_jobs'] for r in hits)}  transport errors: {len(errs)}"
    )
    print(f"evidence: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
