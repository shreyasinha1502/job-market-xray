"""Adzuna search API (dormant: needs ADZUNA_APP_ID / ADZUNA_APP_KEY, which this project
does not have yet). Kept so it can be enabled as a second source later."""

from __future__ import annotations

from pathlib import Path

from xray.fetch import Fetcher, Params, RawResponse, slug

SOURCE = "adzuna"
SEARCH_URL = "https://api.adzuna.com/v1/api/jobs/{country}/search/{page}"
# Adzuna's default per-key caps: 25 calls/min, 250/day, 1000/week, 2500/month.
DEFAULT_MAX_PER_MINUTE = 25


class AdzunaClient(Fetcher):
    def __init__(self, app_id: str, app_key: str, **kw) -> None:
        kw.setdefault("max_per_minute", DEFAULT_MAX_PER_MINUTE)
        super().__init__(secret_params={"app_id": app_id, "app_key": app_key}, **kw)

    def search(self, country: str, page: int, params: Params) -> RawResponse:
        if page < 1:
            raise ValueError("Adzuna pages are 1-based")
        return self.get(SEARCH_URL.format(country=country, page=page), params)


def raw_page_path(raw_root: Path, country: str, run_id: str, role: str, page: int) -> Path:
    return raw_root / SOURCE / country / run_id / slug(role) / f"page-{page}"
