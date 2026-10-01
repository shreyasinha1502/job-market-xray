"""Adzuna job-search client: throttling, retry/backoff, raw-response caching.

Every HTTP response that ends an attempt chain is returned as bytes so the caller can cache it to
disk *before* any parsing. Credentials never appear in returned URLs, cached metadata or logs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from xray.config import ROOT
from xray.log import kv

log = logging.getLogger(__name__)

SOURCE = "adzuna"
SEARCH_URL = "https://api.adzuna.com/v1/api/jobs/{country}/search/{page}"
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
# Adzuna's default per-key caps: 25 calls/min, 250/day, 1000/week, 2500/month.
DEFAULT_MAX_PER_MINUTE = 25

Params = dict[str, str | int]


class AdzunaError(RuntimeError):
    pass


class AdzunaAuthError(AdzunaError):
    pass


@dataclass(frozen=True)
class RawResponse:
    status: int
    body: bytes
    endpoint: str  # URL + non-secret params only
    params: Params  # non-secret params only
    fetched_at: datetime
    content_type: str | None
    attempts: int

    @property
    def ok(self) -> bool:
        return self.status == 200

    def raise_for_status(self) -> None:
        if self.ok:
            return
        snippet = self.body[:300].decode("utf-8", errors="replace")
        msg = f"HTTP {self.status} from {self.endpoint}: {snippet}"
        if self.status in (401, 403):
            raise AdzunaAuthError(msg)
        raise AdzunaError(msg)


class AdzunaClient:
    def __init__(
        self,
        app_id: str,
        app_key: str,
        *,
        min_interval_sec: float = 1.5,
        max_per_minute: int = DEFAULT_MAX_PER_MINUTE,
        max_retries: int = 5,
        backoff_base_sec: float = 2.0,
        backoff_cap_sec: float = 60.0,
        timeout_sec: float = 30.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._creds = {"app_id": app_id, "app_key": app_key}
        self._min_interval = min_interval_sec
        self._max_per_minute = max_per_minute
        self._max_retries = max_retries
        self._backoff_base = backoff_base_sec
        self._backoff_cap = backoff_cap_sec
        self._sleep = sleep
        self._monotonic = monotonic
        self._recent: deque[float] = deque()
        self._last_call: float | None = None
        self.calls_made = 0
        self._http = httpx.Client(
            timeout=timeout_sec,
            transport=transport,
            headers={"Accept": "application/json", "User-Agent": "job-market-xray/0.1"},
        )

    def __enter__(self) -> AdzunaClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    def _throttle(self) -> None:
        now = self._monotonic()
        if self._last_call is not None:
            wait = self._min_interval - (now - self._last_call)
            if wait > 0:
                self._sleep(wait)
                now = self._monotonic()
        while self._recent and now - self._recent[0] >= 60:
            self._recent.popleft()
        if len(self._recent) >= self._max_per_minute:
            wait = 60 - (now - self._recent[0])
            log.info("per-minute cap reached, waiting", extra=kv(wait_sec=round(wait, 2)))
            self._sleep(wait)
            now = self._monotonic()
            while self._recent and now - self._recent[0] >= 60:
                self._recent.popleft()
        self._recent.append(now)
        self._last_call = now

    def _backoff(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(self._backoff_cap, max(0.0, float(retry_after)))
            except ValueError:
                pass  # HTTP-date form; fall through to exponential
        return min(self._backoff_cap, self._backoff_base * (2**attempt))

    def search(self, country: str, page: int, params: Params) -> RawResponse:
        """Fetch one search page. Retries 429/5xx/transport errors; returns the final response."""
        if page < 1:
            raise ValueError("Adzuna pages are 1-based")
        url = SEARCH_URL.format(country=country, page=page)
        endpoint = str(httpx.URL(url, params=params))
        for attempt in range(self._max_retries + 1):
            self._throttle()
            fetched_at = datetime.now(UTC)
            try:
                resp = self._http.get(url, params={**self._creds, **params})
            except httpx.TransportError as e:
                self.calls_made += 1
                if attempt == self._max_retries:
                    raise AdzunaError(
                        f"transport error after {attempt + 1} attempts on {endpoint}: {e!r}"
                    ) from e
                delay = self._backoff(attempt, None)
                log.warning(
                    "transport error, retrying",
                    extra=kv(endpoint=endpoint, attempt=attempt + 1, delay_sec=delay, err=repr(e)),
                )
                self._sleep(delay)
                continue

            self.calls_made += 1
            status = resp.status_code
            if status in RETRY_STATUS and attempt < self._max_retries:
                delay = self._backoff(attempt, resp.headers.get("Retry-After"))
                log.warning(
                    "retryable status, backing off",
                    extra=kv(
                        endpoint=endpoint, status=status, attempt=attempt + 1, delay_sec=delay
                    ),
                )
                self._sleep(delay)
                continue

            log.info(
                "adzuna response",
                extra=kv(
                    endpoint=endpoint, status=status, bytes=len(resp.content), attempts=attempt + 1
                ),
            )
            return RawResponse(
                status=status,
                body=resp.content,
                endpoint=endpoint,
                params=dict(params),
                fetched_at=fetched_at,
                content_type=resp.headers.get("Content-Type"),
                attempts=attempt + 1,
            )
        raise AssertionError("unreachable")


# ---------------------------------------------------------------- raw cache


def new_run_id(now: datetime | None = None) -> str:
    return (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def raw_page_path(raw_root: Path, country: str, run_id: str, role: str, page: int) -> Path:
    return raw_root / SOURCE / country / run_id / slug(role) / f"page-{page}"


@dataclass(frozen=True)
class CachedRaw:
    path: Path
    sha256: str

    def rel(self) -> str:
        try:
            return self.path.relative_to(ROOT).as_posix()
        except ValueError:
            return self.path.as_posix()


def cache_raw(raw: RawResponse, stem: Path) -> CachedRaw:
    """Write the response body byte-for-byte, plus a sidecar with request provenance."""
    is_json = bool(raw.content_type and "json" in raw.content_type)
    body_path = stem.with_name(f"{stem.name}{'.json' if is_json else '.body'}")
    body_path.parent.mkdir(parents=True, exist_ok=True)
    body_path.write_bytes(raw.body)
    sha = hashlib.sha256(raw.body).hexdigest()
    meta = {
        "source": SOURCE,
        "endpoint": raw.endpoint,
        "params": raw.params,
        "status": raw.status,
        "attempts": raw.attempts,
        "fetched_at": raw.fetched_at.isoformat(),
        "content_type": raw.content_type,
        "bytes": len(raw.body),
        "sha256": sha,
    }
    stem.with_name(f"{stem.name}.meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return CachedRaw(body_path, sha)
