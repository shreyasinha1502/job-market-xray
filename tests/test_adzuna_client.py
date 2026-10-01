"""Client control-flow tests. Response bodies are real bytes captured from the live API;
only HTTP status sequencing (e.g. a 429 before the final response) is simulated."""

import hashlib
import json

import httpx
import pytest

from xray.fetch import AuthError, cache_raw, read_raw
from xray.sources.adzuna import AdzunaClient, raw_page_path

APP_ID, APP_KEY = "test-id-not-real", "test-key-not-real"


def _client(handler, sleeps, clock=None):
    t = [0.0]
    return AdzunaClient(
        APP_ID,
        APP_KEY,
        min_interval_sec=1.5,
        max_retries=3,
        transport=httpx.MockTransport(handler),
        sleep=lambda s: (sleeps.append(s), t.__setitem__(0, t[0] + s)),
        monotonic=clock or (lambda: t[0]),
    )


@pytest.fixture
def auth_fail_body(fixtures_dir) -> bytes:
    return (fixtures_dir / "adzuna" / "auth_fail_401.json").read_bytes()


def test_auth_failure_is_not_retried_and_raises(auth_fail_body):
    calls, sleeps = [], []

    def handler(req):
        calls.append(req)
        return httpx.Response(
            401, content=auth_fail_body, headers={"Content-Type": "application/json; charset=utf8"}
        )

    with _client(handler, sleeps) as c:
        raw = c.search("in", 1, {"what": "data scientist", "results_per_page": 50})
    assert len(calls) == 1 and raw.attempts == 1 and raw.status == 401
    assert raw.body == auth_fail_body
    with pytest.raises(AuthError, match="AUTH_FAIL"):
        raw.raise_for_status()


def test_retries_429_with_exponential_backoff_then_returns_final(auth_fail_body):
    statuses = iter([429, 429, 401])
    sleeps = []

    def handler(req):
        s = next(statuses)
        return httpx.Response(s, content=auth_fail_body if s == 401 else b"")

    with _client(handler, sleeps) as c:
        raw = c.search("in", 1, {"what": "x"})
    assert raw.attempts == 3 and raw.status == 401
    assert [s for s in sleeps if s >= 2.0] == [2.0, 4.0]


def test_retry_after_header_is_honoured(auth_fail_body):
    statuses = iter([429, 401])
    sleeps = []

    def handler(req):
        s = next(statuses)
        h = {"Retry-After": "7"} if s == 429 else {}
        return httpx.Response(s, content=auth_fail_body if s == 401 else b"", headers=h)

    with _client(handler, sleeps) as c:
        c.search("in", 1, {"what": "x"})
    assert 7.0 in sleeps


def test_credentials_sent_but_never_exposed(auth_fail_body, tmp_path):
    seen = []

    def handler(req):
        seen.append(req.url)
        return httpx.Response(
            401, content=auth_fail_body, headers={"Content-Type": "application/json"}
        )

    with _client(handler, []) as c:
        raw = c.search("in", 1, {"what": "data scientist"})
    assert seen[0].params["app_key"] == APP_KEY  # sent to the API
    cached = cache_raw(raw, raw_page_path(tmp_path, "in", "run1", "data scientist", 1), "adzuna")
    meta = (cached.path.parent / "page-1.meta.json").read_text(encoding="utf-8")
    for text in (raw.endpoint, meta):
        assert APP_KEY not in text and APP_ID not in text
    assert read_raw(cached.path) == auth_fail_body
    assert hashlib.sha256(auth_fail_body).hexdigest() == cached.sha256
    assert json.loads(meta)["sha256"] == cached.sha256


def test_min_interval_between_calls(auth_fail_body):
    sleeps = []

    def handler(req):
        return httpx.Response(401, content=auth_fail_body)

    with _client(handler, sleeps) as c:
        c.search("in", 1, {"what": "x"})
        c.search("in", 2, {"what": "x"})
    assert sleeps == [1.5]
