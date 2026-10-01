"""M6 site: payload built from the committed real data, API endpoints, model download checks."""

import hashlib
import io
import re
import zipfile

import httpx
import pytest

from xray.app import payload
from xray.app.model import ModelUnavailable, ensure_bundle


@pytest.fixture(scope="module")
def doc():
    return payload.build()


def test_payload_is_built_from_committed_snapshot(doc):
    assert doc["as_of"] == doc["history"]["last"]
    ids = {s["id"] for s in doc["scopes"]}
    assert "all" in ids and any(i.startswith("role:") for i in ids)
    assert set(doc["trends"]) == ids
    all_scope = next(s for s in doc["scopes"] if s["id"] == "all")
    assert all_scope["n"] == doc["kpis"]["postings"]
    assert all(0 <= r["ci95"][0] <= r["share"] <= r["ci95"][1] <= 1 for r in all_scope["skills"])


def test_payload_categories_cover_every_skill(doc):
    assert set(doc["skill_category"].values()) <= set(doc["categories"])


def test_examples_are_held_out_real_postings(doc):
    assert doc["examples"]
    for e in doc["examples"]:
        assert e["description"] and e["target"] in {"senior", "below_senior"}


def test_model_results_include_floor_and_baseline(doc):
    keys = [r["key"] for r in doc["model"]["results"]]
    assert keys[:2] == ["majority", "baseline"]
    floor = doc["model"]["results"][0]["macro_f1"]
    assert all(r["macro_f1"] >= floor for r in doc["model"]["results"])


@pytest.fixture(scope="module")
def client():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from xray.app.server import app

    with TestClient(app) as c:
        yield c


def test_site_and_api_respond(client):
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    body = client.get("/api/dashboard").json()
    assert body["as_of"] and "serving" in body
    assert client.get("/healthz").json()["status"] == "ok"


def test_classify_rejects_short_text(client):
    assert client.post("/api/classify", json={"text": "too short"}).status_code == 422


def _zip_of(fixtures_dir) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:  # a real fixture file, but no model inside
        z.write(fixtures_dir / "greenhouse" / "not_found_404.json", "not_found_404.json")
    return buf.getvalue()


def test_model_download_requires_url(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_URL", raising=False)
    with pytest.raises(ModelUnavailable, match="MODEL_URL is not set"):
        ensure_bundle(tmp_path)


def test_model_download_rejects_checksum_mismatch(tmp_path, fixtures_dir):
    body = _zip_of(fixtures_dir)
    t = httpx.MockTransport(lambda req: httpx.Response(200, content=body))
    with pytest.raises(ModelUnavailable, match="checksum mismatch"):
        ensure_bundle(tmp_path, "https://example.invalid/b.zip", "0" * 64, transport=t)


def test_model_download_rejects_bundle_without_model(tmp_path, fixtures_dir):
    body = _zip_of(fixtures_dir)
    sha = hashlib.sha256(body).hexdigest()
    t = httpx.MockTransport(lambda req: httpx.Response(200, content=body))
    with pytest.raises(ModelUnavailable, match=re.escape("has no model.int8.onnx")):
        ensure_bundle(tmp_path, "https://example.invalid/b.zip", sha, transport=t)
