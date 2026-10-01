"""M6 dashboard pieces on the committed real data (data/processed) + model download checks."""

import hashlib
import io
import re
import zipfile

import httpx
import pytest

from xray.app import charts
from xray.app.data import load
from xray.app.model import ModelUnavailable, ensure_bundle


@pytest.fixture(scope="module")
def data():
    return load()


def test_committed_dashboard_inputs_load(data):
    assert data.as_of == data.history["last"]
    assert "all in-region postings" in data.scope_labels()
    assert data.quality["gaps"] is not None


def test_share_chart_draws_one_bar_and_one_whisker_per_skill(data):
    snap = data.scope("all in-region postings")["snapshot"]
    out = charts.share_chart(snap["skills"], snap["postings"], "t", top=10)
    assert out.count("<path ") == 10
    assert out.count("<title>") == 10  # hover text for every bar
    assert out.count('stroke-opacity=".55"') == 20  # whisker + cap per bar
    assert f"of {snap['postings']} postings" in out


def test_delta_chart_uses_both_poles_and_marks_significance():
    rows = [
        {"skill": "go", "delta_pp": 1.5, "first_share": 0.1, "last_share": 0.115,
         "flow_significant": True},
        {"skill": "java", "delta_pp": -0.5, "first_share": 0.2, "last_share": 0.195,
         "flow_significant": False},
    ]  # fmt: skip
    out = charts.delta_chart(rows, "t", "n")
    assert "var(--pos)" in out and "var(--neg)" in out
    assert "+1.5 *" in out and "-0.5<" in out


def test_status_is_never_color_alone():
    out = charts.status_list("g", [("warning", "only 1 snapshot day")])
    assert "<b>Gap</b>" in out


def test_dashboard_views_render(data):
    pytest.importorskip("gradio")
    from xray.app.dashboard import classify, quality_view, snapshot_view, trend_view

    html_, table = snapshot_view(data, "all in-region postings")
    assert table and len(table[0]) == 5
    head, _, _ = trend_view(data, "all in-region postings")
    assert "real snapshot day" in head
    assert "Known gaps" in quality_view(data)
    assert classify("too short")[0] is None


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
