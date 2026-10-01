"""Get the serving model onto local disk once, verify it, load it once.

Weights are not in git. On a fresh (ephemeral) instance the bundle is downloaded from
MODEL_URL (a GitHub release asset), checked against MODEL_SHA256, unzipped into XRAY_MODEL_DIR
and loaded into memory a single time for all requests.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import zipfile
from functools import lru_cache
from pathlib import Path

import httpx

from xray.classify.predict import ONNX_FILE
from xray.config import MODEL_DIR
from xray.log import kv

log = logging.getLogger(__name__)

BUNDLE = "seniority-onnx"


class ModelUnavailable(RuntimeError):
    pass


def ensure_bundle(
    model_dir: Path = MODEL_DIR,
    url: str | None = None,
    expected_sha256: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> Path:
    bundle = model_dir / BUNDLE
    if (bundle / ONNX_FILE).exists():
        return bundle
    url = url or os.environ.get("MODEL_URL")
    expected_sha256 = expected_sha256 or os.environ.get("MODEL_SHA256")
    if not url:
        raise ModelUnavailable(f"no model at {bundle} and MODEL_URL is not set")
    log.info("downloading model bundle", extra=kv(url=url))
    with httpx.Client(follow_redirects=True, timeout=120, transport=transport) as c:
        resp = c.get(url)
    if resp.status_code != 200:
        raise ModelUnavailable(f"model download failed: HTTP {resp.status_code} from {url}")
    digest = hashlib.sha256(resp.content).hexdigest()
    if expected_sha256 and digest != expected_sha256.lower():
        raise ModelUnavailable(f"model checksum mismatch: got {digest}, want {expected_sha256}")
    if not expected_sha256:
        log.warning("MODEL_SHA256 not set; bundle not verified", extra=kv(sha256=digest))
    bundle.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        z.extractall(bundle)
    if not (bundle / ONNX_FILE).exists():
        raise ModelUnavailable(f"bundle from {url} has no {ONNX_FILE}")
    return bundle


@lru_cache(maxsize=1)
def predictor():
    """Loaded once per process; raises ModelUnavailable with a reason the UI can show."""
    from xray.classify.predict import OnnxSeniorityPredictor

    return OnnxSeniorityPredictor.load(ensure_bundle())
