"""FastAPI backend for the live site: one JSON document for the dashboard, one classify endpoint.
The page itself is static (static/index.html + app.js); data comes only from committed files."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from xray.app import payload
from xray.app.model import ModelUnavailable, predictor

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
STATE: dict[str, Any] = {"payload": None, "model": {"available": False, "reason": "not loaded"}}


def load_model() -> None:
    try:
        p = predictor()
        STATE["model"] = {"available": True, "kind": p.kind}
        log.info("model loaded")
    except ModelUnavailable as e:
        STATE["model"] = {"available": False, "reason": str(e)}
        log.error(f"classifier disabled: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["payload"] = payload.build()
    load_model()  # once per process, before serving
    yield


app = FastAPI(title="Job Market X-Ray", docs_url=None, redoc_url=None, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class ClassifyIn(BaseModel):
    text: str = Field(min_length=1, max_length=50_000)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/dashboard")
def dashboard() -> JSONResponse:
    body = {**STATE["payload"], "serving": STATE["model"]}
    return JSONResponse(body, headers={"Cache-Control": "public, max-age=300"})


@app.post("/api/classify")
def classify(req: ClassifyIn) -> dict[str, Any]:
    if len(req.text.strip()) < 200:
        raise HTTPException(422, "Paste a full job description (at least a few sentences).")
    try:
        out = predictor().predict(req.text)
    except ModelUnavailable as e:
        raise HTTPException(503, f"Model not available: {e}") from e
    if "error" in out:
        raise HTTPException(422, out["error"])
    return out


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    p = STATE["payload"] or {}
    return {"status": "ok", "as_of": p.get("as_of"), "model": STATE["model"]}
