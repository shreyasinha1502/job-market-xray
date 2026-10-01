"""Shared test helpers: real captured board responses served through a local transport."""

import json
import shutil
from datetime import date
from pathlib import Path

import httpx

from xray.config import CONFIG_DIR
from xray.store import Store

FIXTURES = Path(__file__).parent / "fixtures"

BOARDS = [("greenhouse", "groww"), ("lever", "cred"), ("lever", "epifi"), ("ashby", "atlan")]
CAPTURED = date(2026, 10, 1)
IN_REGION = 7 + 8 + 2 + 4  # groww all, cred all, epifi all, atlan 4 of 6


def make_config(tmp_path: Path, boards) -> Path:
    cfg = tmp_path / "config"
    shutil.copytree(CONFIG_DIR, cfg, dirs_exist_ok=True)
    lines = ["boards:"] + [f"  - {{source: {s}, board: {b}, added: 2026-10-01}}" for s, b in boards]
    (cfg / "panel.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return cfg


def fixture_transport(
    fixtures_dir: Path, fail: frozenset[str] = frozenset()
) -> httpx.MockTransport:
    """Serve real captured bodies; boards in `fail` (or without a fixture) get the real 404 body."""
    hosts = {"boards-api.greenhouse.io": "greenhouse", "api.lever.co": "lever"}

    def handler(req: httpx.Request) -> httpx.Response:
        source = hosts.get(req.url.host, "ashby")
        parts = req.url.path.strip("/").split("/")
        board = parts[2] if source == "greenhouse" else parts[-1]  # /v1/boards/{b}/jobs
        body = fixtures_dir / source / f"{board}.json"
        if board in fail or not body.exists():
            nf = (fixtures_dir / "greenhouse" / "not_found_404.json").read_bytes()
            return httpx.Response(404, content=nf, headers={"Content-Type": "application/json"})
        meta = json.loads((fixtures_dir / source / f"{board}.meta.json").read_text("utf-8"))
        return httpx.Response(
            200, content=body.read_bytes(), headers={"Content-Type": meta["content_type"]}
        )

    return httpx.MockTransport(handler)


def count(store: Store, sql: str) -> int:
    con = store.connect()
    try:
        return con.execute(sql).fetchone()[0]
    finally:
        con.close()
