"""Real held-out postings for the classifier demo ("try an example").

Only test-split postings are used, so the demo never shows the model a posting it was trained
on. Each example carries its rule-derived label so a wrong prediction is visible as wrong.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb

from xray.classify.train import REPORT_DIR_NAME
from xray.store import Store

EXAMPLES_FILE = "examples.json"


def write_examples(store: Store | None = None, per_class: int = 2) -> Path | None:
    store = store or Store()
    labels = store.root / REPORT_DIR_NAME / "seniority_labels.parquet"
    if not labels.exists():
        return None
    con = store.connect()
    try:
        con.execute(f"CREATE VIEW labels AS SELECT * FROM read_parquet('{labels.as_posix()}')")
        rows = con.execute(
            """WITH latest AS (
                   SELECT * FROM postings QUALIFY row_number() OVER (
                       PARTITION BY posting_key ORDER BY first_seen_date DESC) = 1)
               SELECT l.posting_key, l.board, l.title, l.label, l.target, l.evidence,
                      p.description, p.url, p.published_at_utc, p.cities
               FROM labels l JOIN latest p USING (posting_key)
               WHERE l.split = 'test' AND length(p.description) BETWEEN 1500 AND 6000
               ORDER BY l.target, l.board, l.posting_key"""
        ).fetchall()
    except duckdb.Error:
        return None
    finally:
        con.close()
    picked: list[dict[str, Any]] = []
    for target in sorted({r[4] for r in rows}):
        boards: set[str] = set()
        for key, board, title, label, tgt, ev, desc, url, pub, cities in rows:
            if tgt != target or board in boards:
                continue  # one per employer, for variety
            boards.add(board)
            picked.append({
                "posting_key": key, "board": board, "title": title, "rule_label": label,
                "target": tgt, "evidence": list(ev or []), "url": url,
                "published": pub.date().isoformat() if pub else None,
                "cities": list(cities or []), "description": desc,
            })  # fmt: skip
            if len(boards) == per_class:
                break
    out = store.root / REPORT_DIR_NAME / EXAMPLES_FILE
    out.write_text(json.dumps({"source": "held-out test split", "examples": picked},
                              indent=1, ensure_ascii=False), encoding="utf-8")  # fmt: skip
    return out
