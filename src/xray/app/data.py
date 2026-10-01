"""Read the committed dashboard inputs. The app never writes data: the daily GitHub Actions run
commits new snapshots, and the deployed image is rebuilt from them."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xray.config import PROCESSED_DIR


def _latest(folder: Path, pattern: str = "*.json") -> Path | None:
    files = sorted(p for p in folder.glob(pattern) if p.stem[:4].isdigit())
    return files[-1] if files else None


def _read(path: Path | None) -> dict[str, Any] | None:
    return json.loads(path.read_text("utf-8")) if path and path.exists() else None


@dataclass(frozen=True)
class DashboardData:
    trends: dict[str, Any]
    quality: dict[str, Any]
    skills_report: dict[str, Any] | None
    model_metrics: dict[str, Any] | None
    model_card: str | None
    skill_map_dump: str | None

    @property
    def as_of(self) -> str:
        return self.trends["as_of"]

    @property
    def history(self) -> dict[str, Any]:
        return self.trends["history"]

    def scope_labels(self) -> list[str]:
        return [s["scope"] for s in self.trends["scopes"]]

    def scope(self, label: str) -> dict[str, Any]:
        return next(s for s in self.trends["scopes"] if s["scope"] == label)


def load(root: Path = PROCESSED_DIR) -> DashboardData:
    trends = _read(_latest(root / "trends"))
    quality = _read(_latest(root / "quality"))
    if trends is None or quality is None:
        raise FileNotFoundError(
            f"no trends/quality JSON under {root}; run `python -m xray report` first"
        )
    card = root / "model" / "MODEL_CARD.md"
    dump = _latest(root / "skill_map", "clusters_*.md")
    return DashboardData(
        trends=trends,
        quality=quality,
        skills_report=_read(_latest(root / "skill_reports")),
        model_metrics=_read(root / "model" / "seniority_metrics.json"),
        model_card=card.read_text("utf-8") if card.exists() else None,
        skill_map_dump=dump.name if dump else None,
    )
