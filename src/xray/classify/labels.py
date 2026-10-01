"""M5 label derivation from real posting fields, per config/labeling.yaml. No invented labels.

A posting gets a label only when its own title (or, failing that, its description's experience
range) matches exactly one class. Everything else is excluded with a recorded reason.
The model input never contains the evidence a label was derived from: the title is left out and
every rule pattern is deleted from the description (see `model_input`).
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from xray.config import CONFIG_DIR, ConfigError, load_yaml
from xray.skills.sections import tag_sections

# Most informative first: experience/level cues live in requirements.
INPUT_SECTIONS = ("requirements", "responsibilities", "intro")


def _rx(patterns: list[str]) -> re.Pattern[str] | None:
    pats = [p.strip() for p in patterns if p.strip()]
    if not pats:
        return None
    alts = "|".join(re.escape(p) for p in sorted(pats, key=len, reverse=True))
    return re.compile(rf"(?<![a-z0-9])(?:{alts})(?![a-z0-9])", re.I)


def normalize(text: str | None) -> str:
    """Unicode dashes -> '-'; '0 – 2 years' reads as '0-2 years'. Line structure is kept."""
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", text or "")
    return re.sub(r"(?<=\d)[ \t]*-[ \t]*(?=\d)", "-", text)


@dataclass
class LabelRules:
    classes: dict[str, list[str]]
    exclude_if_ambiguous: bool
    min_per_class: int
    class_groups: dict[str, list[str]]  # target -> fine classes; identity if not configured
    exclude_titles_with: list[str]
    word: dict[str, re.Pattern[str] | None] = field(init=False)  # title-only cues
    years: dict[str, re.Pattern[str] | None] = field(init=False)  # experience ranges, anywhere
    mgmt: re.Pattern[str] | None = field(init=False)
    any_rule: re.Pattern[str] | None = field(init=False)

    def __post_init__(self) -> None:
        self.word = {c: _rx([p for p in ps if "year" not in p]) for c, ps in self.classes.items()}
        self.years = {c: _rx([p for p in ps if "year" in p]) for c, ps in self.classes.items()}
        self.mgmt = _rx(self.exclude_titles_with)
        self.any_rule = _rx([p for ps in self.classes.values() for p in ps])

    def target_of(self, fine: str) -> str:
        return next(g for g, members in self.class_groups.items() if fine in members)


def load_label_rules(config_dir: Path = CONFIG_DIR) -> LabelRules:
    cfg = load_yaml("labeling.yaml", config_dir)
    if cfg.get("allow_synthetic_labels") is not False:
        raise ConfigError("labeling.yaml: allow_synthetic_labels must be false")
    if cfg.get("exclude_if_ambiguous") is not True:
        raise ConfigError("labeling.yaml: exclude_if_ambiguous must be true (never guess a label)")
    classes = {c: list(map(str, ps)) for c, ps in cfg["rules"].items()}
    groups = cfg.get("class_groups") or {c: [c] for c in classes}
    flat = [c for ms in groups.values() for c in ms]
    if sorted(flat) != sorted(classes):
        raise ConfigError("class_groups must assign every rule class to exactly one group")
    return LabelRules(
        classes=classes,
        exclude_if_ambiguous=True,
        min_per_class=int(cfg["min_confident_examples_per_class"]),
        class_groups={g: list(ms) for g, ms in groups.items()},
        exclude_titles_with=list(cfg.get("exclude_titles_with") or []),
    )


@dataclass(frozen=True)
class Derived:
    label: str | None  # fine class from labeling.yaml rules
    reason: str  # labeled_from_title | labeled_from_description_years | excluded:*
    evidence: tuple[str, ...]


def _hits(rx_by_class: dict[str, re.Pattern[str] | None], text: str) -> dict[str, list[str]]:
    out = {}
    for c, rx in rx_by_class.items():
        if rx is not None:
            found = [m.group(0) for m in rx.finditer(text)]
            if found:
                out[c] = found
    return out


def derive(title: str, description: str | None, rules: LabelRules) -> Derived:
    t, d = normalize(title), normalize(description)
    if rules.mgmt is not None and (m := rules.mgmt.search(t)):
        return Derived(None, "excluded:management_title", (m.group(0),))
    title_hits = _hits(rules.word, t)
    for c, found in _hits(rules.years, t).items():
        title_hits.setdefault(c, []).extend(found)
    if len(title_hits) == 1:
        ((c, found),) = title_hits.items()
        return Derived(c, "labeled_from_title", tuple(found))
    if len(title_hits) > 1:
        ev = tuple(f"{c}:{f}" for c, fs in title_hits.items() for f in fs)
        return Derived(None, "excluded:ambiguous_title", ev)
    desc_hits = _hits(rules.years, d)
    if len(desc_hits) == 1:
        ((c, found),) = desc_hits.items()
        return Derived(c, "labeled_from_description_years", tuple(found))
    if len(desc_hits) > 1:
        ev = tuple(f"{c}:{f}" for c, fs in desc_hits.items() for f in fs)
        return Derived(None, "excluded:ambiguous_description", ev)
    return Derived(None, "excluded:no_signal", ())


def mask(text: str, rules: LabelRules) -> str:
    """Delete every rule pattern (all classes): the model must not see label evidence."""
    out = rules.any_rule.sub(" ", normalize(text)) if rules.any_rule else normalize(text)
    return re.sub(r"[ \t]{2,}", " ", out)


def model_input(description: str | None, rules: LabelRules) -> str:
    """Role sections of the description (requirements first), masked. The title is excluded."""
    desc = description or ""
    secs = tag_sections(desc)
    bounds = [*secs.starts[1:], len(desc)]
    parts: dict[str, list[str]] = {s: [] for s in INPUT_SECTIONS}
    for start, end, label in zip(secs.starts, bounds, secs.labels, strict=True):
        if label in parts:
            parts[label].append(desc[start:end].strip())
    text = "\n".join(p for s in INPUT_SECTIONS for p in parts[s] if p)
    return mask(text, rules).strip()


def dup_group(text: str) -> str:
    """Identical inputs (same job posted for several cities) must never straddle splits."""
    return hashlib.sha1(re.sub(r"\s+", " ", text.lower()).strip().encode()).hexdigest()[:16]


def label_report(rows: list[dict[str, Any]], rules: LabelRules) -> dict[str, Any]:
    labeled = [r for r in rows if r["target"] is not None]
    targets = Counter(r["target"] for r in labeled)
    viable = {g: targets.get(g, 0) >= rules.min_per_class for g in rules.class_groups}
    return {
        "postings_considered": len(rows),
        "fine_class_counts": dict(Counter(r["label"] for r in rows if r["label"])),
        "target_counts": dict(targets),
        "by_reason": dict(Counter(r["reason"] for r in rows)),
        "min_confident_examples_per_class": rules.min_per_class,
        "class_groups": rules.class_groups,
        "viable": viable,
        "trainable": all(viable.values()) and len(viable) >= 2,
        "duplicate_groups": len({r["dup_group"] for r in labeled}),
    }
