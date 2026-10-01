"""Seed skill vocabulary + matching rules from config/skills.yaml."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from xray.config import CONFIG_DIR, ConfigError, load_yaml

RULE_KEYS = {"aliases", "case_sensitive", "needs_context", "exclude_employer_self_mentions"}


class NeedsContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    forms: list[str]
    window: int = Field(ge=1)
    reject_if_next: list[str]
    cue_words_before: list[str]
    cue_words_after: list[str]
    context_only_words: list[str]


@dataclass(frozen=True)
class SkillVocab:
    category: dict[str, str]  # canonical skill -> category
    lower_forms: dict[str, list[str]]  # matched case-insensitively
    cased_forms: dict[str, list[str]]  # matched on exact spelling only
    needs_context: NeedsContext
    exclude_employer_self_mentions: bool
    sha256: str  # of the skills.yaml bytes, to detect stale extractions

    @property
    def skills(self) -> list[str]:
        return list(self.category)

    def n_surface_forms(self) -> int:
        return sum(map(len, self.lower_forms.values())) + sum(map(len, self.cased_forms.values()))


def load_vocab(config_dir: Path = CONFIG_DIR) -> SkillVocab:
    raw = load_yaml("skills.yaml", config_dir)
    category: dict[str, str] = {}
    for cat, terms in raw.items():
        if cat in RULE_KEYS:
            continue
        if not isinstance(terms, list):
            raise ConfigError(f"skills.yaml category {cat!r} must be a list")
        for t in terms:
            skill = str(t).strip().lower()
            if skill in category:
                raise ConfigError(f"skill {skill!r} listed in both {category[skill]} and {cat}")
            category[skill] = cat

    aliases = {k.lower(): v for k, v in (raw.get("aliases") or {}).items()}
    cased = {k.lower(): v for k, v in (raw.get("case_sensitive") or {}).items()}
    for section, mapping in (("aliases", aliases), ("case_sensitive", cased)):
        unknown = sorted(set(mapping) - set(category))
        if unknown:
            raise ConfigError(f"skills.yaml {section} refers to unknown skills: {unknown}")

    lower_forms: dict[str, list[str]] = {}
    for skill in category:
        forms = [] if skill in cased else [skill]
        forms += [a.strip().lower() for a in aliases.get(skill, [])]
        if forms:
            lower_forms[skill] = sorted(set(forms))
    cased_forms = {s: sorted(set(f.strip() for f in forms)) for s, forms in cased.items()}

    owner: dict[str, str] = {}
    for skill, forms in [
        *lower_forms.items(),
        *((s, [f.lower() for f in fs]) for s, fs in cased_forms.items()),
    ]:
        for f in forms:
            if owner.setdefault(f, skill) != skill:
                raise ConfigError(f"surface form {f!r} maps to both {owner[f]} and {skill}")

    nc = NeedsContext.model_validate(raw.get("needs_context") or {})
    all_cased = {f for fs in cased_forms.values() for f in fs}
    if not set(nc.forms) <= all_cased:
        raise ConfigError(f"needs_context forms must be case_sensitive forms: {nc.forms}")
    return SkillVocab(
        category=category,
        lower_forms=lower_forms,
        cased_forms=cased_forms,
        needs_context=nc,
        exclude_employer_self_mentions=bool(raw.get("exclude_employer_self_mentions", False)),
        sha256=hashlib.sha256((config_dir / "skills.yaml").read_bytes()).hexdigest(),
    )
