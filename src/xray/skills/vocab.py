"""Seed skill vocabulary + matching rules from config/skills.yaml."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from xray.config import CONFIG_DIR, ConfigError, load_yaml

RULE_KEYS = {
    "aliases", "case_sensitive", "needs_context", "exclude_employer_self_mentions",
    "reject_if_next_word", "employer_products",
}  # fmt: skip


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
    reject_if_next_word: dict[str, list[str]]  # exact surface form -> following words that reject
    employer_products: dict[str, list[str]]  # board -> its own product skills (self-mentions)
    sha256: str  # of the effective matching rules, to detect stale extractions

    @property
    def skills(self) -> list[str]:
        return list(self.category)

    def n_surface_forms(self) -> int:
        return sum(map(len, self.lower_forms.values())) + sum(map(len, self.cased_forms.values()))


def accepted_aliases(config_dir: Path) -> dict[str, list[str]]:
    """skill -> variants with status `accepted` in skill_map.yaml (M3 review output)."""
    path = config_dir / "skill_map.yaml"
    if not path.exists():
        return {}
    doc = yaml.safe_load(path.read_text("utf-8")) or {}
    out: dict[str, list[str]] = {}
    for a in doc.get("aliases") or []:
        if a.get("status") == "accepted" and not a.get("stale"):
            out.setdefault(str(a["skill"]).lower(), []).append(str(a["variant"]))
    return out


def load_vocab(config_dir: Path = CONFIG_DIR, *, apply_skill_map: bool = True) -> SkillVocab:
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

    mapped = accepted_aliases(config_dir) if apply_skill_map else {}
    unknown = sorted(set(mapped) - set(category))
    if unknown:
        raise ConfigError(f"skill_map.yaml accepted aliases refer to unknown skills: {unknown}")

    lower_forms: dict[str, list[str]] = {}
    for skill in category:
        forms = [] if skill in cased else [skill]
        forms += [a.strip().lower() for a in aliases.get(skill, [])]
        if skill not in cased:
            forms += [v.strip().lower() for v in mapped.get(skill, [])]
        if forms:
            lower_forms[skill] = sorted(set(forms))
    # variants of exact-spelling skills stay exact-spelling (e.g. "GoLang" stays cased)
    cased_forms = {
        s: sorted({f.strip() for f in forms} | {v.strip() for v in mapped.get(s, [])})
        for s, forms in cased.items()
    }

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
    self_mentions = bool(raw.get("exclude_employer_self_mentions", False))
    reject_next = {
        str(form): [w.lower() for w in words]
        for form, words in (raw.get("reject_if_next_word") or {}).items()
    }
    unknown_forms = sorted(
        set(reject_next) - all_cased - {f for fs in lower_forms.values() for f in fs}
    )
    if unknown_forms:
        raise ConfigError(f"reject_if_next_word refers to unknown surface forms: {unknown_forms}")
    products = {
        str(b).lower(): [p.lower() for p in ps]
        for b, ps in (raw.get("employer_products") or {}).items()
    }
    unknown = sorted({p for ps in products.values() for p in ps} - set(category))
    if unknown:
        raise ConfigError(f"employer_products refers to unknown skills: {unknown}")
    # Hash what matching actually uses, so regenerated metadata never forces a re-extraction.
    effective = {
        "category": category, "lower": lower_forms, "cased": cased_forms,
        "needs_context": nc.model_dump(), "self_mentions": self_mentions,
        "reject_next": reject_next, "products": products,
    }  # fmt: skip
    return SkillVocab(
        category=category,
        lower_forms=lower_forms,
        cased_forms=cased_forms,
        needs_context=nc,
        exclude_employer_self_mentions=self_mentions,
        reject_if_next_word=reject_next,
        employer_products=products,
        sha256=hashlib.sha256(json.dumps(effective, sort_keys=True).encode()).hexdigest(),
    )
