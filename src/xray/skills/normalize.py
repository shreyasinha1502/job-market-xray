"""M3: semantic normalization of skill surface forms into a reviewable canonical skill map.

Pipeline (one snapshot):
  1. harvest candidate terms from real postings (candidates.py); spelling variants share a key
  2. embed each term with sentence-transformers (project.yaml, pinned revision)
  3. DBSCAN on cosine distance (project.yaml) proposes semantic merges
  4. decisions, all written out for review:
       - a cluster with one vocabulary skill proposes aliases for it
       - a cluster with two or more vocabulary skills is a conflict and never merges
       - a cluster with none is a proposed new skill group (reported, not counted)
     A proposal is auto-accepted only with lexical support (same key, containment, close
     spelling); embeddings alone confuse short acronyms (measured: SASE~SAST 0.84,
     Excel~Google Sheets 0.86), so everything else waits for review.

Only aliases with status `accepted` in config/skill_map.yaml change extraction.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from xray.config import CONFIG_DIR, ConfigError, load_yaml
from xray.skills.candidates import Term, lexical_key
from xray.skills.vocab import SkillVocab

SKILL_MAP_FILE = "skill_map.yaml"
EPS_GRID = (0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30)
_VENDOR_PREFIX = re.compile(r"^(microsoft|ms|google|apache|amazon|advanced)(?=[a-z0-9])")
_ACRONYM_SKIP = {"of", "and", "for", "the", "&"}
REVIEW_STATUSES = {"accepted", "rejected"}


@dataclass(frozen=True)
class NormConfig:
    model: str
    revision: str | None
    eps: float
    min_samples: int
    min_postings: int
    min_boards: int


def load_norm_config(config_dir: Path = CONFIG_DIR) -> NormConfig:
    cfg = load_yaml("project.yaml", config_dir)
    cl = cfg.get("clustering") or {}
    if cl.get("algorithm") != "dbscan":
        raise ConfigError(
            "clustering.algorithm must be 'dbscan' for variant merging: k-means assigns every "
            "term to some cluster, which would force merges between unrelated skills"
        )
    norm = cfg.get("normalization") or {}
    return NormConfig(
        model=cfg["embeddings_model"],
        revision=cfg.get("embeddings_revision"),
        eps=float(cl["eps"]),
        min_samples=int(cl["min_cluster_size"]),
        min_postings=int(norm.get("min_postings", 3)),
        min_boards=int(norm.get("min_boards", 3)),
    )


# ---------------------------------------------------------------- lexical evidence


def _core(surface: str) -> str:
    k = _VENDOR_PREFIX.sub("", lexical_key(surface))
    k = re.sub(r"\d+$", "", k)  # versions: oauth2, oauth20, html5
    if len(k) > 4 and k.endswith("js"):  # React.js ~ React (only as lexical *support*)
        k = k[:-2]
    return k[:-1] if len(k) > 3 and k.endswith("s") else k  # plurals: APIs, SLAs


def lexically_supported(a: str, b: str) -> bool:
    ka, kb = _core(a), _core(b)
    if not ka or not kb:
        return False
    if ka == kb:
        return True
    short, long_ = sorted((ka, kb), key=len)
    if len(short) >= 3 and long_.startswith(short) and len(long_) - len(short) <= 4:
        return True  # GitLab ~ GitLab CI; not Linux ~ Linux systems (suffix too long)
    return SequenceMatcher(None, ka, kb).ratio() >= 0.85


def acronym_of(phrase: str) -> str | None:
    words = [w for w in re.split(r"[\s\-/]+", phrase) if w and w.lower() not in _ACRONYM_SKIP]
    if len(words) < 2 or not all(w[0].isalpha() for w in words):
        return None
    return "".join(w[0] for w in words).lower()


def _covered_by_existing(variant: str, skill: str, vocab: SkillVocab) -> bool:
    forms = [*vocab.lower_forms.get(skill, [])]
    cased = vocab.cased_forms.get(skill, [])
    v = variant.lower()
    if any(re.search(rf"(?<![a-z0-9]){re.escape(f)}(?![a-z0-9])", v) for f in forms):
        return True
    return any(re.search(rf"(?<![A-Za-z0-9]){re.escape(f)}(?![A-Za-z0-9])", variant) for f in cased)


# ---------------------------------------------------------------- embedding + clustering


def embed(terms: list[Term], cfg: NormConfig) -> tuple[np.ndarray, str]:
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(cfg.model, device="cpu", revision=cfg.revision)
    emb = model.encode(
        [t.display for t in terms],
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=False,
    )
    return np.asarray(emb, dtype=np.float32), f"{cfg.model}@{cfg.revision or 'unpinned'}"


def dbscan(emb: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    from sklearn.cluster import DBSCAN

    return DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(emb)


def _groups(labels: np.ndarray) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for i, lbl in enumerate(labels):
        if lbl >= 0:
            out.setdefault(int(lbl), []).append(i)
    return out


def sensitivity(terms: list[Term], emb: np.ndarray, min_samples: int) -> list[dict[str, Any]]:
    rows = []
    for eps in EPS_GRID:
        groups = _groups(dbscan(emb, eps, min_samples))
        conflicts = sum(
            len({terms[i].anchor for i in g if terms[i].anchor}) >= 2 for g in groups.values()
        )
        unsupported = 0
        for g in groups.values():
            head = max(g, key=lambda i: len(terms[i].postings))
            unsupported += sum(
                not lexically_supported(terms[i].display, terms[head].display)
                for i in g
                if i != head
            )
        rows.append({
            "eps": eps,
            "min_similarity": round(1 - eps, 2),
            "clusters": len(groups),
            "terms_in_clusters": sum(map(len, groups.values())),
            "multi_skill_conflicts": conflicts,
            "merges_without_lexical_support": unsupported,
        })  # fmt: skip
    return rows


# ---------------------------------------------------------------- proposals


def _stats(t: Term) -> dict[str, Any]:
    return {"postings": len(t.postings), "boards": len(t.boards), "sources": sorted(t.sources)}


def propose(
    terms: list[Term], emb: np.ndarray, labels: np.ndarray, vocab: SkillVocab
) -> dict[str, Any]:
    sims = emb @ emb.T
    aliases: list[dict[str, Any]] = []
    seen_alias: set[tuple[str, str]] = set()

    def alias(variant: str, skill: str, why: str, cos: float | None, t: Term, lexical: bool):
        k = (variant.lower(), skill)
        if k in seen_alias:
            return
        seen_alias.add(k)
        if _covered_by_existing(variant, skill, vocab):
            status = "already_matched"
        else:
            status = "accepted" if lexical else "needs_review"
        aliases.append({
            "variant": variant, "skill": skill, "status": status, "why": why,
            "cos": None if cos is None else round(float(cos), 3), **_stats(t),
        })  # fmt: skip

    # 1. spelling variants of vocabulary skills that the matcher did not catch. Exact-spelling
    # skills (Go, R, Spark) reject other casings on purpose, so those always need review.
    for t in terms:
        if t.anchor:
            auto = t.anchor not in vocab.cased_forms
            for surface in t.surfaces:
                if surface not in t.vocab_surfaces:
                    why = "same lexical key as a vocabulary form" + (
                        "" if auto else " (exact-spelling skill: casing variant needs review)"
                    )
                    alias(surface, t.anchor, why, None, t, auto)

    groups = _groups(labels)
    conflicts, new_groups, covered = [], [], []

    def covering_skill(t: Term) -> str | None:
        """A phrase like "Strong SQL skills" already counts as sql through the vocabulary."""
        return next((sk for sk in vocab.skills if _covered_by_existing(t.display, sk, vocab)), None)

    for g in groups.values():
        members = sorted(g, key=lambda i: -len(terms[i].postings))
        anchors = sorted({terms[i].anchor for i in members if terms[i].anchor})
        if len(anchors) >= 2:
            conflicts.append({
                "skills": anchors,
                "members": [terms[i].display for i in members],
                "min_cos": round(float(min(sims[i, j] for i in g for j in g if i != j)), 3),
            })  # fmt: skip
        elif len(anchors) == 1:
            head = next(i for i in members if terms[i].anchor)
            for i in members:
                if not terms[i].anchor:
                    lex = lexically_supported(terms[i].display, terms[head].display)
                    alias(terms[i].display, anchors[0], "semantic cluster", sims[i, head],
                          terms[i], lex)  # fmt: skip
        else:
            members = [i for i in members if not _note_covered(terms[i], covering_skill, covered)]
            if not members:
                continue
            head = members[0]
            new_groups.append({
                "canonical": terms[head].display,
                "members": [
                    {"term": terms[i].display, "cos_to_canonical": round(float(sims[i, head]), 3),
                     "lexical_support": i == head
                     or lexically_supported(terms[i].display, terms[head].display),
                     "spellings": dict(terms[i].surfaces.most_common(5)), **_stats(terms[i])}
                    for i in members
                ],
                "status": "proposed",
            })  # fmt: skip

    # spelling-only groups (React / ReactJS / React.js) that DBSCAN had no reason to touch
    clustered = {i for g in groups.values() for i in g}
    for i, t in enumerate(terms):
        if (
            i not in clustered
            and not t.anchor
            and len(t.surfaces) > 1
            and not _note_covered(t, covering_skill, covered)
        ):
            new_groups.append({
                "canonical": t.display,
                "members": [{"term": t.display, "cos_to_canonical": 1.0, "lexical_support": True,
                             "spellings": dict(t.surfaces.most_common(5)), **_stats(t)}],
                "status": "proposed",
            })  # fmt: skip

    # 3. acronym expansions ("Amazon Web Services" -> aws): embeddings score these low (0.66)
    anchor_keys = {lexical_key(s): s for s in vocab.skills}
    for idx, t in enumerate(terms):
        acr = acronym_of(t.display)
        if not t.anchor and acr and acr in anchor_keys:
            head = next((j for j, u in enumerate(terms) if u.anchor == anchor_keys[acr]), None)
            alias(t.display, anchor_keys[acr], "acronym expansion",
                  None if head is None else sims[idx, head], t, False)  # fmt: skip

    new_groups.sort(key=lambda g: -sum(m["postings"] for m in g["members"]))
    aliases.sort(key=lambda a: (a["skill"], -a["postings"]))
    return {
        "aliases": aliases,
        "conflicts": conflicts,
        "new_skill_groups": new_groups,
        "covered_phrases": sorted(covered, key=lambda c: -c["postings"]),
    }


def _note_covered(t: Term, covering_skill, covered: list[dict[str, Any]]) -> bool:
    skill = covering_skill(t)
    if skill is not None and not t.anchor:
        covered.append({"term": t.display, "already_counted_as": skill, **_stats(t)})
        return True
    return False


def near_misses(terms: list[Term], emb: np.ndarray, eps: float, low: float = 0.75) -> list[dict]:
    """Pairs just below the merge threshold: what the chosen eps deliberately did not merge."""
    sims = emb @ emb.T
    hi = 1 - eps
    out = []
    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            if low <= sims[i, j] < hi:
                out.append({
                    "a": terms[i].display, "b": terms[j].display,
                    "cos": round(float(sims[i, j]), 3),
                    "lexical_support": lexically_supported(terms[i].display, terms[j].display),
                    "anchors": [x for x in (terms[i].anchor, terms[j].anchor) if x],
                })  # fmt: skip
    return sorted(out, key=lambda r: -r["cos"])


# ---------------------------------------------------------------- skill_map.yaml


def merge_reviewed(proposals: dict[str, Any], existing: dict[str, Any]) -> dict[str, Any]:
    """Keep human/agent review decisions (reviewed: true) across regenerations."""
    reviewed = {
        (a["variant"].lower(), a["skill"]): a
        for a in existing.get("aliases") or []
        if a.get("reviewed")
    }
    out = []
    for a in proposals["aliases"]:
        prev = reviewed.pop((a["variant"].lower(), a["skill"]), None)
        if prev is not None:
            a = {**a, "status": prev["status"], "reviewed": True, "note": prev.get("note", "")}
        out.append(a)
    # reviewed decisions whose variant no longer appears in this snapshot are kept, marked stale
    out += [{**a, "stale": True} for a in reviewed.values()]
    groups_reviewed = {
        g["canonical"].lower(): g
        for g in existing.get("new_skill_groups") or []
        if g.get("reviewed")
    }
    groups = []
    for g in proposals["new_skill_groups"]:
        prev = groups_reviewed.get(g["canonical"].lower())
        groups.append(
            {**g, "status": prev["status"], "reviewed": True, "note": prev.get("note", "")}
            if prev
            else g
        )
    return {**proposals, "aliases": out, "new_skill_groups": groups}


def write_skill_map(path: Path, doc: dict[str, Any], meta: dict[str, Any]) -> None:
    header = (
        "# Canonical skill map (M3). Generated by `xray normalize`, then reviewed.\n"
        "# aliases: variant -> vocabulary skill. ONLY status: accepted changes extraction.\n"
        "#   accepted | rejected (reviewed) | needs_review | already_matched (vocab counts it)\n"
        "# new_skill_groups: proposals for config/skills.yaml. Never counted from this file.\n"
        "# conflicts: clusters holding 2+ different vocabulary skills. Never merged.\n"
    )
    body = {
        "generated": meta,
        "aliases": doc["aliases"],
        "conflicts": doc["conflicts"],
        "new_skill_groups": [
            {
                "canonical": g["canonical"],
                "status": g["status"],
                **({"reviewed": True, "note": g.get("note", "")} if g.get("reviewed") else {}),
                "members": [m["term"] for m in g["members"]],
                "postings_max_member": max(m["postings"] for m in g["members"]),
                "boards_max_member": max(m["boards"] for m in g["members"]),
            }
            for g in doc["new_skill_groups"]
        ],
    }
    path.write_text(
        header + yaml.safe_dump(body, sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8",
    )


def read_skill_map(config_dir: Path = CONFIG_DIR) -> dict[str, Any]:
    p = config_dir / SKILL_MAP_FILE
    return (yaml.safe_load(p.read_text("utf-8")) or {}) if p.exists() else {}


def generated_meta(day: date, cfg: NormConfig, model_id: str, n_terms: int) -> dict[str, Any]:
    return {
        "snapshot": day.isoformat(),
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": model_id,
        "eps": cfg.eps,
        "min_cluster_size": cfg.min_samples,
        "candidate_terms": n_terms,
    }
