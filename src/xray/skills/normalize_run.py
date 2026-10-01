"""`xray normalize`: harvest -> embed -> cluster -> proposals -> skill_map.yaml + readable dump."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from xray.config import CONFIG_DIR
from xray.skills.candidates import Term, frequent, harvest, source_counts
from xray.skills.extract import norm_key
from xray.skills.normalize import (
    SKILL_MAP_FILE,
    dbscan,
    embed,
    generated_meta,
    load_norm_config,
    merge_reviewed,
    near_misses,
    propose,
    read_skill_map,
    sensitivity,
    write_skill_map,
)
from xray.skills.pipeline import location_terms
from xray.skills.vocab import load_vocab
from xray.store import Store


def _md_table(rows: list[dict[str, Any]], cols: list[str]) -> list[str]:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")).replace("|", "/") for c in cols) + " |")
    return out


def render_dump(day: date, meta: dict, sens: list[dict], doc: dict, misses: list[dict],
                anchors: list[Term]) -> str:  # fmt: skip
    L = [f"# Skill normalization dump — snapshot {day}", ""]
    L += [f"Model `{meta['model']}`, DBSCAN eps={meta['eps']} (cosine distance, i.e. similarity "
          f">= {1 - meta['eps']:.2f}), min_cluster_size={meta['min_cluster_size']}, "
          f"{meta['candidate_terms']} candidate terms.", ""]  # fmt: skip
    L += [
        "## Threshold sensitivity",
        "",
        "Why this eps: wider radii start merging different skills or unrelated acronyms.",
        "",
    ]
    L += _md_table(sens, list(sens[0])) + [""]
    L += ["## Vocabulary skills and the spellings observed for them", ""]
    L += _md_table(
        [{"skill": t.anchor,
          "spellings": ", ".join(f"{s} ({n})" for s, n in t.surfaces.most_common(6)),
          "postings": len(t.postings)} for t in sorted(anchors, key=lambda t: t.anchor or "")],
        ["skill", "spellings", "postings"],
    ) + [""]  # fmt: skip
    L += ["## Alias proposals for vocabulary skills", ""]
    L += _md_table(
        doc["aliases"], ["variant", "skill", "status", "why", "cos", "postings", "boards"]
    )
    L += ["", "## Conflicts (2+ vocabulary skills in one cluster; never merged)", ""]
    L += (
        _md_table(doc["conflicts"], ["skills", "members", "min_cos"])
        if doc["conflicts"]
        else ["None at this eps."]
    )
    L += ["", "## Proposed new skill groups (not counted; candidates for skills.yaml)", ""]
    for g in doc["new_skill_groups"]:
        L.append(f"### {g['canonical']}")
        L += _md_table(
            [{**m, "spellings": ", ".join(f"{k} ({v})" for k, v in m["spellings"].items())}
             for m in g["members"]],
            ["term", "spellings", "cos_to_canonical", "lexical_support", "postings", "boards"],
        ) + [""]  # fmt: skip
    L += ["## Phrases that already count through a vocabulary skill", ""]
    L += _md_table(doc.get("covered_phrases", []), ["term", "already_counted_as", "postings"])
    L += ["", "## Near misses (similarity just below the threshold; NOT merged)", ""]
    L += _md_table(misses[:80], ["a", "b", "cos", "lexical_support", "anchors"])
    return "\n".join(L) + "\n"


def run_normalize(
    store: Store | None = None, day: date | None = None, config_dir: Path = CONFIG_DIR
) -> dict[str, Any]:
    store = store or Store()
    day = day or store.dates("sightings")[-1]
    cfg = load_norm_config(config_dir)
    vocab = load_vocab(config_dir, apply_skill_map=False)
    exclude = {norm_key(x) for x in location_terms(config_dir)}
    terms = frequent(
        harvest(store, day, vocab, exclude_terms=exclude), cfg.min_postings, cfg.min_boards
    )
    emb, model_id = embed(terms, cfg)
    labels = dbscan(emb, cfg.eps, cfg.min_samples)
    doc = merge_reviewed(propose(terms, emb, labels, vocab), read_skill_map(config_dir))
    meta = generated_meta(day, cfg, model_id, len(terms))
    sens = sensitivity(terms, emb, cfg.min_samples)
    misses = near_misses(terms, emb, cfg.eps)

    write_skill_map(config_dir / SKILL_MAP_FILE, doc, meta)
    out = store.root / "skill_map"
    out.mkdir(parents=True, exist_ok=True)
    md = out / f"clusters_{day.isoformat()}.md"
    md.write_text(
        render_dump(day, meta, sens, doc, misses, [t for t in terms if t.anchor]), "utf-8"
    )
    js = out / f"clusters_{day.isoformat()}.json"
    js.write_text(json.dumps({"generated": meta, "sensitivity": sens, **doc, "near_misses": misses,
                              "candidate_sources": source_counts(terms)},
                             indent=1, ensure_ascii=False, default=str), "utf-8")  # fmt: skip
    return {
        "meta": meta,
        "sensitivity": sens,
        "doc": doc,
        "files": [config_dir / SKILL_MAP_FILE, md, js],
    }
