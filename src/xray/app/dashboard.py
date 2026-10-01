"""Gradio dashboard: skill demand snapshot + trends (with real history windows), the seniority
classifier on a pasted job description, and a visible coverage / gaps panel."""

from __future__ import annotations

import html
import logging
from typing import Any

import gradio as gr

from xray.app import charts
from xray.app.data import DashboardData, load
from xray.app.model import ModelUnavailable, predictor

log = logging.getLogger(__name__)

ALL = "all in-region postings"


def _history_line(h: dict[str, Any]) -> str:
    n = h["n_snapshots"]
    if n < 2:
        return (f"<b>History: {n} real snapshot day</b> ({h['first']}). Trends need at least "
                "2 days and are never backfilled, so this page shows a point-in-time snapshot. "
                "A new day is added by the scheduled daily run.")  # fmt: skip
    missing = f", {len(h['missing_dates'])} day(s) missing" if h["missing_dates"] else ""
    return (f"<b>History: {n} real snapshot days</b>, {h['first']} to {h['last']} "
            f"(span {h['span_days']} days{missing}).")  # fmt: skip


def snapshot_view(d: DashboardData, scope: str) -> tuple[str, list[list[Any]]]:
    s = d.scope(scope)["snapshot"]
    title = f"Skills in demand: {scope}, {s['day']}"
    table = [
        [r["skill"], r["category"], r["postings"], f"{r['share']:.1%}",
         f"{r['ci95'][0]:.1%} - {r['ci95'][1]:.1%}"]
        for r in s["skills"]
    ]  # fmt: skip
    return charts.share_chart(s["skills"], s["postings"], title), table


def trend_view(d: DashboardData, scope: str) -> tuple[str, str, str]:
    t = d.scope(scope)["trend"]
    head = charts.banner(_history_line(t["history"]))
    if t["status"] != "ok":
        msg = charts.banner(f"<b>No trend for this scope yet.</b> {html.escape(t['reason'])}")
        return head, msg, ""
    excluded = t["panel"]["boards_excluded"]
    note = (f"Change in share of open postings, {t['history']['first']} to "
            f"{t['history']['last']}, over {t['panel']['boards_compared']} employers fetched OK "
            f"on every day{f' ({len(excluded)} excluded)' if excluded else ''}. Descriptive: "
            "the same postings stay open across days. * = significant in the new-posting flow "
            "test (Fisher exact, Benjamini-Hochberg 5%).")  # fmt: skip
    f = t["flow"]
    flow = charts.banner(
        f"<b>Flow test (new postings):</b> {html.escape(f['status'])} "
        f"(early half {f['early']}, late half {f['late']} new postings)"
        + (f". {html.escape(f['reason'])}" if f.get("reason") else "")
    )
    chart = charts.delta_chart(t["risers"], "Top risers", note) + charts.delta_chart(
        t["fallers"], "Top fallers", note
    )
    return head, chart, flow


def classify(text: str) -> tuple[dict[str, float] | None, str]:
    if not text or len(text.strip()) < 200:
        return None, "Paste a full job description (at least a few sentences)."
    try:
        out = predictor().predict(text)
    except ModelUnavailable as e:
        return None, f"**Model not available:** {e}"
    if "error" in out:
        return None, f"**Could not classify:** {out['error']}"
    return out["probabilities"], (
        f"**Prediction: {out['prediction']}** (model `{out['model']}`, "
        f"{out['input_tokens_used']} tokens of the role sections used).\n\n"
        f"*{out['scope_note']}.* The title and every seniority keyword are removed before the "
        "model sees the text, so it has to infer the level from the duties and requirements. "
        "On held-out real postings it reaches macro-F1 ≈ 0.69, the same as the TF-IDF baseline "
        "(see About). The probabilities are not calibrated, so read them as a ranking, not as "
        "a likelihood. Treat it as a demo, not as a decision about a person."
    )


def quality_view(d: DashboardData) -> str:
    q, rep = d.quality, d.skills_report or {}
    last = q["days"][-1] if q["days"] else {}
    age = q.get("posting_age_latest") or {}
    cov = (rep.get("coverage") or {}).get("all_in_region") or {}
    out = charts.tiles([
        ("Snapshot days", str(q["history"]["n_snapshots"]),
         f"{q['history']['first']} .. {q['history']['last']}"),
        ("Employers fetched OK", f"{last.get('boards_ok', 0)} / {q['panel']['boards']}",
         f"latest day {last.get('day', '-')}"),
        ("India postings", f"{last.get('in_region_postings', 0):,}", "open on the latest day"),
        ("With >= 1 skill", f"{cov.get('coverage', 0):.0%}" if cov else "-",
         "role sections, seed vocabulary"),
        ("Median days open", str(age.get("median_days_open", "-")),
         f"{age.get('share_open_over_90_days', 0):.0%} open > 90 days"),
    ])  # fmt: skip
    gaps = [("warning", g) for g in q["gaps"]] or [("good", "No known gaps.")]
    out += charts.status_list("Known gaps", gaps)
    by_role = (rep.get("coverage") or {}).get("by_role") or {}
    if by_role:
        rows = "".join(
            f"<li><span>{html.escape(r)}: {v['with_skill']}/{v['postings']} postings "
            f"({v['coverage']:.0%})</span></li>"
            for r, v in by_role.items()
            if v["postings"]
        )
        out += (f'<div class="viz-root"><h4>Skill match coverage by role</h4><p class="sub">'
                "Share of postings with at least one vocabulary skill. Low coverage outside "
                "tech roles is expected (sales, finance, operations).</p>"
                f'<ul class="status">{rows}</ul></div>')  # fmt: skip
    vg = (rep.get("vocab_gaps_observed") or {}).get("terms") or []
    if vg:
        terms = ", ".join(f"{t['term']} ({t['postings']})" for t in vg[:10])
        out += charts.banner(
            f"<b>Not in the seed vocabulary (observed, not counted):</b> {html.escape(terms)}"
        )
    return out


ABOUT = """
### What this is
Real job postings from the public job-board APIs (Greenhouse, Lever, Ashby) of a **fixed panel of
55 employers**, India locations only, fetched daily. It covers this panel, **not the whole Indian
job market**, and the panel skews to product/SaaS companies in Bengaluru.

### Honesty rules
- No synthetic or backfilled data. Every posting keeps provenance to the raw API response.
- Trends need at least 2 real snapshot days, and each one states its history window. Board
  failures are excluded from comparisons rather than read as demand drops.
- Skills come from a seed vocabulary with spelling aliases and context rules. Terms outside it
  are reported, not counted. Match coverage is shown, including where it is low.
- The classifier is trained on rule-derived labels. It was scoped down to senior vs
  below_senior because the other classes had too few real examples.
"""


def build(d: DashboardData | None = None) -> gr.Blocks:
    d = d or load()
    scopes = d.scope_labels()
    with gr.Blocks(title="Job Market X-Ray") as demo:
        gr.Markdown(f"# Job Market X-Ray\nTech-skill demand in real job postings · data as of "
                    f"**{d.as_of}**")  # fmt: skip
        gr.HTML(charts.banner(_history_line(d.history)))
        with gr.Tab("Skills now"):
            s_scope = gr.Dropdown(scopes, value=ALL, label="Role or city")
            s_chart = gr.HTML()
            s_table = gr.Dataframe(
                headers=["skill", "category", "postings", "share", "95% CI"],
                interactive=False, label="Table view",
            )  # fmt: skip
        with gr.Tab("Trends"):
            t_scope = gr.Dropdown(scopes, value=ALL, label="Role or city")
            t_head, t_chart, t_flow = gr.HTML(), gr.HTML(), gr.HTML()
        with gr.Tab("Classify a job description"):
            jd = gr.Textbox(lines=14, label="Paste a job description",
                            placeholder="Paste the full posting text here...")  # fmt: skip
            go = gr.Button("Classify seniority", variant="primary")
            probs = gr.Label(label="Probability", num_top_classes=2)
            expl = gr.Markdown()
        with gr.Tab("Coverage & gaps"):
            gr.HTML(quality_view(d))
        with gr.Tab("About"):
            gr.Markdown(ABOUT)
            if d.model_card:
                with gr.Accordion("Model card", open=False):
                    gr.Markdown(d.model_card)

        s_scope.change(lambda sc: snapshot_view(d, sc), s_scope, [s_chart, s_table])
        t_scope.change(lambda sc: trend_view(d, sc), t_scope, [t_head, t_chart, t_flow])
        go.click(classify, jd, [probs, expl])
        demo.load(lambda: (*snapshot_view(d, ALL), *trend_view(d, ALL)), None,
                  [s_chart, s_table, t_head, t_chart, t_flow])  # fmt: skip
    return demo
