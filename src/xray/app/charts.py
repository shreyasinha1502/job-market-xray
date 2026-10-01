"""Inline SVG/HTML charts for the dashboard (no plotting dependency).

Colors are role tokens defined once in CSS (light + dark), from the dataviz reference palette:
one blue series for shares; blue/red poles around a gray zero for changes; status colors only
with an icon and a word. Text always uses ink tokens, never the series color.
"""

from __future__ import annotations

import html
import math
from typing import Any

CSS = """
.viz-root { color-scheme: light;
  --surface-1:#fcfcfb; --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781; --grid:#e1e0d9;
  --axis:#c3c2b7; --series-1:#2a78d6; --pos:#2a78d6; --neg:#e34948; --zero:#c3c2b7;
  --good:#0ca30c; --warning:#fab219; --serious:#ec835a; --critical:#d03b3b;
  --ring:rgba(11,11,11,0.10);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--ink);
  background: var(--surface-1); border: 1px solid var(--ring); border-radius: 10px;
  padding: 14px 16px; margin: 4px 0 10px; }
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root { color-scheme: dark;
    --surface-1:#1a1a19; --ink:#ffffff; --ink-2:#c3c2b7; --grid:#2c2c2a; --axis:#383835;
    --series-1:#3987e5; --pos:#3987e5; --neg:#e66767; --zero:#383835;
    --ring:rgba(255,255,255,0.10); } }
.dark .viz-root, :root[data-theme="dark"] .viz-root { color-scheme: dark;
  --surface-1:#1a1a19; --ink:#ffffff; --ink-2:#c3c2b7; --grid:#2c2c2a; --axis:#383835;
  --series-1:#3987e5; --pos:#3987e5; --neg:#e66767; --zero:#383835;
  --ring:rgba(255,255,255,0.10); }
.viz-root h4 { margin: 0 0 2px; font-size: 15px; font-weight: 600; color: var(--ink); }
.viz-root .sub { margin: 0 0 10px; font-size: 12.5px; color: var(--ink-2); }
.viz-root svg { width: 100%; height: auto; display: block; }
.viz-root svg text { fill: var(--ink-2); font-size: 12px; }
.viz-root svg text.lbl { fill: var(--ink); }
.viz-root svg text.tick { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
.viz-root .row:hover rect.hit { fill: var(--grid); opacity: .45; }
.viz-root .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 10px; }
.viz-root .tile { border: 1px solid var(--ring); border-radius: 8px; padding: 10px 12px; }
.viz-root .tile .k { font-size: 12px; color: var(--ink-2); }
.viz-root .tile .v { font-size: 22px; font-weight: 600; margin-top: 2px; color: var(--ink); }
.viz-root .tile .n { font-size: 11.5px; color: var(--muted); margin-top: 2px; }
.viz-root ul.status { list-style: none; padding: 0; margin: 0; }
.viz-root ul.status li { display: flex; gap: 8px; align-items: baseline; padding: 4px 0;
  font-size: 13.5px; color: var(--ink); }
.viz-root .dot { width: 9px; height: 9px; border-radius: 50%; flex: none;
  transform: translateY(1px); }
.viz-root .banner { font-size: 13.5px; color: var(--ink); }
.viz-root .banner b { font-weight: 600; }
"""

BAR = 14  # bar thickness (<= 24px)
ROW = 28
LABEL_W = 130
PLOT_W = 430
VALUE_W = 70


def _esc(s: Any) -> str:
    return html.escape(str(s), quote=True)


def _nice_max(v: float) -> float:
    for step in (0.1, 0.2, 0.25, 0.5, 1.0):
        if v <= step:
            return step
    return 1.0


def _bar_right(x0: float, y: float, w: float, h: float, r: float = 4) -> str:
    """Rounded data-end on the right, square at the baseline."""
    r = min(r, w, h / 2)
    return (f"M{x0:.1f},{y:.1f} h{w - r:.1f} a{r},{r} 0 0 1 {r},{r} v{h - 2 * r:.1f} "
            f"a{r},{r} 0 0 1 -{r},{r} h-{w - r:.1f} z")  # fmt: skip


def _bar_left(x0: float, y: float, w: float, h: float, r: float = 4) -> str:
    """Rounded data-end on the left (negative values), square at the zero line."""
    r = min(r, w, h / 2)
    return (f"M{x0:.1f},{y:.1f} h-{w - r:.1f} a{r},{r} 0 0 0 -{r},{r} v{h - 2 * r:.1f} "
            f"a{r},{r} 0 0 0 {r},{r} h{w - r:.1f} z")  # fmt: skip


def share_chart(rows: list[dict[str, Any]], n: int, title: str, top: int = 20) -> str:
    """Horizontal bars of share with 95% interval whiskers; hover a row for exact numbers."""
    rows = [r for r in rows if r.get("share") is not None][:top]
    if not rows:
        return (
            f'<div class="viz-root"><h4>{_esc(title)}</h4><p class="sub">No skills found.</p></div>'
        )
    xmax = _nice_max(max(r["ci95"][1] for r in rows))
    sx = PLOT_W / xmax
    height = ROW * len(rows) + 30
    width = LABEL_W + PLOT_W + VALUE_W
    out = [
        f'<svg viewBox="0 0 {width} {height}" style="max-width:{width}px" role="img" aria-label="{_esc(title)}">'
    ]
    step = xmax / 5
    for i in range(6):
        x = LABEL_W + i * step * sx
        out.append(f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{ROW * len(rows):.1f}" '
                   f'stroke="var(--grid)" stroke-width="1"/>')  # fmt: skip
        out.append(f'<text class="tick" x="{x:.1f}" y="{ROW * len(rows) + 16}" '
                   f'text-anchor="middle">{round(i * step * 100)}%</text>')  # fmt: skip
    for i, r in enumerate(rows):
        y = i * ROW + (ROW - BAR) / 2
        cy = y + BAR / 2
        lo, hi = r["ci95"]
        w = max(r["share"] * sx, 1.5)
        tip = (f"{r['skill']}: {r['postings']} of {n} postings = {r['share']:.1%} "
               f"(95% CI {lo:.1%}-{hi:.1%})")  # fmt: skip
        xh = LABEL_W + hi * sx
        out.append(
            f'<g class="row"><title>{_esc(tip)}</title>'
            f'<rect class="hit" x="0" y="{i * ROW}" width="{width}" height="{ROW}" '
            f'fill="transparent"/>'
            f'<text class="lbl" x="{LABEL_W - 10}" y="{cy + 4:.1f}" text-anchor="end">'
            f"{_esc(r['skill'])}</text>"
            f'<path d="{_bar_right(LABEL_W, y, w, BAR)}" fill="var(--series-1)"/>'
            f'<line x1="{LABEL_W + lo * sx:.1f}" y1="{cy:.1f}" x2="{xh:.1f}" y2="{cy:.1f}" '
            f'stroke="var(--ink)" stroke-opacity=".55" stroke-width="1.5"/>'
            f'<line x1="{xh:.1f}" y1="{cy - 4:.1f}" x2="{xh:.1f}" y2="{cy + 4:.1f}" '
            f'stroke="var(--ink)" stroke-opacity=".55" stroke-width="1.5"/>'
            f'<text x="{xh + 6:.1f}" y="{cy + 4:.1f}">{r["share"]:.0%}</text></g>'
        )  # fmt: skip
    out.append(f'<line x1="{LABEL_W}" y1="0" x2="{LABEL_W}" y2="{ROW * len(rows)}" '
               f'stroke="var(--axis)" stroke-width="1"/></svg>')  # fmt: skip
    return (f'<div class="viz-root"><h4>{_esc(title)}</h4><p class="sub">Share of {n} postings '
            f"mentioning each skill. Whiskers: 95% Wilson interval. Hover a row for counts.</p>"
            + "".join(out) + "</div>")  # fmt: skip


def delta_chart(rows: list[dict[str, Any]], title: str, note: str) -> str:
    """Diverging bars for change in share (percentage points) around a zero line."""
    rows = [r for r in rows if r.get("delta_pp") is not None]
    if not rows:
        return f'<div class="viz-root"><h4>{_esc(title)}</h4><p class="sub">{_esc(note)}</p></div>'
    m = max(abs(r["delta_pp"]) for r in rows) or 1.0
    m = math.ceil(m)
    half = PLOT_W / 2
    zero = LABEL_W + half
    height = ROW * len(rows) + 30
    width = LABEL_W + PLOT_W + VALUE_W
    out = [
        f'<svg viewBox="0 0 {width} {height}" style="max-width:{width}px" role="img" aria-label="{_esc(title)}">'
    ]
    for frac in (-1, -0.5, 0.5, 1):
        x = zero + frac * half
        out.append(f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{ROW * len(rows)}" '
                   f'stroke="var(--grid)" stroke-width="1"/>'
                   f'<text class="tick" x="{x:.1f}" y="{ROW * len(rows) + 16}" '
                   f'text-anchor="middle">{frac * m:+g} pp</text>')  # fmt: skip
    for i, r in enumerate(rows):
        y = i * ROW + (ROW - BAR) / 2
        cy = y + BAR / 2
        d = r["delta_pp"]
        w = max(abs(d) / m * half, 1.5)
        path = _bar_right(zero, y, w, BAR) if d >= 0 else _bar_left(zero, y, w, BAR)
        color = "var(--pos)" if d >= 0 else "var(--neg)"
        sig = " *" if r.get("flow_significant") else ""
        tip = (f"{r['skill']}: {r['first_share']:.1%} -> {r['last_share']:.1%} ({d:+.2f} pp)"
               f"{' - significant in the new-posting flow test' if sig else ''}")  # fmt: skip
        tx = zero + w + 6 if d >= 0 else zero - w - 6
        anchor = "start" if d >= 0 else "end"
        out.append(
            f'<g class="row"><title>{_esc(tip)}</title>'
            f'<rect class="hit" x="0" y="{i * ROW}" width="{width}" height="{ROW}" '
            f'fill="transparent"/>'
            f'<text class="lbl" x="{LABEL_W - 10}" y="{cy + 4:.1f}" text-anchor="end">'
            f"{_esc(r['skill'])}</text>"
            f'<path d="{path}" fill="{color}"/>'
            f'<text x="{tx:.1f}" y="{cy + 4:.1f}" text-anchor="{anchor}">{d:+.1f}{sig}</text></g>'
        )  # fmt: skip
    out.append(f'<line x1="{zero}" y1="0" x2="{zero}" y2="{ROW * len(rows)}" '
               f'stroke="var(--zero)" stroke-width="1.5"/></svg>')  # fmt: skip
    return (f'<div class="viz-root"><h4>{_esc(title)}</h4><p class="sub">{_esc(note)}</p>'
            + "".join(out) + "</div>")  # fmt: skip


def tiles(items: list[tuple[str, str, str]]) -> str:
    cells = "".join(
        f'<div class="tile"><div class="k">{_esc(k)}</div><div class="v">{_esc(v)}</div>'
        f'<div class="n">{_esc(n)}</div></div>'
        for k, v, n in items
    )
    return f'<div class="viz-root"><div class="tiles">{cells}</div></div>'


STATUS = {"good": ("var(--good)", "OK"), "warning": ("var(--warning)", "Gap"),
          "critical": ("var(--critical)", "Problem")}  # fmt: skip


def status_list(title: str, items: list[tuple[str, str]]) -> str:
    """(status, text) rows; status shown as dot + word, never color alone."""
    li = "".join(
        f'<li><span class="dot" style="background:{STATUS[s][0]}"></span>'
        f"<b>{STATUS[s][1]}</b><span>{_esc(t)}</span></li>"
        for s, t in items
    )
    return f'<div class="viz-root"><h4>{_esc(title)}</h4><ul class="status">{li}</ul></div>'


def banner(text_html: str) -> str:
    return f'<div class="viz-root"><div class="banner">{text_html}</div></div>'
