/* Job Market X-Ray frontend. Plain JS + hand-built SVG (no chart library).
   Every number comes from /api/dashboard, which is built from committed real data. */
"use strict";

const $ = (s, r = document) => r.querySelector(s);
const NS = "http://www.w3.org/2000/svg";
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x, d = 0) => (x == null ? "–" : (100 * x).toFixed(d) + "%");
const num = (x) => (x == null ? "–" : Number(x).toLocaleString("en-IN"));
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches || new URLSearchParams(location.search).has("still");

let D = null;
const state = { scope: "all", cat: "all", top: 15, trendScope: "all", example: null };

/* ---------------------------------------------------------------- tooltip */
const tip = $("#tip");
function placeTip(x, y) {
  const r = tip.getBoundingClientRect();
  let left = x + 14, top = y + 14;
  if (left + r.width > innerWidth - 8) left = x - r.width - 14;
  if (top + r.height > innerHeight - 8) top = y - r.height - 14;
  tip.style.left = Math.max(8, left) + "px";
  tip.style.top = Math.max(8, top) + "px";
}
document.addEventListener("mouseover", (e) => {
  const t = e.target.closest("[data-tip]");
  if (!t) return;
  tip.innerHTML = t.getAttribute("data-tip");
  tip.hidden = false;
  placeTip(e.clientX, e.clientY);
});
document.addEventListener("mousemove", (e) => { if (!tip.hidden) placeTip(e.clientX, e.clientY); });
document.addEventListener("mouseout", (e) => {
  const t = e.target.closest("[data-tip]");
  if (t && !t.contains(e.relatedTarget)) tip.hidden = true;
});
document.addEventListener("focusin", (e) => {
  const t = e.target.closest("[data-tip]");
  if (!t) return;
  tip.innerHTML = t.getAttribute("data-tip");
  tip.hidden = false;
  const r = t.getBoundingClientRect();
  placeTip(r.left + r.width / 2, r.bottom);
});
document.addEventListener("focusout", () => { tip.hidden = true; });

/* ---------------------------------------------------------------- svg helpers */
function el(tag, attrs = {}, parent) {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
}
function text(parent, x, y, s, cls, anchor = "start", extra = {}) {
  const t = el("text", { x, y, class: cls, "text-anchor": anchor, ...extra }, parent);
  t.textContent = s;
  return t;
}
function svg(container, w, h, label) {
  container.innerHTML = "";
  return el("svg", { width: w, height: h, viewBox: `0 0 ${w} ${h}`, role: "img", "aria-label": label }, container);
}
// rounded data-end, square at the baseline (dataviz mark spec)
function barRight(x, y, w, h, r = 4) {
  r = Math.min(r, w, h / 2);
  return `M${x},${y}h${w - r}a${r},${r} 0 0 1 ${r},${r}v${h - 2 * r}a${r},${r} 0 0 1 -${r},${r}h-${w - r}z`;
}
function barLeft(x, y, w, h, r = 4) {
  r = Math.min(r, w, h / 2);
  return `M${x},${y}h-${w - r}a${r},${r} 0 0 0 -${r},${r}v${h - 2 * r}a${r},${r} 0 0 0 ${r},${r}h${w - r}z`;
}
function barUp(x, yBase, w, h, r = 4) {
  r = Math.min(r, h, w / 2);
  return `M${x},${yBase}v-${h - r}a${r},${r} 0 0 1 ${r},-${r}h${w - 2 * r}a${r},${r} 0 0 1 ${r},${r}v${h - r}z`;
}
function niceMax(v) {
  for (const s of [0.05, 0.1, 0.2, 0.25, 0.4, 0.5, 0.6, 0.8, 1]) if (v <= s) return s;
  return 1;
}
function hexToRgb(h) {
  h = h.replace("#", "");
  if (h.length === 3) h = h.split("").map((c) => c + c).join("");
  const n = parseInt(h, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
function mix(a, b, t) {
  const A = hexToRgb(a), B = hexToRgb(b);
  const c = A.map((v, i) => Math.round(v + (B[i] - v) * t));
  return { css: `rgb(${c[0]},${c[1]},${c[2]})`, lum: (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) / 255 };
}

/* ---------------------------------------------------------------- charts */
function shareBars(container, rows, opts = {}) {
  const W = Math.max(container.clientWidth, 300);
  const longest = Math.max(...rows.map((r) => String(r.label ?? r.skill).length));
  const labelW = opts.labelW || Math.round(Math.min(W * 0.5, Math.max(90, longest * 6.9 + 18)));
  const valueW = 64, rowH = opts.rowH || 30, barH = opts.barH || 14;
  const plotW = W - labelW - valueW;
  const maxHi = Math.max(...rows.map((r) => (r.ci95 ? r.ci95[1] : r.share)), 0.01);
  const xmax = niceMax(maxHi);
  const sx = (v) => labelW + (v / xmax) * plotW;
  const H = rows.length * rowH + (opts.axis === false ? 4 : 26);
  const s = svg(container, W, H, opts.label || "bar chart");
  if (opts.axis !== false) {
    for (let i = 0; i <= 4; i++) {
      const v = (xmax * i) / 4, x = sx(v);
      el("line", { x1: x, x2: x, y1: 0, y2: rows.length * rowH, stroke: "var(--grid)" }, s);
      text(s, x, rows.length * rowH + 17, Math.round(v * 100) + "%", "tick", "middle");
    }
  }
  rows.forEach((r, i) => {
    const y = i * rowH + (rowH - barH) / 2, cy = y + barH / 2;
    const g = el("g", { class: "row", tabindex: opts.focusable === false ? null : 0, "data-tip": opts.tip ? opts.tip(r) : null }, s);
    el("rect", { class: "hit", x: 0, y: i * rowH, width: W, height: rowH, rx: 6 }, g);
    text(g, labelW - 10, cy + 4, r.label ?? r.skill, "lbl", "end");
    const w = Math.max((r.share / xmax) * plotW, 2);
    el("path", { d: barRight(labelW, y, w, barH), fill: opts.color || "var(--series)" }, g);
    let tipX = labelW + w;
    if (r.ci95) {
      const [lo, hi] = r.ci95, xl = sx(lo), xh = sx(hi);
      el("line", { x1: xl, x2: xh, y1: cy, y2: cy, stroke: "var(--ink)", "stroke-opacity": 0.5, "stroke-width": 1.5 }, g);
      el("line", { x1: xh, x2: xh, y1: cy - 4, y2: cy + 4, stroke: "var(--ink)", "stroke-opacity": 0.5, "stroke-width": 1.5 }, g);
      tipX = Math.max(tipX, xh);
    }
    text(g, tipX + 7, cy + 4, r.valueText ?? pct(r.share), "val");
  });
  el("line", { x1: labelW, x2: labelW, y1: 0, y2: rows.length * rowH, stroke: "var(--axis)" }, s);
}

function columns(container, items, label) {
  const W = Math.max(container.clientWidth, 280), H = 210, pad = 26, top = 18;
  const s = svg(container, W, H, label);
  const max = Math.max(...items.map((d) => d.value), 1);
  const bw = Math.min(46, ((W - 10) / items.length) * 0.62), step = (W - 10) / items.length;
  el("line", { x1: 0, x2: W, y1: H - pad, y2: H - pad, stroke: "var(--axis)" }, s);
  items.forEach((d, i) => {
    const h = ((H - pad - top) * d.value) / max, x = 5 + i * step + (step - bw) / 2;
    const g = el("g", { class: "row", tabindex: 0, "data-tip": d.tip }, s);
    el("rect", { class: "hit", x: 5 + i * step, y: 0, width: step, height: H - pad, rx: 6 }, g);
    if (h > 0) el("path", { d: barUp(x, H - pad, bw, Math.max(h, 2)), fill: "var(--series)" }, g);
    text(g, x + bw / 2, H - pad - h - 6, num(d.value), "val", "middle");
    text(g, x + bw / 2, H - 8, d.label, "tick", "middle");
  });
}

function diverging(container, rows, label) {
  const W = Math.max(container.clientWidth, 300), labelW = Math.min(150, W * 0.22), valueW = 70;
  const rowH = 30, barH = 14, plotW = W - labelW - valueW, half = plotW / 2, zero = labelW + half;
  const m = Math.max(1, Math.ceil(Math.max(...rows.map((r) => Math.abs(r.delta_pp)))));
  const H = rows.length * rowH + 26;
  const s = svg(container, W, H, label);
  [-1, -0.5, 0.5, 1].forEach((f) => {
    const x = zero + f * half;
    el("line", { x1: x, x2: x, y1: 0, y2: rows.length * rowH, stroke: "var(--grid)" }, s);
    text(s, x, rows.length * rowH + 17, (f * m > 0 ? "+" : "") + f * m + " pp", "tick", "middle");
  });
  rows.forEach((r, i) => {
    const y = i * rowH + (rowH - barH) / 2, cy = y + barH / 2, d = r.delta_pp;
    const w = Math.max((Math.abs(d) / m) * half, 2);
    const g = el("g", { class: "row", tabindex: 0, "data-tip":
      `<b>${esc(r.skill)}</b><br>${pct(r.first_share, 1)} → ${pct(r.last_share, 1)} (${d > 0 ? "+" : ""}${d.toFixed(2)} pp)` +
      (r.flow_significant ? "<br>significant in the new-posting flow test" : "<br><span class='t-sub'>descriptive (stock)</span>") }, s);
    el("rect", { class: "hit", x: 0, y: i * rowH, width: W, height: rowH, rx: 6 }, g);
    text(g, labelW - 10, cy + 4, r.skill, "lbl", "end");
    el("path", { d: d >= 0 ? barRight(zero, y, w, barH) : barLeft(zero, y, w, barH), fill: d >= 0 ? "var(--pos)" : "var(--neg)" }, g);
    text(g, d >= 0 ? zero + w + 6 : zero - w - 6, cy + 4, (d > 0 ? "+" : "") + d.toFixed(1) + (r.flow_significant ? " *" : ""), "val", d >= 0 ? "start" : "end");
  });
  el("line", { x1: zero, x2: zero, y1: 0, y2: rows.length * rowH, stroke: "var(--zero)", "stroke-width": 1.5 }, s);
}

function heatmap(container, cities, skills) {
  const W = Math.max(container.clientWidth, 520);
  const labelW = 110, headH = 74, cellH = 40;
  const cellW = Math.max(38, (W - labelW) / skills.length);
  const width = labelW + cellW * skills.length, H = headH + cellH * cities.length + 6;
  const s = svg(container, width, H, "city by skill heatmap");
  const lo = cssVar("--heat-lo"), hi = cssVar("--heat-hi");
  let max = 0.01;
  cities.forEach((c) => skills.forEach((k) => { const r = c.map[k]; if (r) max = Math.max(max, r.share); }));
  skills.forEach((k, j) => {
    const x = labelW + j * cellW + cellW / 2;
    text(s, x, headH - 10, k, "lbl", "start", { transform: `rotate(-38 ${x} ${headH - 10})` });
  });
  cities.forEach((c, i) => {
    const y = headH + i * cellH;
    text(s, labelW - 10, y + cellH / 2 + 4, c.label, "lbl", "end");
    text(s, labelW - 10, y + cellH / 2 + 17, "n=" + num(c.n), "tick", "end");
    skills.forEach((k, j) => {
      const r = c.map[k] || { share: 0, postings: 0, ci95: [0, 0] };
      const col = mix(lo, hi, Math.sqrt(r.share / max));
      const g = el("g", { tabindex: 0, "data-tip": `<b>${esc(k)} · ${esc(c.label)}</b><br>${num(r.postings)} of ${num(c.n)} postings = ${pct(r.share, 1)}<br><span class='t-sub'>95% CI ${pct(r.ci95[0], 1)}–${pct(r.ci95[1], 1)}</span>` }, s);
      el("rect", { x: labelW + j * cellW + 1, y: y + 1, width: cellW - 2, height: cellH - 2, rx: 6, fill: col.css }, g);
      text(g, labelW + j * cellW + cellW / 2, y + cellH / 2 + 4, Math.round(r.share * 100) + "", "val", "middle",
        { style: `fill:${col.lum < 0.5 ? "#fff" : "#0b0b0b"};font-weight:600` });
    });
  });
}

function sparkline(series, w = 90, h = 22) {
  const pts = series.map((v, i) => [i, v]).filter((p) => p[1] != null);
  if (pts.length < 2) return "";
  const xs = (i) => (i / (series.length - 1)) * (w - 4) + 2;
  const vals = pts.map((p) => p[1]), lo = Math.min(...vals), hi = Math.max(...vals);
  const ys = (v) => h - 3 - ((v - lo) / (hi - lo || 1)) * (h - 6);
  const d = pts.map((p, k) => (k ? "L" : "M") + xs(p[0]).toFixed(1) + "," + ys(p[1]).toFixed(1)).join("");
  return `<svg width="${w}" height="${h}" aria-hidden="true"><path d="${d}" fill="none" stroke="var(--series)" stroke-width="2" stroke-linecap="round"/></svg>`;
}

/* ---------------------------------------------------------------- sections */
function kpi(k, v, n) { return `<div class="kpi"><div class="k">${esc(k)}</div><div class="v" data-count="${typeof v === "number" ? v : ""}">${esc(typeof v === "number" ? num(v) : v)}</div><div class="n">${esc(n)}</div></div>`; }

function countUp(root) {
  if (reduced) return;
  root.querySelectorAll("[data-count]").forEach((node) => {
    const target = Number(node.dataset.count);
    if (!target) return;
    const t0 = performance.now();
    const step = (t) => {
      const p = Math.min(1, (t - t0) / 900);
      node.textContent = num(Math.round(target * (1 - Math.pow(1 - p, 3))));
      if (p < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  });
}

function historyText(h) {
  if (h.n_snapshots < 2) return `<span class="dot warning"></span><span><b>History: ${h.n_snapshots} real snapshot day</b> (${esc(h.first)}). Trends need at least 2 days and are never backfilled, so for now you are looking at a point-in-time snapshot. A scheduled daily run adds the next real day.</span>`;
  const miss = h.missing_dates.length ? `, ${h.missing_dates.length} day(s) missing` : "";
  return `<span class="dot good"></span><span><b>History: ${h.n_snapshots} real snapshot days</b>, ${esc(h.first)} → ${esc(h.last)} (span ${h.span_days} days${miss}).</span>`;
}

function renderHero() {
  const k = D.kpis;
  $("#asOf").textContent = `Data as of ${D.as_of} · ${k.snapshot_days} snapshot day${k.snapshot_days === 1 ? "" : "s"}`;
  $("#kpis").innerHTML = [
    kpi("Open India postings", k.postings, `from ${num(k.jobs_scanned)} jobs scanned worldwide`),
    kpi("Employers in panel", k.employers, `${k.employers_ok} fetched OK on the latest day`),
    kpi("Skills tracked", k.skills_tracked, "reviewed vocabulary, 9 families"),
    kpi("Postings with a skill", pct(k.coverage), "role sections only, no boilerplate"),
    kpi("Median days open", k.median_days_open, `${pct(k.share_over_90)} open over 90 days`),
  ].join("");
  countUp($("#kpis"));
  $("#historyBanner").innerHTML = `<div class="banner">${historyText(D.history)}</div>`;
}

const scopeById = (id) => D.scopes.find((s) => s.id === id);
function fillScopeSelect(sel, value) {
  const groups = { all: "Overall", role: "Roles", city: "Cities" };
  sel.innerHTML = Object.entries(groups).map(([kind, label]) => {
    const opts = D.scopes.filter((s) => s.kind === kind).map((s) => `<option value="${esc(s.id)}">${esc(s.label)} (${num(s.n)})</option>`).join("");
    return opts ? `<optgroup label="${label}">${opts}</optgroup>` : "";
  }).join("");
  sel.value = value;
}

function skillTip(r, n) {
  const m = D.skill_meta[r.skill];
  let s = `<b>${esc(r.skill)}</b> <span class="t-sub">· ${esc(D.categories[r.category] || r.category)}</span><br>${num(r.postings)} of ${num(n)} postings = <b>${pct(r.share, 1)}</b><br><span class="t-sub">95% CI ${pct(r.ci95[0], 1)}–${pct(r.ci95[1], 1)}</span>`;
  if (m && state.scope === "all") s += `<br>${m.boards} employers · top: ${esc(m.top_board)} (${pct(m.top_board_share)})<br>${num(m.in_requirements_or_title)} in requirements or title`;
  return s;
}

function renderSkills() {
  const sc = scopeById(state.scope);
  const present = [...new Set(sc.skills.map((r) => r.category))];
  const order = Object.keys(D.categories).filter((c) => present.includes(c));
  if (state.cat !== "all" && !present.includes(state.cat)) state.cat = "all";
  $("#catChips").innerHTML = [`<button type="button" class="chip ${state.cat === "all" ? "on" : ""}" data-cat="all">All families</button>`]
    .concat(order.map((c) => `<button type="button" class="chip ${state.cat === c ? "on" : ""}" data-cat="${c}">${esc(D.categories[c])}</button>`)).join("");
  const rows = sc.skills.filter((r) => state.cat === "all" || r.category === state.cat).slice(0, state.top);
  $("#skillsTitle").textContent = `${sc.label}`;
  $("#skillsSub").textContent = `${num(sc.n)} postings · ${sc.day} · showing ${rows.length} of ${sc.skills.filter((r) => state.cat === "all" || r.category === state.cat).length} skills`;
  if (!rows.length) { $("#skillsChart").innerHTML = `<p class="muted">No skills in this family for this scope.</p>`; return; }
  shareBars($("#skillsChart"), rows, { label: "skill share", tip: (r) => skillTip(r, sc.n) });
  $("#skillsTable").innerHTML = `<table class="data"><thead><tr><th>Skill</th><th>Family</th><th>Postings</th><th>Share</th><th>95% CI</th></tr></thead><tbody>` +
    rows.map((r) => `<tr><td>${esc(r.skill)}</td><td>${esc(D.categories[r.category] || r.category)}</td><td>${num(r.postings)}</td><td>${pct(r.share, 1)}</td><td>${pct(r.ci95[0], 1)}–${pct(r.ci95[1], 1)}</td></tr>`).join("") + `</tbody></table>`;
}

function renderRoles() {
  const all = scopeById("all");
  const allShare = Object.fromEntries(all.skills.map((r) => [r.skill, r.share]));
  const roles = D.scopes.filter((s) => s.kind === "role").sort((a, b) => b.n - a.n);
  const grid = $("#roleGrid");
  grid.innerHTML = roles.map((r, i) => {
    const distinct = r.skills.filter((x) => x.postings >= 3 && allShare[x.skill])
      .map((x) => ({ ...x, lift: x.share / allShare[x.skill] })).sort((a, b) => b.lift - a.lift)[0];
    return `<div class="card role-card"><div class="card-head"><h3>${esc(r.label)}</h3>
      <span>${r.n < 30 ? `<span class="badge warn" data-tip="Under 30 postings: shares are noisy. See the interval whiskers."><span class="dot warning"></span>small sample · n=${r.n}</span>` : `<span class="badge">n=${num(r.n)}</span>`}</span></div>
      <div class="chart" id="role-${i}"></div>
      ${distinct ? `<p class="note">Most distinctive: <b>${esc(distinct.skill)}</b>, ${distinct.lift.toFixed(1)}× its rate across all postings</p>` : ""}</div>`;
  }).join("");
  roles.forEach((r, i) => shareBars($(`#role-${i}`), r.skills.slice(0, 6), { rowH: 26, barH: 11, axis: false, label: r.label, tip: (x) => skillTip(x, r.n) }));
}

function renderCities() {
  const top = scopeById("all").skills.slice(0, 12).map((r) => r.skill);
  const cities = D.scopes.filter((s) => s.kind === "city").map((c) => ({ label: c.label, n: c.n, map: Object.fromEntries(c.skills.map((r) => [r.skill, r])) }));
  if (!cities.length) { $("#cityHeat").innerHTML = `<p class="muted">No city has enough postings yet.</p>`; return; }
  heatmap($("#cityHeat"), cities, top);
}

function nextRunText() {
  const now = new Date(), next = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate(), 3, 30));
  if (next <= now) next.setUTCDate(next.getUTCDate() + 1);
  return next.toLocaleString(undefined, { weekday: "short", hour: "2-digit", minute: "2-digit", timeZoneName: "short" });
}

function renderTrends() {
  const t = D.trends[state.trendScope];
  const body = $("#trendBody");
  const h = t.history, need = 14;
  if (t.status !== "ok") {
    const days = h.n_snapshots, span = h.span_days || 0;
    const dots = Array.from({ length: need }, (_, i) => `<span class="d ${i < days ? "on" : ""}" data-tip="${i < days ? "real snapshot day collected" : "future day: collected by the daily run"}"></span>`).join("");
    body.innerHTML = `<div class="builder">
      <div class="card"><div class="card-head"><h3>History builder</h3><span class="muted">${esc(t.scope)}</span></div>
        <div class="steps">
          <div class="step"><div class="row1"><span>Stock trend (risers / fallers)</span><span>${Math.min(days, 2)} / 2 days</span></div><div class="bar"><span style="width:${Math.min(100, (days / 2) * 100)}%"></span></div><span class="muted">Needs 2 real snapshot days.</span></div>
          <div class="step"><div class="row1"><span>Flow significance test</span><span>${span} / ${need} days</span></div><div class="bar"><span style="width:${Math.min(100, (span / need) * 100)}%"></span></div><span class="muted">Needs a 14-day window and at least 30 new postings in each half.</span></div>
        </div>
        <div class="timeline" aria-label="snapshot days">${dots}</div>
        <p class="note">Next scheduled snapshot: <b>${esc(nextRunText())}</b>.</p></div>
      <div class="card"><div class="card-head"><h3>Why nothing is drawn yet</h3></div>
        <p class="note" style="margin-top:0">${esc(t.reason || "")}</p>
        <p class="note">With one day of data, any "trend" chart would be invented. As real days accumulate, this panel switches to risers and fallers with sparklines. Each change is measured over employers fetched OK on every day, so one failed board cannot fake a drop.</p></div>
    </div>`;
    return;
  }
  const f = t.flow;
  const list = (rows) => rows.map((r) => `<div class="ex-row"><span>${esc(r.skill)} ${sparkline(r.series || [])}</span><span>${pct(r.first_share, 1)} → ${pct(r.last_share, 1)}</span></div>`).join("");
  body.innerHTML = `<div class="banner" style="margin-bottom:14px">${historyText(h)}</div>
    <div class="trend-grid">
      <div class="card chart-card"><div class="card-head"><h3>Top risers</h3><span class="muted">change in share, percentage points</span></div><div class="chart" id="risers"></div>${list(t.risers.slice(0, 5))}</div>
      <div class="card chart-card"><div class="card-head"><h3>Top fallers</h3><span class="muted">change in share, percentage points</span></div><div class="chart" id="fallers"></div>${list(t.fallers.slice(0, 5))}</div>
    </div>
    <p class="note">Compared over ${t.panel.boards_compared} employers fetched OK every day${Object.keys(t.panel.boards_excluded).length ? ` (${Object.keys(t.panel.boards_excluded).length} excluded)` : ""}.
      Flow test: <b>${esc(f.status)}</b> (new postings ${f.early} early / ${f.late} late)${f.reason ? ". " + esc(f.reason) : ""}. * = significant after Benjamini-Hochberg.</p>`;
  if (t.risers.length) diverging($("#risers"), t.risers, "top risers"); else $("#risers").innerHTML = `<p class="muted">No skill rose.</p>`;
  if (t.fallers.length) diverging($("#fallers"), t.fallers, "top fallers"); else $("#fallers").innerHTML = `<p class="muted">No skill fell.</p>`;
}

/* ---------------------------------------------------------------- classifier */
function renderExamples() {
  const box = $("#examples");
  if (!D.examples.length) { box.innerHTML = ""; return; }
  box.innerHTML = `<span class="muted" style="width:100%;font-size:12.5px">Real held-out postings, never seen in training:</span>` +
    D.examples.map((e, i) => `<button type="button" class="ex" data-ex="${i}"><span class="t">${esc(e.title)}</span><span class="m">${esc(e.board)} · rule label: ${esc(e.rule_label)}</span></button>`).join("");
}

function resultHtml(out) {
  const p = out.probabilities, bs = p.below_senior ?? 0, se = p.senior ?? 0;
  const ex = state.example;
  let ruleRow = "";
  if (ex) {
    const ok = ex.target === out.prediction;
    ruleRow = `<dt>Rule label</dt><dd>${esc(ex.rule_label)} (${esc(ex.target)}) · <span class="${ok ? "match" : "err"}">${ok ? "model agrees" : "model disagrees"}</span></dd>
      <dt>Posting</dt><dd><a href="${esc(ex.url)}" target="_blank" rel="noopener">${esc(ex.title)}</a> · ${esc(ex.board)}</dd>`;
  }
  return `<div class="verdict">Prediction<span class="big">${esc(out.prediction.replace("_", " "))}</span></div>
    <div class="split" role="img" aria-label="below senior ${pct(bs)}, senior ${pct(se)}">
      <div class="s1" style="flex:${Math.max(bs, 0.001)}">${bs >= 0.14 ? pct(bs) : ""}</div>
      <div class="s2" style="flex:${Math.max(se, 0.001)}">${se >= 0.14 ? pct(se) : ""}</div></div>
    <div class="legend"><span><i style="background:var(--neg)"></i>below senior ${pct(bs, 1)}</span><span><i style="background:var(--pos)"></i>senior ${pct(se, 1)}</span></div>
    <dl class="facts">${ruleRow}<dt>Model</dt><dd>${esc(out.model)}</dd><dt>Input used</dt><dd>${num(out.input_tokens_used)} tokens of the role sections (title + seniority words removed)</dd></dl>
    <p class="note">${esc(out.scope_note)}. The probabilities are not calibrated, so read them as a ranking. This is a demo, not a decision about a person.</p>`;
}

async function classify() {
  const btn = $("#classifyBtn"), out = $("#result"), txt = $("#jdText").value;
  if (txt.trim().length < 200) { out.innerHTML = `<p class="err">Paste a full job description (at least a few sentences).</p>`; return; }
  btn.disabled = true; btn.textContent = "Classifying…";
  out.innerHTML = `<div class="skeleton"></div>`;
  try {
    const r = await fetch("/api/classify", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: txt }) });
    const j = await r.json();
    out.innerHTML = r.ok ? resultHtml(j) : `<p class="err">${esc(j.detail || "Request failed")}</p>`;
  } catch (e) {
    out.innerHTML = `<p class="err">Network error: ${esc(e.message)}. On the free tier the first request after idle can take up to a minute.</p>`;
  } finally { btn.disabled = false; btn.textContent = "Classify seniority"; }
}

function renderModel() {
  const M = D.model;
  if (!M) { $("#modelChart").innerHTML = `<p class="muted">No model metrics committed.</p>`; return; }
  const rows = M.results.map((r) => ({ label: r.name, share: r.macro_f1, ci95: r.ci, valueText: r.macro_f1.toFixed(3) }));
  const tipFor = (r) => {
    const x = M.results.find((m) => m.name === r.label);
    return `<b>${esc(x.name)}</b><br>macro-F1 ${x.macro_f1.toFixed(3)} (95% CI ${x.ci[0].toFixed(2)}–${x.ci[1].toFixed(2)})<br>accuracy ${pct(x.accuracy, 1)}${x.note ? `<br><span class="t-sub">${esc(x.note)}</span>` : ""}`;
  };
  shareBars($("#modelChart"), rows, { label: "model comparison", tip: tipFor, rowH: 34 });
  const c = M.comparison, par = M.parity;
  $("#modelNote").innerHTML = (c ? `Paired exact McNemar test, fine-tuned vs TF-IDF baseline: p = ${c.mcnemar_exact_p} (${c.only_transformer_correct} postings only DistilBERT got right, ${c.only_baseline_correct} only the baseline). <b>${esc(c.reading.charAt(0).toUpperCase() + c.reading.slice(1))}</b>.` : "") +
    (par ? ` The served int8 model agrees with fp32 on ${pct(par.agreement_with_fp32, 1)} of test postings.` : "");
  const served = M.results.find((r) => r.key === "int8") || M.results[M.results.length - 1];
  const cm = served.confusion, labels = cm.labels, m = cm.rows_true_cols_pred;
  const total = m.flat().reduce((a, b) => a + b, 0) || 1;
  const box = $("#confusion");
  const W = Math.max(box.clientWidth, 260), lw = 104, top = 34, cw = (W - lw) / 2, ch = 64;
  const s = svg(box, W, top + ch * 2 + 4, "confusion matrix");
  labels.forEach((l, j) => text(s, lw + j * cw + cw / 2, top - 12, "pred: " + l.replace("_", " "), "tick", "middle"));
  labels.forEach((l, i) => {
    text(s, lw - 10, top + i * ch + ch / 2 + 4, "true: " + l.replace("_", " "), "lbl", "end");
    labels.forEach((_, j) => {
      const v = m[i][j], col = mix(cssVar("--heat-lo"), i === j ? cssVar("--heat-hi") : cssVar("--neg"), Math.sqrt(v / total));
      const g = el("g", { tabindex: 0, "data-tip": `${v} postings: true ${esc(l)}, predicted ${esc(labels[j])}` }, s);
      el("rect", { x: lw + j * cw + 2, y: top + i * ch + 2, width: cw - 4, height: ch - 4, rx: 8, fill: col.css }, g);
      text(g, lw + j * cw + cw / 2, top + i * ch + ch / 2 + 6, String(v), "val", "middle", { style: `font-size:18px;font-weight:700;fill:${col.lum < 0.5 ? "#fff" : "#0b0b0b"}` });
    });
  });
  const recall = m[0][0] / Math.max(1, m[0][0] + m[0][1]);
  $("#confusionNote").innerHTML = `It catches <b>${m[0][0]} of ${m[0][0] + m[0][1]}</b> below-senior postings (recall ${pct(recall)}) and is right on <b>${m[1][1]} of ${m[1][0] + m[1][1]}</b> senior ones. More real labelled postings, not a bigger model, are what will move this.`;
  const L = (D.quality.classifier_labels || {}).fine_class_counts || M.labels.fine;
  const need = (D.quality.classifier_labels || {}).min_per_class || M.labels.min_per_class;
  const order = ["intern", "junior", "mid", "senior"];
  const gate = $("#labelGate");
  const max = Math.max(need * 1.4, ...order.map((k) => L[k] || 0));
  const gw = Math.max(gate.clientWidth, 260), glw = 64, gvw = 104, rowH = 34, gtop = 20;
  const g = svg(gate, gw, order.length * rowH + gtop + 4, "labels per class vs minimum");
  const sx = (v) => glw + (v / max) * (gw - glw - gvw);
  order.forEach((k, i) => {
    const v = L[k] || 0, y = gtop + i * rowH + 10, ok = v >= need;
    const grp = el("g", { tabindex: 0, "data-tip": `<b>${k}</b>: ${v} real labelled postings<br>${ok ? "clears" : "below"} the minimum of ${need}` }, g);
    text(grp, glw - 10, y + 11, k, "lbl", "end");
    if (v > 0) el("path", { d: barRight(glw, y, Math.max(sx(v) - glw, 2), 14), fill: ok ? "var(--series)" : "var(--warning)" }, grp);
    text(grp, gw - 2, y + 11, ok ? `${num(v)} ✓` : `${num(v)} · needs ${need - v}`, "val", "end");
  });
  el("line", { x1: sx(need), x2: sx(need), y1: gtop - 2, y2: gtop + order.length * rowH, stroke: "var(--ink)", "stroke-dasharray": "4 3", "stroke-opacity": 0.6 }, g);
  text(g, sx(need), gtop - 7, `minimum ${need}`, "tick", "middle");
}

/* ---------------------------------------------------------------- quality */
const REASONS = {
  employer_self_mention: "Employer naming itself or its product (e.g. Databricks at Databricks)",
  ambiguous_followed_by_symbol: "“Go-live”, “Go-to-market” style uses",
  ambiguous_negative_pattern: "“Go To Market”, “Go beyond”",
  ambiguous_no_context: "“Go”/“R” with no programming context",
  negative_next_word: "“Excel at …” used as a verb",
};
function renderQuality() {
  const q = D.quality, last = q.days[q.days.length - 1] || {}, cs = q.coverage_scopes || {};
  $("#qTiles").innerHTML = [
    kpi("Employers fetched OK", `${last.boards_ok ?? "–"} / ${D.kpis.employers}`, `latest day ${last.day || "–"}`),
    kpi("India postings", last.in_region_postings, "after the region filter"),
    kpi("Skill coverage", pct(cs.all_in_region?.coverage), `requirements only ${pct(cs.requirements_only?.coverage)} · anywhere ${pct(cs.anywhere_incl_boilerplate?.coverage)}`),
    kpi("Sections recognised", pct(q.headings_share), "postings with section headings"),
    kpi("Descriptions present", pct(last.description_completeness), "full text, not snippets"),
  ].join("");
  $("#gaps").innerHTML = (q.gaps.length ? q.gaps.map((g) => ["warning", g]) : [["good", "No known gaps."]])
    .map(([s, g]) => `<li><span class="dot ${s}"></span><b>${s === "good" ? "OK" : "Gap"}</b><span>${esc(g)}</span></li>`).join("");
  $("#snapLog").innerHTML = `<table class="data"><thead><tr><th>Day</th><th>Employers OK</th><th>India postings</th><th>New</th><th>Skill coverage</th></tr></thead><tbody>` +
    q.days.slice(-10).reverse().map((d) => `<tr><td>${esc(d.day)}</td><td>${d.boards_ok}${d.boards_failed.length ? ` <span class="err">(${d.boards_failed.length} failed)</span>` : ""}</td><td>${num(d.in_region_postings)}</td><td>${num((d.stored || {}).new)}</td><td>${pct(d.skill_coverage)}</td></tr>`).join("") + `</tbody></table>`;
  const roles = Object.entries(q.coverage_by_role).filter(([, v]) => v.postings).map(([k, v]) => ({ label: k, share: v.coverage, valueText: `${v.with_skill}/${v.postings}` }));
  shareBars($("#covRole"), roles, { label: "coverage by role", tip: (r) => `<b>${esc(r.label)}</b><br>${r.valueText} postings with at least one skill (${pct(r.share)})`, rowH: 28 });
  columns($("#ageHist"), q.age_histogram.map((b) => ({ label: b.bucket.replace(" days", "d").replace("over a year", ">1y"), value: b.postings, tip: `<b>${esc(b.bucket)}</b><br>${num(b.postings)} open postings` })), "posting age");
  $("#exclusions").innerHTML = Object.entries(q.exclusions).sort((a, b) => b[1] - a[1])
    .map(([k, v]) => `<div class="ex-row"><span>${esc(REASONS[k] || k)}</span><span>${num(v)}</span></div>`).join("") || `<p class="muted">None.</p>`;
  $("#vocabGaps").innerHTML = q.vocab_gaps.length ? q.vocab_gaps.map((t) => `<span class="chip" data-tip="${t.postings} postings at ${t.boards} employers">${esc(t.term)} · ${t.postings}</span>`).join("") : `<span class="muted">Nothing notable outside the vocabulary.</span>`;
}

function renderMethod() {
  const k = D.kpis;
  const steps = [
    ["Ingest", "Public Greenhouse, Lever and Ashby job boards of 55 employers, fetched daily with raw bytes cached and hashed.", `${num(k.jobs_scanned)} jobs scanned`],
    ["Filter & store", "India filter from real location fields with evidence. Postings are versioned by content hash; sightings are kept per day.", `${num(k.postings)} India postings`],
    ["Extract skills", "spaCy PhraseMatcher with exact-spelling and context rules (Go vs Go-live). Section tags keep boilerplate out.", `${pct(k.coverage)} coverage`],
    ["Normalize", "MiniLM embeddings and DBSCAN at a measured threshold. Merges are auto-accepted only with spelling support, everything else is reviewed.", `${num(k.skills_tracked)} skills`],
    ["Trends", "Balanced-panel stock change, plus a flow test on new postings (Fisher exact, Benjamini-Hochberg). History windows are always shown.", `${k.snapshot_days} day${k.snapshot_days === 1 ? "" : "s"} so far`],
    ["Classify", "Rule-derived seniority labels, DistilBERT vs TF-IDF baseline, then exported to int8 ONNX to fit a 512 MB server.", D.model ? `macro-F1 ${D.model.results.at(-1).macro_f1.toFixed(2)}` : ""],
  ];
  $("#pipeline").innerHTML = steps.map(([t, d, s]) => `<li><b>${t}</b>${esc(d)}<span class="stat">${esc(s)}</span></li>`).join("");
  const L = D.links;
  $("#links").innerHTML = [["GitHub repository", L.repo], ["Model card", L.model_card], ["Model release", L.release], ["Reviewed skill map", L.skill_map], ["Daily snapshot workflow", L.workflow]]
    .map(([t, u]) => `<a class="btn ghost" href="${esc(u)}" target="_blank" rel="noopener">${esc(t)} ↗</a>`).join("");
  $("#builtAt").textContent = `Data as of ${D.as_of} · page built ${new Date(D.built_at).toLocaleString()}`;
  $("#repoLink").href = L.repo;
}

/* ---------------------------------------------------------------- wiring */
function renderCharts() {
  renderSkills(); renderRoles(); renderCities(); renderTrends(); renderModel(); renderQuality();
}

function wire() {
  fillScopeSelect($("#scopeSelect"), state.scope);
  fillScopeSelect($("#trendScope"), state.trendScope);
  $("#scopeSelect").addEventListener("change", (e) => { state.scope = e.target.value; renderSkills(); });
  $("#trendScope").addEventListener("change", (e) => { state.trendScope = e.target.value; renderTrends(); });
  $("#catChips").addEventListener("click", (e) => { const b = e.target.closest("[data-cat]"); if (b) { state.cat = b.dataset.cat; renderSkills(); } });
  document.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll(".seg button").forEach((x) => x.classList.toggle("on", x === b));
    state.top = Number(b.dataset.top); renderSkills();
  }));
  $("#examples").addEventListener("click", (e) => {
    const b = e.target.closest("[data-ex]"); if (!b) return;
    state.example = D.examples[Number(b.dataset.ex)];
    $("#jdText").value = state.example.description; updateCount(); classify();
  });
  const updateCount = () => { $("#charCount").textContent = `${num($("#jdText").value.length)} characters`; };
  $("#jdText").addEventListener("input", () => { state.example = null; updateCount(); });
  $("#classifyBtn").addEventListener("click", classify);
  $("#themeToggle").addEventListener("click", () => {
    const t = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", t);
    try { localStorage.setItem("xray-theme", t); } catch (e) { /* storage unavailable */ }
    renderCharts();
  });
  let rt; addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(renderCharts, 150); });
  const links = [...document.querySelectorAll(".nav a")];
  const io = new IntersectionObserver((entries) => entries.forEach((en) => {
    if (en.isIntersecting) links.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === "#" + en.target.id));
  }), { rootMargin: "-45% 0px -50% 0px" });
  document.querySelectorAll("main section[id]").forEach((s) => io.observe(s));
}

async function main() {
  ["#skillsChart", "#cityHeat", "#modelChart"].forEach((s) => { $(s).innerHTML = `<div class="skeleton"></div>`; });
  try {
    const r = await fetch("/api/dashboard");
    if (!r.ok) throw new Error("HTTP " + r.status);
    D = await r.json();
  } catch (e) {
    $("#asOf").textContent = "Could not load data: " + e.message;
    return;
  }
  renderHero(); renderExamples(); renderMethod(); wire(); renderCharts();
  const demo = new URLSearchParams(location.search).get("demo");
  if (demo != null && D.examples.length) {
    state.example = D.examples[Math.min(Number(demo) || 0, D.examples.length - 1)];
    $("#jdText").value = state.example.description;
    $("#charCount").textContent = `${num(state.example.description.length)} characters`;
    classify();
  }
  if (D.serving && !D.serving.available) $("#result").innerHTML = `<p class="err">Classifier unavailable: ${esc(D.serving.reason)}</p>`;
}
main();
