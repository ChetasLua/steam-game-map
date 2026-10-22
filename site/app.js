"use strict";
/* Steam Game Map: every Steam game with 10+ reviews, placed by tag + description
   embeddings (UMAP). Search games you like, combine several, and rerank the
   neighbourhood by review quality and obscurity. */

const $ = (s) => document.querySelector(s);
const PALETTE = [
  [1.00, 0.45, 0.80], // Visual Novel
  [0.95, 0.25, 0.30], // Horror
  [1.00, 0.55, 0.20], // Shooter
  [0.35, 0.55, 1.00], // Strategy
  [0.66, 0.40, 1.00], // RPG
  [0.30, 0.90, 0.50], // Simulation
  [0.20, 0.85, 1.00], // Puzzle
  [1.00, 0.85, 0.25], // Platformer
  [0.65, 0.95, 0.20], // Racing & Sports
  [1.00, 0.35, 0.45], // Action
  [0.45, 0.70, 1.00], // Adventure
  [1.00, 0.68, 0.85], // Casual
  [0.55, 0.60, 0.78], // Other
];
const css = (c, a = 1) => `rgba(${c.map((v) => Math.round(v * 255)).join(",")},${a})`;
const header = (id) => `https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/${id}/header.jpg`;
const fmtN = (n) => (n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : n >= 1e3 ? (n / 1e3).toFixed(n >= 1e4 ? 0 : 1) + "k" : String(n));
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pctOf = (v) => Math.max(0, Math.min(99, Math.round(v * 90)));
const short = (s) => (s.length > 20 ? s.slice(0, 18).trim() + "…" : s);
const capsule = (id) => `https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/${id}/capsule_231x87.jpg`;
// Steam's own review summary labels and thresholds
function steamRating(pos, n) {
  const p = n ? (100 * pos) / n : 0;
  if (p >= 95 && n >= 500) return ["Overwhelmingly Positive", "pos"];
  if (p >= 80 && n >= 50) return ["Very Positive", "pos"];
  if (p >= 80) return ["Positive", "pos"];
  if (p >= 70) return ["Mostly Positive", "pos"];
  if (p >= 40) return ["Mixed", "mixed"];
  if (p >= 20) return ["Mostly Negative", "neg"];
  return [n >= 500 ? "Overwhelmingly Negative" : n >= 50 ? "Very Negative" : "Negative", "neg"];
}
const priceOf = (i) => (PRICE[i] === 0 ? "Free" : `$${(PRICE[i] / 100).toFixed(2)}`);
const STARTS = [413150, 367520, 646570, 753640, 294100, 1145360];
const norm = (s) => s.toLowerCase().normalize("NFKD").replace(/[̀-ͯ]/g, "").replace(/[^a-z0-9]+/g, " ").trim();

let M, N, DIM, XY, POS, NEG, YEAR, PRICE, TT, FAM, VEC, REV, WIL, POP, NAMES, BYAPP;
const state = { picks: [], mode: "all", quality: 45, obscure: 0, minrev: 50, maxprice: 99999, y0: 1997, y1: 2026 };
let results = [];      // [{i, sim, score, near}]
let simField = null;   // Float32Array(N) similarity of every game to the query
let perPick = [];      // one normalized similarity field per liked game
let famMask = new Float32Array(13).fill(1);
let hover = -1, hot = -1;

/* ---------------- data ---------------- */
async function load() {
  const msg = $("#loadmsg");
  const [meta, buf] = await Promise.all([
    fetch("data/meta.json").then((r) => r.json()),
    fetch("data/games.bin").then((r) => r.arrayBuffer()),
  ]);
  M = meta;
  const dv = new DataView(buf);
  N = dv.getUint32(4, true); DIM = dv.getUint32(8, true);
  let o = 12;
  const take = (T, len, bytes) => { const a = new T(buf, o, len); o += Math.ceil((len * bytes) / 4) * 4; return a; };
  XY = take(Float32Array, N * 2, 4);
  POS = take(Uint32Array, N, 4);
  NEG = take(Uint32Array, N, 4);
  YEAR = take(Uint16Array, N, 2);
  PRICE = take(Uint16Array, N, 2);
  TT = take(Uint16Array, N * 6, 2);
  FAM = take(Uint8Array, N, 1);
  const Q = take(Int8Array, N * DIM, 1);

  VEC = new Float32Array(N * DIM);
  REV = new Uint32Array(N); WIL = new Float32Array(N); POP = new Float32Array(N);
  let maxLog = 0;
  for (let i = 0; i < N; i++) {
    let ss = 0;
    for (let d = 0; d < DIM; d++) { const v = Q[i * DIM + d]; ss += v * v; }
    const inv = 1 / Math.sqrt(ss || 1);
    for (let d = 0; d < DIM; d++) VEC[i * DIM + d] = Q[i * DIM + d] * inv;
    const n = POS[i] + NEG[i];
    REV[i] = n;
    WIL[i] = wilson(POS[i], n);
    maxLog = Math.max(maxLog, Math.log10(n));
  }
  for (let i = 0; i < N; i++) POP[i] = (Math.log10(REV[i]) - 1) / (maxLog - 1);
  NAMES = M.name.map(norm);
  BYAPP = new Map(M.appid.map((a, i) => [a, i]));
  $("#count").textContent = N.toLocaleString();
  $("#src").innerHTML = `Every Steam game with 10+ reviews, placed by its user tags and description. Data from <a href="https://huggingface.co/datasets/FronkonGames/steam-games-dataset" target="_blank" rel="noopener">FronkonGames</a> (MIT) · idea by <a href="https://x.com/SyntheticBeef/status/2112904858718367750" target="_blank" rel="noopener">@SyntheticBeef</a>`;
}

function wilson(pos, n, z = 1.96) {
  if (!n) return 0;
  const p = pos / n;
  return (p + (z * z) / (2 * n) - z * Math.sqrt((p * (1 - p) + (z * z) / (4 * n)) / n)) / (1 + (z * z) / n);
}

/* ---------------- recommendations ---------------- */
function dot(i, q) {
  let s = 0; const b = i * DIM;
  for (let d = 0; d < DIM; d++) s += VEC[b + d] * q[d];
  return s;
}
function vecOf(i) { return VEC.subarray(i * DIM, i * DIM + DIM); }

function recompute() {
  const likes = state.picks.filter((p) => !p.avoid).map((p) => p.i);
  const avoids = state.picks.filter((p) => p.avoid).map((p) => p.i);
  if (!likes.length) { results = []; simField = null; return; }

  const picked = new Set(state.picks.map((p) => p.i));
  // Per-pick similarity, normalized by that pick's own 20th-nearest neighbour, so
  // 1.0 means "as close as this game's usual close neighbours" whatever its niche.
  const field = (i0) => {
    const v = vecOf(i0), S = new Float32Array(N);
    for (let i = 0; i < N; i++) S[i] = dot(i, v);
    const top = Array.from(S).sort((a, b) => b - a);
    const ref = Math.max(0.2, top[21]);
    for (let i = 0; i < N; i++) S[i] /= ref;
    return S;
  };
  const L = likes.map(field), A = avoids.map(field);
  const T = 12; // softmin temperature for "like all"

  simField = new Float32Array(N);
  const near = new Uint8Array(N);
  for (let i = 0; i < N; i++) {
    let sim, best = -9, acc = 0;
    for (let k = 0; k < L.length; k++) {
      const s = L[k][i];
      if (s > best) { best = s; near[i] = k; }
      acc += Math.exp(-T * s);
    }
    sim = state.mode === "any" || L.length === 1 ? best : -Math.log(acc / L.length) / T;
    if (A.length) {
      let pen = -9;
      for (const a of A) pen = Math.max(pen, a[i]);
      sim -= 0.9 * Math.max(0, pen - 0.45);
    }
    simField[i] = sim;
  }
  perPick = L;

  // Candidate pool = the semantic neighbourhood; the sliders only rerank inside it,
  // so "hidden gems" stays on-topic instead of surfacing random obscure games.
  const ok = [];
  const { minrev, maxprice, y0, y1 } = state;
  for (let i = 0; i < N; i++) {
    if (picked.has(i) || REV[i] < minrev || PRICE[i] > maxprice) continue;
    if ((y0 > 1997 || y1 < 2026) && (YEAR[i] < y0 || YEAR[i] > y1)) continue;
    ok.push(i);
  }
  ok.sort((a, b) => simField[b] - simField[a]);
  const pool = ok.slice(0, 500);
  const qw = state.quality / 100, ow = state.obscure / 100;
  const out = pool.map((i) => {
    const s = simField[i];
    const score = s + qw * (WIL[i] - 0.78) * 0.8 + ow * (0.42 - POP[i]) * 0.6;
    return { i, sim: s, score, near: likes[near[i]], each: L.map((S) => S[i]) };
  });
  out.sort((a, b) => b.score - a.score);
  results = out.slice(0, 60);
}

function sharedTags(i) {
  const mine = new Map();
  for (const p of state.picks) if (!p.avoid) for (let k = 0; k < 6; k++) { const t = TT[p.i * 6 + k]; if (t !== 0xffff) mine.set(t, (mine.get(t) || 0) + (6 - k)); }
  const sh = [];
  for (let k = 0; k < 6; k++) { const t = TT[i * 6 + k]; if (t !== 0xffff && mine.has(t)) sh.push(M.tags[t]); }
  return sh.slice(0, 3);
}
function topTags(i, n = 4) {
  const a = [];
  for (let k = 0; k < 6 && a.length < n; k++) { const t = TT[i * 6 + k]; if (t !== 0xffff) a.push(M.tags[t]); }
  return a;
}

/* ---------------- UI ---------------- */
function renderChips() {
  $("#chips").innerHTML = state.picks.map((p, k) =>
    `<span class="chip ${p.avoid ? "avoid" : ""}" data-k="${k}">
       <img src="${capsule(M.appid[p.i])}" alt="" onerror="this.style.visibility='hidden'">
       <span class="lbl" title="${p.avoid ? "Avoiding. Tap to like it again" : "Tap to avoid this game"}">${esc(M.name[p.i])}</span>
       <button class="x" aria-label="Remove ${esc(M.name[p.i])}">×</button></span>`).join("");
  const nLikes = state.picks.filter((p) => !p.avoid).length;
  $("#controls .seg").classList.toggle("single", nLikes < 2);
}

function renderList() {
  const list = $("#list");
  $("#empty").hidden = state.picks.length > 0;
  $("#rhead").hidden = !state.picks.length;
  if (!results.length) {
    list.innerHTML = state.picks.length ? `<li class="none">No games pass these filters. Loosen the year, price or review limits.</li>` : "";
    $("#rcount").textContent = "";
    return;
  }
  $("#rtitle").textContent = state.obscure > 35 ? "Hidden gems" : state.obscure < -35 ? "Big names" : "Recommended";
  $("#rcount").textContent = `${results.length} of ${N.toLocaleString()}`;
  const likes = state.picks.filter((p) => !p.avoid);
  const multi = likes.length > 1;
  list.innerHTML = results.map((r, k) => {
    const i = r.i, pct = REV[i] ? Math.round((100 * POS[i]) / REV[i]) : 0;
    const [rl, rc] = steamRating(POS[i], REV[i]);
    const gem = POP[i] < 0.28 && WIL[i] > 0.82;
    const sh = sharedTags(i);
    const why = multi
      ? `<div class="pp">${r.each.map((v, j) => `<span><em>${esc(short(M.name[likes[j].i]))}</em><s><i style="width:${pctOf(v)}%"></i></s><b>${pctOf(v)}</b></span>`).join("")}</div>`
      : `<div class="why">${sh.length ? "Shares " + sh.map(esc).join(", ") : esc(topTags(i, 3).join(", "))}</div>`;
    return `<li class="card" data-i="${i}" tabindex="0">
      <div class="cover"><img loading="lazy" src="${header(M.appid[i])}" alt="" onerror="this.style.visibility='hidden'"><span class="rank">${k + 1}</span></div>
      <div class="body">
        <div class="t">${esc(M.name[i])}</div>
        <div class="match"><span class="meter"><i style="width:${pctOf(r.sim)}%"></i></span><b>${pctOf(r.sim)}% match</b></div>
        <div class="meta"><span class="rating ${rc}" title="${pct}% of ${REV[i].toLocaleString()} reviews positive">${rl}</span><span class="num">${fmtN(REV[i])} · ${YEAR[i] || "—"} · ${priceOf(i)}</span>${gem ? '<span class="gem">◆ Hidden gem</span>' : ""}</div>
        ${why}
      </div>
      <div class="acts"><button data-add="${i}">+ Add</button><a href="https://store.steampowered.com/app/${M.appid[i]}" target="_blank" rel="noopener">Steam ↗</a></div>
    </li>`;
  }).join("");
}

function markHot() { document.querySelectorAll("#list .card").forEach((c) => c.classList.toggle("hot", +c.dataset.i === hot)); }

function renderStarts() {
  $("#starts").innerHTML = STARTS.map((a) => BYAPP.get(a)).filter((i) => i != null).map((i) =>
    `<button class="start" data-try="${M.appid[i]}"><img src="${header(M.appid[i])}" alt="" onerror="this.style.visibility='hidden'"><span>${esc(M.name[i])}</span></button>`).join("");
}

function renderFilterSummary() {
  const bits = [`${fmtN(state.minrev)}+ reviews`];
  if (state.maxprice < 99999) bits.push(state.maxprice === 0 ? "free" : `under $${state.maxprice / 100}`);
  if (state.y0 > 1997 || state.y1 < 2026) bits.push(`${state.y0}–${state.y1}`);
  $("#fsum").textContent = bits.join(" · ");
}

function update({ fly = true } = {}) {
  recompute();
  renderChips();
  renderList();
  renderFilterSummary();
  pushHighlights();
  syncHash();
  if (fly) flyToQuery();
  dirty();
}

function addPick(i, avoid = false) {
  if (state.picks.some((p) => p.i === i)) return;
  state.picks.push({ i, avoid });
  update();
}

/* search + autocomplete */
let sugg = [], act = 0;
function suggest(q) {
  const nq = norm(q);
  const box = $("#suggest");
  if (!nq) { box.classList.remove("open"); sugg = []; return; }
  const words = nq.split(" ");
  const hits = [];
  for (let i = 0; i < N; i++) {
    const nm = NAMES[i];
    let rank;
    if (nm === nq) rank = 0;
    else if (nm.startsWith(nq)) rank = 1;
    else if (nm.includes(" " + nq)) rank = 2;
    else if (words.every((w) => (" " + nm).includes(" " + w))) rank = 3;
    else continue;
    hits.push([rank, -REV[i], i]);
  }
  hits.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  sugg = hits.slice(0, 8).map((h) => h[2]);
  act = 0;
  box.innerHTML = sugg.length ? sugg.map((i, k) =>
    `<li role="option" data-i="${i}" class="${k === act ? "act" : ""}"><img loading="lazy" src="${header(M.appid[i])}" alt="" onerror="this.style.visibility='hidden'"><span class="nm">${esc(M.name[i])}</span><span class="mt">${YEAR[i] || ""}<br>${fmtN(REV[i])} reviews</span></li>`).join("")
    : `<li class="empty-row">No game with 10+ reviews matches “${esc(q)}”</li>`;
  box.classList.add("open");
}
function choose(i) {
  $("#q").value = ""; $("#suggest").classList.remove("open"); sugg = [];
  addPick(i);
  $("#q").focus();
}

function bindUI() {
  const q = $("#q");
  q.addEventListener("input", () => suggest(q.value));
  q.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      if (!sugg.length) return;
      act = (act + (e.key === "ArrowDown" ? 1 : -1) + sugg.length) % sugg.length;
      document.querySelectorAll("#suggest li").forEach((li, k) => li.classList.toggle("act", k === act));
    } else if (e.key === "Enter" && sugg.length) { e.preventDefault(); choose(sugg[act]); }
    else if (e.key === "Escape") { $("#suggest").classList.remove("open"); }
    else if (e.key === "Backspace" && !q.value && state.picks.length) { state.picks.pop(); update(); }
  });
  $("#suggest").addEventListener("mousedown", (e) => { const li = e.target.closest("li[data-i]"); if (li) { e.preventDefault(); choose(+li.dataset.i); } });
  q.addEventListener("blur", () => setTimeout(() => $("#suggest").classList.remove("open"), 120));
  window.addEventListener("keydown", (e) => { if (e.key === "/" && document.activeElement !== q) { e.preventDefault(); q.focus(); } });

  $("#chips").addEventListener("click", (e) => {
    const chip = e.target.closest(".chip"); if (!chip) return;
    const k = +chip.dataset.k;
    if (e.target.closest(".x")) state.picks.splice(k, 1);
    else state.picks[k].avoid = !state.picks[k].avoid;
    update();
  });
  document.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => {
    state.mode = b.dataset.mode;
    document.querySelectorAll(".seg button").forEach((x) => { x.classList.toggle("on", x === b); x.setAttribute("aria-checked", x === b); });
    update();
  }));
  const sl = (id, key, fmt) => {
    const el = $("#" + id), out = $("#o-" + id);
    const show = () => {
      out.textContent = fmt(+el.value);
      el.style.setProperty("--fill", ((+el.value - +el.min) / (+el.max - +el.min)) * 100 + "%");
    };
    el.addEventListener("input", () => { state[key] = +el.value; show(); update({ fly: false }); });
    el._sync = () => { el.value = state[key]; show(); };
    show();
  };
  sl("quality", "quality", (v) => (v === 0 ? "off" : v < 35 ? "light" : v < 70 ? "balanced" : "strict"));
  sl("obscure", "obscure", (v) => (Math.abs(v) < 8 ? "balanced" : v > 0 ? (v > 60 ? "deep cuts" : "lean hidden") : v < -60 ? "mainstream" : "lean popular"));
  for (const id of ["minrev", "maxprice", "y0", "y1"]) $("#" + id).addEventListener("change", (e) => { state[id] = +e.target.value; update({ fly: false }); });

  $("#list").addEventListener("click", (e) => {
    const add = e.target.closest("[data-add]");
    if (add) { addPick(+add.dataset.add); return; }
    if (e.target.closest("a")) return;
    const li = e.target.closest("li[data-i]"); if (!li) return;
    const i = +li.dataset.i;
    hot = i; flyTo(XY[2 * i], XY[2 * i + 1], Math.max(view.s, fitScale() * 6)); dirty();
  });
  $("#list").addEventListener("mouseover", (e) => { const li = e.target.closest("li[data-i]"); const i = li ? +li.dataset.i : -1; if (i !== hot) { hot = i; markHot(); dirty(); } });
  $("#list").addEventListener("mouseleave", () => { hot = -1; markHot(); dirty(); });
  $("#empty").addEventListener("click", (e) => {
    const b = e.target.closest("[data-try]"); if (!b) return;
    for (const a of b.dataset.try.split(",")) { const i = BYAPP.get(+a); if (i != null && !state.picks.some((p) => p.i === i)) state.picks.push({ i, avoid: false }); }
    update();
  });

  $("#legend").innerHTML = M.families.map((f, k) => `<button data-k="${k}" title="Show or hide ${esc(f)} games (alt-click to solo)"><i style="background:${css(PALETTE[k])};color:${css(PALETTE[k])}"></i>${esc(f)}</button>`).join("");
  $("#legend").addEventListener("click", (e) => {
    const s = e.target.closest("[data-k]"); if (!s) return;
    const k = +s.dataset.k;
    const solo = e.altKey || e.metaKey;
    if (solo) famMask.fill(0.04), (famMask[k] = 1);
    else famMask[k] = famMask[k] > 0.5 ? 0.04 : 1;
    document.querySelectorAll("#legend [data-k]").forEach((x) => x.classList.toggle("off", famMask[+x.dataset.k] < 0.5));
    dirty();
  });
  $("#zin").onclick = () => zoomAt(mapCenter()[0], mapCenter()[1], 1.6);
  $("#zout").onclick = () => zoomAt(mapCenter()[0], mapCenter()[1], 1 / 1.6);
  $("#zfit").onclick = () => { fitS = fitScale() || 1; flyTo(fitC[0], fitC[1], fitS); };
}

/* hash state: #like=413150,646570&avoid=730&mode=all&q=45&o=30 */
function syncHash() {
  const p = new URLSearchParams();
  const likes = state.picks.filter((x) => !x.avoid).map((x) => M.appid[x.i]);
  const avoid = state.picks.filter((x) => x.avoid).map((x) => M.appid[x.i]);
  if (likes.length) p.set("like", likes.join(","));
  if (avoid.length) p.set("avoid", avoid.join(","));
  if (state.mode !== "all") p.set("mode", state.mode);
  if (state.quality !== 45) p.set("q", state.quality);
  if (state.obscure !== 0) p.set("o", state.obscure);
  const h = p.toString();
  history.replaceState(null, "", h ? "#" + h : location.pathname + location.search);
}
function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  const add = (key, avoid) => (p.get(key) || "").split(",").filter(Boolean).forEach((a) => { const i = BYAPP.get(+a); if (i != null) state.picks.push({ i, avoid }); });
  add("like", false); add("avoid", true);
  if (["all", "any"].includes(p.get("mode"))) state.mode = p.get("mode");
  if (p.has("q")) state.quality = Math.max(0, Math.min(100, +p.get("q") || 0));
  if (p.has("o")) state.obscure = Math.max(-100, Math.min(100, +p.get("o") || 0));
  $("#quality")._sync(); $("#obscure")._sync();
  document.querySelectorAll(".seg button").forEach((x) => { x.classList.toggle("on", x.dataset.mode === state.mode); x.setAttribute("aria-checked", x.dataset.mode === state.mode); });
}

/* ---------------- map: WebGL points + 2D overlay ---------------- */
const glc = $("#gl"), ovc = $("#overlay");
const gl = glc.getContext("webgl2", { antialias: false, premultipliedAlpha: false });
const ctx = ovc.getContext("2d");
let W = 0, H = 0, DPR = 1, prog, hiBuf, litBuf, litCount = 0, uni = {};
const view = { cx: 0, cy: 0, s: 100 };
let fitC = [0, 0];

function panelW() { return innerWidth <= 760 ? 0 : $("#panel").offsetWidth + 28; }
function mapH() { return innerWidth <= 760 ? innerHeight * 0.46 : innerHeight; }
function mapCenter() { return [(innerWidth - panelW()) / 2, mapH() / 2]; }
function toScreen(x, y) { const [ox, oy] = mapCenter(); return [ox + (x - view.cx) * view.s, oy - (y - view.cy) * view.s]; }
function toData(px, py) { const [ox, oy] = mapCenter(); return [view.cx + (px - ox) / view.s, view.cy - (py - oy) / view.s]; }
function fitScale() {
  // frame the 0.5–99.5 percentile box so a few outlier games don't shrink the map
  if (!fitScale.box) {
    const xs = [], ys = [];
    for (let i = 0; i < N; i += 3) { xs.push(XY[2 * i]); ys.push(XY[2 * i + 1]); }
    xs.sort((a, b) => a - b); ys.sort((a, b) => a - b);
    const q = (a, p) => a[Math.floor(p * (a.length - 1))];
    fitScale.box = [q(xs, 0.005), q(xs, 0.995), q(ys, 0.005), q(ys, 0.995)];
  }
  const [x0, x1, y0, y1] = fitScale.box;
  fitC = [(x0 + x1) / 2, (y0 + y1) / 2];
  return 0.9 * Math.min((innerWidth - panelW()) / (x1 - x0), (mapH() - 60) / (y1 - y0));
}

function shader(type, src) {
  const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}
function initGL() {
  if (!gl) throw new Error("WebGL2 is not available in this browser.");
  prog = gl.createProgram();
  gl.attachShader(prog, shader(gl.VERTEX_SHADER, `#version 300 es
    in vec2 a_pos; in float a_fam; in float a_size; in float a_hi;
    uniform vec2 u_c; uniform float u_s; uniform vec2 u_off; uniform vec2 u_vp; uniform float u_dpr;
    uniform float u_active; uniform vec3 u_pal[13]; uniform float u_mask[13]; uniform float u_fit; uniform float u_halo; uniform float u_gain; uniform float u_skipLit;
    out vec4 v_col; out float v_halo;
    void main() {
      vec2 px = u_off + vec2(a_pos.x - u_c.x, -(a_pos.y - u_c.y)) * u_s;
      gl_Position = vec4(px.x / u_vp.x * 2.0 - 1.0, 1.0 - px.y / u_vp.y * 2.0, 0.0, 1.0);
      int f = int(a_fam);
      vec3 col = u_pal[f];
      float zoom = clamp(pow(u_s / u_fit, 0.6), 0.7, 6.0);
      float size = a_size * zoom;
      float alpha = u_halo > 0.5 ? 0.55 : 0.8;
      if (u_active > 0.5) {
        if (a_hi >= 10.0) {
          col = a_hi > 10.5 ? vec3(1.0, 0.42, 0.54) : vec3(1.0, 0.71, 0.28);
          size = max(size * 2.2, 11.0); alpha = 1.0;
        } else if (a_hi >= 2.0) {
          col = vec3(0.37, 0.94, 0.82);
          size = max(size * 1.6, a_hi < 7.0 ? 10.0 : 7.5); alpha = 1.0;
        } else {
          alpha = 0.09 + 0.55 * a_hi * a_hi;
        }
      }
      alpha *= u_mask[f];
      if (a_hi < 2.0) alpha *= u_gain;
      if (u_halo > 0.5) {
        // wide, faint second pass that turns dense clusters into coloured nebulae
        bool lit = u_active > 0.5 && a_hi >= 2.0;
        size = lit ? size * 3.2 : min(size * 4.5, 46.0);
        alpha *= lit ? 0.32 : 0.1;
      }
      gl_PointSize = size * u_dpr;
      v_col = vec4(col, alpha);
      v_halo = u_halo;
      if (u_skipLit > 0.5 && u_active > 0.5 && a_hi >= 2.0) gl_Position = vec4(2.0, 2.0, 0.0, 1.0);
    }`));
  gl.attachShader(prog, shader(gl.FRAGMENT_SHADER, `#version 300 es
    precision mediump float;
    in vec4 v_col; in float v_halo; out vec4 o;
    void main() {
      float d = length(gl_PointCoord - 0.5) * 2.0;
      if (d > 1.0) discard;
      float core = v_halo > 0.5 ? exp(-d * d * 2.6) * (1.0 - d) : smoothstep(1.0, 0.35, d);
      vec3 rgb = v_halo > 0.5 ? v_col.rgb : mix(v_col.rgb, vec3(1.0), 0.18 * exp(-d * d * 20.0));
      o = vec4(rgb, v_col.a * core);
    }`));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
  gl.useProgram(prog);

  const attr = (name, data, size) => {
    const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, name); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
    return b;
  };
  attr("a_pos", XY, 2);
  attr("a_fam", Float32Array.from(FAM), 1);
  attr("a_size", Float32Array.from(POP, (p) => 3.6 + 7.0 * p * p * p), 1);
  hiBuf = attr("a_hi", new Float32Array(N), 1);
  litBuf = gl.createBuffer();
  for (const u of ["u_c", "u_s", "u_off", "u_vp", "u_dpr", "u_active", "u_pal", "u_mask", "u_fit", "u_halo", "u_gain", "u_skipLit"]) uni[u] = gl.getUniformLocation(prog, u);
  gl.uniform3fv(uni.u_pal, PALETTE.flat());
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE); // additive glow
}

function pushHighlights() {
  const hi = new Float32Array(N);
  if (simField) {
    for (let i = 0; i < N; i++) hi[i] = Math.max(0, Math.min(1, (simField[i] - 0.25) / 0.75));
    results.forEach((r, k) => (hi[r.i] = 2 + k));
    for (const p of state.picks) hi[p.i] = p.avoid ? 11 : 10;
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, hiBuf);
  gl.bufferData(gl.ARRAY_BUFFER, hi, gl.DYNAMIC_DRAW);
  // picks and results get their own last draw so dense neighbours never paint over them
  const lit = simField ? [...results.map((r) => r.i).reverse(), ...state.picks.map((p) => p.i)] : [];
  litCount = lit.length;
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, litBuf);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, Uint32Array.from(lit), gl.DYNAMIC_DRAW);
}

let fitS = 100;
function resize() {
  DPR = Math.min(devicePixelRatio || 1, 2);
  W = innerWidth; H = innerHeight;
  for (const c of [glc, ovc]) { c.width = Math.round(W * DPR); c.height = Math.round(H * DPR); }
  const prevFit = fitS;
  fitS = Math.max(1, fitScale() || 1); // a hidden/zero-width first layout gives <= 0
  // keep the whole map framed until the user has panned/zoomed (also covers a 0-width first layout)
  if (!userMoved && !fly) { view.s = fitS; [view.cx, view.cy] = fitC; }
  else if (!(view.s > 0)) view.s = fitS;
  else view.s *= fitS / (prevFit || fitS);
  dirty();
}
let userMoved = false;

let raf = 0;
function dirty() { if (!raf) raf = requestAnimationFrame(draw); }
function draw() {
  raf = 0;
  stepFly();
  gl.viewport(0, 0, glc.width, glc.height);
  gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
  const [ox, oy] = mapCenter();
  gl.uniform2f(uni.u_c, view.cx, view.cy);
  gl.uniform1f(uni.u_s, view.s);
  gl.uniform2f(uni.u_off, ox, oy);
  gl.uniform2f(uni.u_vp, W, H);
  gl.uniform1f(uni.u_dpr, DPR);
  gl.uniform1f(uni.u_fit, fitS);
  gl.uniform1f(uni.u_active, simField ? 1 : 0);
  gl.uniform1fv(uni.u_mask, famMask);
  // Additive glow saturates to white where dots pile up, so dim each dot by how many
  // land per pixel at the current zoom; colours then survive on small and large screens.
  const dens = N / Math.max(1, (W - panelW()) * mapH()) / Math.pow(view.s / fitS, 2);
  gl.uniform1f(uni.u_gain, Math.max(0.38, Math.min(1, 0.05 / dens)));
  gl.uniform1f(uni.u_halo, 1);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE);                 // glow adds up into nebulae
  gl.drawArrays(gl.POINTS, 0, N);
  gl.uniform1f(uni.u_halo, 0);
  gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA); // dots keep their own hue
  gl.uniform1f(uni.u_skipLit, 1);
  gl.drawArrays(gl.POINTS, 0, N);
  gl.uniform1f(uni.u_skipLit, 0);
  if (litCount) { gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, litBuf); gl.drawElements(gl.POINTS, litCount, gl.UNSIGNED_INT, 0); }
  drawOverlay();
  if (fly) dirty();
}

/* overlay: region names, links from picks to recs, markers, labels, hover ring */
const AMBER = "255,178,74", AQUA = "94,240,208", ROSE = "255,107,138";
function pill(x, y, w, h) {
  const r = h / 2;
  ctx.beginPath();
  ctx.moveTo(x + r, y); ctx.lineTo(x + w - r, y); ctx.arc(x + w - r, y + r, r, -Math.PI / 2, Math.PI / 2);
  ctx.lineTo(x + r, y + h); ctx.arc(x + r, y + r, r, Math.PI / 2, -Math.PI / 2); ctx.closePath();
}
function drawOverlay() {
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const placed = [];
  const free = (r) => placed.every((p) => r.x > p.x + p.w || r.x + r.w < p.x || r.y > p.y + p.h || r.y + r.h < p.y);
  const mapW = W - panelW();
  const inMap = (x, y) => x > 0 && x < mapW && y > 0 && y < mapH();
  const at = (i) => toScreen(XY[2 * i], XY[2 * i + 1]);
  // keep labels off the title lockup and the zoom buttons
  for (const el of [$(".lockup"), $("#zoom")]) { const r = el.getBoundingClientRect(); placed.push({ x: r.left - 6, y: r.top - 6, w: r.width + 12, h: r.height + 12 }); }

  // region names: the map's "continents", shown while browsing, hidden during a search
  const z = view.s / fitS;
  const ra = simField ? 0 : Math.max(0, Math.min(0.8, 1.8 - z * 0.45));
  if (ra > 0.02) {
    ctx.font = `600 ${z > 2 ? 13 : 11.5}px "Barlow Condensed", sans-serif`;
    if ("letterSpacing" in ctx) ctx.letterSpacing = "2px";
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    const regs = [...M.regions].sort((a, b) => b.n - a.n);
    const maxLabels = Math.round(10 * Math.min(4, z));
    let shown = 0;
    for (const r of regs) {
      if (shown >= maxLabels) break;
      const [x, y] = toScreen(r.x, r.y);
      const t = r.label.toUpperCase();
      const w = ctx.measureText(t).width;
      const box = { x: x - w / 2 - 16, y: y - 14, w: w + 32, h: 28 };
      if (box.x < 8 || box.x + box.w > mapW - 8 || y < 70 || y > mapH() - 70 || !free(box)) continue;
      placed.push(box); shown++;
      ctx.lineJoin = "round"; ctx.lineWidth = 4; ctx.strokeStyle = `rgba(3,4,10,${ra * 0.9})`;
      ctx.strokeText(t, x, y);
      ctx.fillStyle = `rgba(226,234,252,${ra})`;
      ctx.fillText(t, x, y);
    }
    if ("letterSpacing" in ctx) ctx.letterSpacing = "0px";
    placed.length = 2;
  }

  const likes = state.picks.filter((p) => !p.avoid).map((p) => p.i);
  const top = simField ? results.slice(0, 10) : [];

  // curved links from each recommendation back to the pick it is nearest to
  ctx.lineWidth = 1;
  for (const r of top) {
    const [x, y] = at(r.i);
    const src = likes.includes(r.near) ? r.near : likes[0];
    const [sx, sy] = at(src);
    if (![sx, sy, x, y].every(Number.isFinite)) continue;
    const mx = (sx + x) / 2, my = (sy + y) / 2, dx = x - sx, dy = y - sy;
    const g = ctx.createLinearGradient(sx, sy, x, y);
    g.addColorStop(0, `rgba(${AMBER},.5)`); g.addColorStop(1, `rgba(${AQUA},.32)`);
    ctx.strokeStyle = g;
    ctx.beginPath(); ctx.moveTo(sx, sy); ctx.quadraticCurveTo(mx - dy * 0.18, my + dx * 0.18, x, y); ctx.stroke();
  }

  // markers
  for (const p of state.picks) {
    const [x, y] = at(p.i);
    if (!inMap(x, y)) continue;
    const c = p.avoid ? ROSE : AMBER;
    ctx.strokeStyle = `rgba(${c},.9)`; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(x, y, 9, 0, Math.PI * 2); ctx.stroke();
    ctx.strokeStyle = `rgba(${c},.3)`; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(x, y, 16, 0, Math.PI * 2); ctx.stroke();
    if (p.avoid) { ctx.strokeStyle = `rgba(${c},.9)`; ctx.beginPath(); ctx.moveTo(x - 4, y - 4); ctx.lineTo(x + 4, y + 4); ctx.moveTo(x + 4, y - 4); ctx.lineTo(x - 4, y + 4); ctx.stroke(); }
    placed.push({ x: x - 16, y: y - 16, w: 32, h: 32 });
  }

  // labels: picks first so they win placement, then the hovered card, then the top results
  for (const p of state.picks) label(p.i, { text: M.name[p.i].toUpperCase(), rgb: p.avoid ? ROSE : AMBER, big: true });
  if (hot >= 0) { const k = results.findIndex((r) => r.i === hot); if (k >= 0) label(hot, { text: M.name[hot], rank: k + 1, pct: pctOf(results[k].sim), rgb: AQUA, strong: true }); }
  for (const [k, r] of top.entries()) if (r.i !== hot) label(r.i, { text: M.name[r.i], rank: k + 1, pct: pctOf(r.sim), rgb: AQUA });

  function label(i, o) {
    const [x, y] = at(i);
    if (!inMap(x, y)) return;
    const name = o.text.length > 30 ? o.text.slice(0, 28).trim() + "…" : o.text;
    const fName = o.big ? `700 15px "Barlow Condensed", sans-serif` : `600 12px Manrope, sans-serif`;
    const fMono = `500 10.5px "JetBrains Mono", monospace`;
    ctx.font = fName; const nw = ctx.measureText(name).width;
    ctx.font = fMono;
    const rk = o.rank ? String(o.rank) : "", pc = o.pct != null ? o.pct + "%" : "";
    const rw = rk ? ctx.measureText(rk).width + 8 : 0, pw = pc ? ctx.measureText(pc).width + 8 : 0;
    const h = o.big ? 24 : 22, w = 10 + rw + nw + pw + 10;
    for (const rad of [14, 30, 52, 80, 115]) for (let a = 0; a < 8; a++) {
      const t = (a / 8) * Math.PI * 2 - Math.PI / 8;
      const ox = Math.cos(t) * rad, oy = Math.sin(t) * rad * 0.75;
      const box = { x: x + (ox >= 0 ? ox : ox - w), y: y + oy - h / 2, w, h };
      if (box.x < 6 || box.x + w > mapW - 6 || box.y < 6 || box.y + h > mapH() - 6 || !free({ x: box.x - 3, y: box.y - 3, w: w + 6, h: h + 6 })) continue;
      placed.push(box);
      // leader
      const ex = ox >= 0 ? box.x : box.x + w, ey = box.y + h / 2;
      if (Math.hypot(ex - x, ey - y) > 12) {
        ctx.strokeStyle = `rgba(${o.rgb},.45)`; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(ex, ey); ctx.stroke();
      }
      pill(box.x, box.y, w, h);
      ctx.fillStyle = o.strong ? "rgba(8,30,30,.92)" : "rgba(6,9,20,.84)"; ctx.fill();
      ctx.strokeStyle = `rgba(${o.rgb},${o.strong ? 0.9 : 0.4})`; ctx.lineWidth = 1; ctx.stroke();
      ctx.textBaseline = "middle"; ctx.textAlign = "left";
      let cx = box.x + 10;
      if (rk) { ctx.font = fMono; ctx.fillStyle = `rgba(${o.rgb},1)`; ctx.fillText(rk, cx, ey + 0.5); cx += rw; }
      ctx.font = fName; ctx.fillStyle = o.big ? `rgba(${o.rgb},1)` : "#eef4fb"; ctx.fillText(name, cx, ey + (o.big ? 1 : 0.5)); cx += nw + 8;
      if (pc) { ctx.font = fMono; ctx.fillStyle = "rgba(170,182,206,.85)"; ctx.fillText(pc, cx, ey + 0.5); }
      return;
    }
  }

  for (const [i, col] of [[hot, AQUA], [hover, "255,255,255"]]) {
    if (i < 0) continue;
    const [x, y] = at(i);
    ctx.strokeStyle = `rgba(${col},.95)`; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(x, y, 8, 0, Math.PI * 2); ctx.stroke();
    ctx.strokeStyle = `rgba(${col},.3)`;
    ctx.beginPath(); ctx.arc(x, y, 14, 0, Math.PI * 2); ctx.stroke();
  }
}

/* camera */
let fly = null;
function flyTo(cx, cy, s, ms = 750) { userMoved = true; fly = { from: { ...view }, to: { cx, cy, s }, t0: performance.now(), ms }; dirty(); }
function stepFly() {
  if (!fly) return;
  const t = Math.min(1, (performance.now() - fly.t0) / fly.ms), e = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  if (!(fly.from.s > 0)) fly.from.s = fly.to.s;
  const ls = Math.log(fly.from.s) + (Math.log(fly.to.s) - Math.log(fly.from.s)) * e;
  view.cx = fly.from.cx + (fly.to.cx - fly.from.cx) * e;
  view.cy = fly.from.cy + (fly.to.cy - fly.from.cy) * e;
  view.s = Math.exp(ls);
  if (t >= 1) fly = null;
}
function flyToQuery() {
  if (!state.picks.length) return;
  const ids = [...state.picks.map((p) => p.i), ...results.slice(0, 12).map((r) => r.i)];
  let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
  for (const i of ids) { const x = XY[2 * i], y = XY[2 * i + 1]; x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
  const pad = 1.6;
  const s = Math.min((W - panelW()) / ((x1 - x0) * pad + 0.4), mapH() / ((y1 - y0) * pad + 0.4), fitS * 40);
  flyTo((x0 + x1) / 2, (y0 + y1) / 2, Math.max(s, fitS));
}
function zoomAt(px, py, f) {
  fly = null; userMoved = true;
  const [dx, dy] = toData(px, py);
  view.s = Math.max(fitS * 0.5, Math.min(fitS * 200, view.s * f));
  const [nx, ny] = toData(px, py);
  view.cx += dx - nx; view.cy += dy - ny;
  dirty();
}

/* hover picking via a uniform grid in data space */
let grid, G = 0, gx0, gy0, gcell;
function buildGrid() {
  gx0 = 1e9; gy0 = 1e9; let gx1 = -1e9, gy1 = -1e9;
  for (let i = 0; i < N; i++) { gx0 = Math.min(gx0, XY[2 * i]); gy0 = Math.min(gy0, XY[2 * i + 1]); gx1 = Math.max(gx1, XY[2 * i]); gy1 = Math.max(gy1, XY[2 * i + 1]); }
  G = 256; gcell = Math.max(gx1 - gx0, gy1 - gy0) / G + 1e-6;
  grid = Array.from({ length: G * G }, () => []);
  for (let i = 0; i < N; i++) {
    const cx = Math.min(G - 1, ((XY[2 * i] - gx0) / gcell) | 0), cy = Math.min(G - 1, ((XY[2 * i + 1] - gy0) / gcell) | 0);
    grid[cy * G + cx].push(i);
  }
}
function pick(px, py) {
  const [x, y] = toData(px, py);
  const r = 10 / view.s, rc = Math.ceil(r / gcell);
  const cx = ((x - gx0) / gcell) | 0, cy = ((y - gy0) / gcell) | 0;
  let best = -1, bd = r * r, bestPrio = -1;
  for (let j = cy - rc; j <= cy + rc; j++) for (let i = cx - rc; i <= cx + rc; i++) {
    if (i < 0 || j < 0 || i >= G || j >= G) continue;
    for (const k of grid[j * G + i]) {
      if (famMask[FAM[k]] < 0.5) continue;
      const dx = XY[2 * k] - x, dy = XY[2 * k + 1] - y, d = dx * dx + dy * dy;
      if (d > r * r) continue;
      // prefer highlighted games, then popular ones, then the closest
      const prio = (simField && results.some((q) => q.i === k)) || state.picks.some((p) => p.i === k) ? 2 : POP[k] > 0.5 ? 1 : 0;
      if (prio > bestPrio || (prio === bestPrio && d < bd)) { best = k; bd = d; bestPrio = prio; }
    }
  }
  return best;
}

function showTip(i, px, py) {
  const tip = $("#tip");
  if (i < 0) { tip.style.display = "none"; return; }
  const pct = REV[i] ? Math.round((100 * POS[i]) / REV[i]) : 0;
  const [rl, rc] = steamRating(POS[i], REV[i]);
  const r = results.findIndex((q) => q.i === i);
  const pickIdx = state.picks.findIndex((p) => p.i === i);
  const badge = pickIdx >= 0 ? `<span class="badge" style="color:${state.picks[pickIdx].avoid ? "var(--rose)" : "var(--amber)"}">${state.picks[pickIdx].avoid ? "avoiding" : "your pick"}</span>`
    : r >= 0 ? `<span class="badge" style="color:var(--aqua)">#${r + 1} · ${pctOf(results[r].sim)}% match</span>` : "";
  tip.innerHTML = `<div class="hero"><img src="${header(M.appid[i])}" alt="" onerror="this.style.visibility='hidden'">${badge}</div>
    <div class="in">
      <div class="t">${esc(M.name[i])}</div>
      <div class="m"><span class="rating ${rc}">${rl}</span><span>${pct}% of ${fmtN(REV[i])}</span><span>${YEAR[i] || "—"}</span><span>${priceOf(i)}</span></div>
      ${M.desc && M.desc[i] ? `<div class="d">${esc(M.desc[i])}</div>` : ""}
      <div class="tags">${topTags(i, 5).map((t) => `<span>${esc(t)}</span>`).join("")}</div>
      <div class="cta">${pickIdx >= 0 ? "Already in your search" : "Click to add to your search"}</div>
    </div>`;
  tip.style.display = "block";
  const tw = tip.offsetWidth, th = tip.offsetHeight;
  let x = px + 20, y = py + 20;
  if (x + tw > W - panelW() - 8) x = px - tw - 20;
  if (y + th > H - 8) y = py - th - 20;
  tip.style.left = Math.max(8, x) + "px"; tip.style.top = Math.max(8, y) + "px";
}

function bindMap() {
  const ptrs = new Map();
  let drag = null, moved = 0, pinch = null;
  glc.addEventListener("pointerdown", (e) => {
    glc.setPointerCapture(e.pointerId);
    ptrs.set(e.pointerId, [e.clientX, e.clientY]);
    moved = 0; fly = null;
    if (ptrs.size === 1) drag = { x: e.clientX, y: e.clientY, cx: view.cx, cy: view.cy };
    if (ptrs.size === 2) { const [a, b] = [...ptrs.values()]; pinch = { d: Math.hypot(a[0] - b[0], a[1] - b[1]) }; drag = null; }
  });
  glc.addEventListener("pointermove", (e) => {
    if (ptrs.has(e.pointerId)) ptrs.set(e.pointerId, [e.clientX, e.clientY]);
    if (pinch && ptrs.size === 2) {
      const [a, b] = [...ptrs.values()], d = Math.hypot(a[0] - b[0], a[1] - b[1]);
      zoomAt((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, d / pinch.d); pinch.d = d; moved = 99; return;
    }
    if (drag) {
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      moved = Math.max(moved, Math.hypot(dx, dy));
      if (moved > 3) userMoved = true;
      view.cx = drag.cx - dx / view.s; view.cy = drag.cy + dy / view.s;
      showTip(-1); dirty(); return;
    }
    if (e.pointerType === "mouse") {
      const i = pick(e.clientX, e.clientY);
      if (i !== hover) { hover = i; dirty(); }
      glc.style.cursor = i >= 0 ? "pointer" : "grab";
      showTip(i, e.clientX, e.clientY);
    }
  });
  const end = (e) => {
    ptrs.delete(e.pointerId);
    if (ptrs.size < 2) pinch = null;
    if (drag && moved < 5) {
      const i = pick(e.clientX, e.clientY);
      if (i >= 0) {
        if (e.pointerType !== "mouse" && hover !== i) { hover = i; showTip(i, e.clientX, e.clientY); dirty(); }
        else { showTip(-1); addPick(i); }
      }
    }
    drag = null;
  };
  glc.addEventListener("pointerup", end);
  glc.addEventListener("pointercancel", end);
  glc.addEventListener("pointerleave", () => { if (hover >= 0) { hover = -1; dirty(); } showTip(-1); });
  glc.addEventListener("wheel", (e) => { e.preventDefault(); zoomAt(e.clientX, e.clientY, Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0018))); }, { passive: false });
}

/* ---------------- boot ---------------- */
(async () => {
  try {
    await load();
    initGL();
    buildGrid();
    bindUI();
    bindMap();
    addEventListener("resize", resize);
    resize();
    renderStarts();
    readHash();
    update({ fly: false });
    if (state.picks.length) flyToQuery();
    $("#loading").classList.add("done");
    fetch("data/desc.json").then((r) => r.json()).then((d) => (M.desc = d.desc)).catch(() => {});
  } catch (err) {
    console.error(err);
    $("#loadmsg").textContent = "Could not load the map: " + err.message;
  }
})();
