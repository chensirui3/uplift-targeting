// Shared pieces for every section of the demo: formatting, data loading, URL state, the
// tooltip, chart parts, tables, sliders and redrawing on resize. Nothing in this file knows
// about a particular dataset. Colors come from CSS variables, so light and dark both work.

export const d3 = window.d3;
export const MINUS = "−";

// --- formatting ---------------------------------------------------------------------------
export function fmt(v, { d = 0, cur = false, plus = false } = {}) {
  const body = d3.format(`,.${d}f`)(Math.abs(v));
  const sign = /^[0.,]+$/.test(body) ? "" : v < 0 ? MINUS : plus ? "+" : "";
  return `${sign}${cur ? "$" : ""}${body}`;
}
export const usd = (v, d = 0) => fmt(v, { d, cur: true });
export const usdPlus = (v, d = 0) => fmt(v, { d, cur: true, plus: true });
export const num = (v, d = 1) => fmt(v, { d });
export const numPlus = (v, d = 1) => fmt(v, { d, plus: true });
export const pct = (v, d = 0) => `${fmt(v * 100, { d })}%`;
export const int = (v) => d3.format(",")(Math.round(v));
export const about = (v) => int(Math.round(v / 1000) * 1000);
export const span = (lo, hi, f) => `${f(lo)} to ${f(hi)}`;
export const cents = (v) => Math.round(v * 100);
export const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
export const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const WORDS = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"];
export const word = (n) => WORDS[n] ?? String(n);

export function elem(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

// --- data ---------------------------------------------------------------------------------
export async function loadJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

// --- URL state: one hash for the whole page, so any view can be shared as a link. Each
// section reads and writes only its own keys; a new section should prefix its keys.
export function readParam(key) {
  return new URLSearchParams(location.hash.slice(1)).get(key);
}
export function readNumber(key) {
  const v = readParam(key);
  return v != null && v !== "" && Number.isFinite(Number(v)) ? Number(v) : null;
}
export function writeParams(values) {
  const p = new URLSearchParams(location.hash.slice(1));
  for (const [k, v] of Object.entries(values)) p.set(k, v);
  history.replaceState(null, "", `#${p}`);
}

// --- tooltip ------------------------------------------------------------------------------
const tip = document.getElementById("tooltip");
/** rows: [{ key: series class or null, value, label }]. Text only, never HTML. */
export function showTip(event, title, rows) {
  tip.replaceChildren(elem("div", "t-title", title));
  for (const r of rows) {
    const row = elem("div", "t-row");
    if (r.key) row.append(elem("span", `key line ${r.key}`));
    row.append(elem("strong", "", r.value));
    if (r.label) row.append(elem("span", "t-sub", r.label));
    tip.append(row);
  }
  tip.hidden = false;
  const pad = 14;
  const box = tip.getBoundingClientRect();
  let x = event.clientX + pad;
  let y = event.clientY + pad;
  if (x + box.width > innerWidth - 8) x = event.clientX - box.width - pad;
  if (y + box.height > innerHeight - 8) y = event.clientY - box.height - pad;
  tip.style.left = `${Math.max(8, x)}px`;
  tip.style.top = `${Math.max(8, y)}px`;
}
export const hideTip = () => { tip.hidden = true; };

// --- chart pieces -------------------------------------------------------------------------
export function svgFor(el, height) {
  const W = Math.max(280, Math.floor(el.clientWidth));
  const H = height(W);
  const svg = d3.create("svg").attr("viewBox", `0 0 ${W} ${H}`).attr("width", W).attr("height", H);
  return { svg, W, H };
}
export function yAxis(svg, y, m, W, format, count) {
  const ticks = y.ticks(count);
  svg.append("g").attr("class", "grid").selectAll("line").data(ticks).join("line")
    .attr("x1", m.left).attr("x2", W - m.right).attr("y1", (d) => y(d)).attr("y2", (d) => y(d));
  svg.append("g").selectAll("text").data(ticks).join("text").attr("class", "axis-text")
    .attr("x", m.left - 8).attr("y", (d) => y(d)).attr("dy", "0.32em").attr("text-anchor", "end")
    .text(format);
}
export function xAxis(svg, x, m, W, H, ticks, format, title) {
  svg.append("g").selectAll("text").data(ticks).join("text").attr("class", "axis-text")
    .attr("x", (d) => x(d)).attr("y", H - m.bottom + 18).attr("text-anchor", "middle").text(format);
  svg.append("text").attr("class", "axis-title").attr("text-anchor", "middle")
    .attr("x", (m.left + W - m.right) / 2).attr("y", H - 8).text(title);
}
export function baseline(svg, y, m, W) {
  const [lo, hi] = y.domain();
  const v = lo <= 0 && hi >= 0 ? 0 : lo;
  svg.append("line").attr("class", "axis-line").attr("x1", m.left).attr("x2", W - m.right)
    .attr("y1", y(v)).attr("y2", y(v));
}
export function marker(svg, x, m, H) {
  svg.append("line").attr("class", "marker-line").attr("x1", x).attr("x2", x)
    .attr("y1", m.top).attr("y2", H - m.bottom);
}
export function dot(svg, cx, cy, cls, r = 5) {
  svg.append("circle").attr("class", `dot ${cls}`).attr("cx", cx).attr("cy", cy).attr("r", r);
}
/** A value beside its mark: to the right if there's room, above unless it would leave the plot. */
export function valueLabel(svg, x, y, text, W, m) {
  const right = x + 72 < W - m.right;
  const above = y - 12 > m.top + 4;
  svg.append("text").attr("class", "value-label")
    .attr("x", right ? x + 9 : x - 9).attr("y", above ? y - 10 : y + 20)
    .attr("text-anchor", right ? "start" : "end").text(text);
}
/** Crosshair that snaps to the nearest x in `xs`, with a tooltip; a click picks that x. */
export function crosshair(svg, { x, xs, m, W, H, tipFor, pick }) {
  const line = svg.append("line").attr("class", "crosshair")
    .attr("y1", m.top).attr("y2", H - m.bottom).attr("visibility", "hidden");
  const bisect = d3.bisector((d) => d).center;
  const nearest = (event, node) => bisect(xs, x.invert(d3.pointer(event, node)[0]));
  svg.append("rect").attr("class", "hit")
    .attr("x", m.left - 8).attr("width", W - m.left - m.right + 16)
    .attr("y", m.top).attr("height", H - m.top - m.bottom)
    .on("pointerenter pointermove", function (event) {
      const i = nearest(event, this);
      line.attr("x1", x(xs[i])).attr("x2", x(xs[i])).attr("visibility", "visible");
      showTip(event, ...tipFor(i));
    })
    .on("pointerleave", () => { line.attr("visibility", "hidden"); hideTip(); })
    .on("click", function (event) { hideTip(); pick(nearest(event, this)); });
}
/** A bar from the baseline to `end`: square at the baseline, 4px rounded at the data end. */
export function barPath(x, w, base, end) {
  const h = Math.abs(end - base);
  if (h < 0.5) return "";
  const r = Math.min(4, w / 2, h);
  const s = end < base ? 1 : -1; // grows up (1) or down (-1) from the baseline
  return `M${x},${base}V${end + s * r}Q${x},${end} ${x + r},${end}H${x + w - r}` +
    `Q${x + w},${end} ${x + w},${end + s * r}V${base}Z`;
}
export function fillTable(table, head, rows, caption) {
  table.replaceChildren();
  if (caption) table.createCaption().textContent = caption;
  const tr = table.createTHead().insertRow();
  head.forEach((h) => tr.append(elem("th", "", h)));
  const body = table.createTBody();
  rows.forEach((row) => {
    const r = body.insertRow();
    row.forEach((cell) => { r.insertCell().textContent = cell; });
  });
}

// --- sliders: a .slider block holds a range input, an <output> and the two .ends labels.
export function bindSlider(block, { min, max, step, format, onInput }) {
  const input = block.querySelector('input[type="range"]');
  const out = block.querySelector("output");
  const [lo, hi] = block.querySelectorAll(".ends span");
  Object.assign(input, { min, max, step });
  lo.textContent = format(min);
  hi.textContent = format(max);
  input.addEventListener("input", () => onInput(Number(input.value)));
  return {
    set(value, text, valueText) {
      input.value = value;
      input.style.setProperty("--fill", `${((value - min) / (max - min)) * 100}%`);
      input.setAttribute("aria-valuetext", valueText);
      out.textContent = text;
    },
  };
}

// --- redraw every chart when the page width changes ----------------------------------------
const redraws = [];
export function onResize(redraw) { redraws.push(redraw); }
const main = document.querySelector("main");
let lastWidth = main.clientWidth;
let frame;
new ResizeObserver(() => {
  if (main.clientWidth === lastWidth) return;
  lastWidth = main.clientWidth;
  cancelAnimationFrame(frame);
  frame = requestAnimationFrame(() => redraws.forEach((redraw) => redraw()));
}).observe(main);
