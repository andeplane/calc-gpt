/** Minimal SVG line charts for the blog (no dependencies). */

export interface Series {
  name: string;
  color: string;
  points: { x: number; y: number }[];
}

export interface ChartOptions {
  logY?: boolean;
  yDomain?: [number, number];
  yFormat?: (v: number) => string;
  /** Vertical annotation lines, e.g. a curriculum phase boundary. */
  annotations?: { x: number; label: string }[];
}

const NS = "http://www.w3.org/2000/svg";

function el<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attrs: Record<string, string | number>,
): SVGElementTagNameMap[K] {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  return node;
}

function css(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function drawChart(
  container: HTMLElement,
  series: Series[],
  opts: ChartOptions = {},
): void {
  container.textContent = "";
  const W = 860;
  const H = 280;
  const ml = 54;
  const mr = 14;
  const mt = 12;
  const mb = 28;
  const iw = W - ml - mr;
  const ih = H - mt - mb;
  const all = series.flatMap((s) => s.points);
  if (all.length === 0) {
    const empty = document.createElement("div");
    empty.className = "chart-empty";
    empty.textContent = "waiting for data…";
    container.append(empty);
    return;
  }
  const xMin = Math.min(...all.map((p) => p.x));
  const xMax = Math.max(...all.map((p) => p.x));
  const yFormat = opts.yFormat ?? ((v: number) => v.toPrecision(3));

  let yTicks: { v: number; label: string }[];
  let yTo: (v: number) => number;
  if (opts.logY) {
    const lo = Math.max(1e-6, Math.min(...all.map((p) => p.y)));
    const hi = Math.max(...all.map((p) => p.y));
    let eLo = Math.floor(Math.log10(lo));
    let eHi = Math.ceil(Math.log10(hi));
    if (eHi === eLo) eHi += 1;
    yTo = (v) => mt + ih * (1 - (Math.log10(Math.max(v, 1e-6)) - eLo) / (eHi - eLo));
    yTicks = [];
    for (let e = eLo; e <= eHi; e++) {
      yTicks.push({ v: 10 ** e, label: 10 ** e >= 1 ? String(10 ** e) : (10 ** e).toFixed(-e) });
    }
  } else {
    const [dLo, dHi] = opts.yDomain ?? [
      Math.min(...all.map((p) => p.y)),
      Math.max(...all.map((p) => p.y)),
    ];
    yTo = (v) => mt + ih * (1 - (v - dLo) / (dHi - dLo || 1));
    yTicks = [0, 0.25, 0.5, 0.75, 1].map((f) => {
      const v = dLo + f * (dHi - dLo);
      return { v, label: yFormat(v) };
    });
  }
  const xTo = (v: number) => ml + (xMax === xMin ? iw / 2 : (iw * (v - xMin)) / (xMax - xMin));

  const svg = el("svg", { viewBox: `0 0 ${W} ${H}` });
  for (const t of yTicks) {
    svg.append(el("line", { x1: ml, x2: W - mr, y1: yTo(t.v), y2: yTo(t.v), class: "grid" }));
    const label = el("text", { x: ml - 8, y: yTo(t.v) + 4, "text-anchor": "end", class: "tick" });
    label.textContent = t.label;
    svg.append(label);
  }
  const xStep = niceStep((xMax - xMin) / 6 || 1);
  for (let x = Math.ceil(xMin / xStep) * xStep; x <= xMax; x += xStep) {
    const label = el("text", { x: xTo(x), y: H - 8, "text-anchor": "middle", class: "tick" });
    label.textContent = x.toLocaleString();
    svg.append(label);
  }
  svg.append(el("line", { x1: ml, x2: W - mr, y1: mt + ih, y2: mt + ih, class: "axis" }));

  for (const a of opts.annotations ?? []) {
    svg.append(el("line", { x1: xTo(a.x), x2: xTo(a.x), y1: mt, y2: mt + ih, class: "annot" }));
    const label = el("text", { x: xTo(a.x) + 6, y: mt + 14, class: "annot-label" });
    label.textContent = a.label;
    svg.append(label);
  }

  for (const s of series) {
    if (s.points.length === 0) continue;
    const d = s.points
      .map((p, i) => `${i ? "L" : "M"}${xTo(p.x).toFixed(1)},${yTo(p.y).toFixed(1)}`)
      .join("");
    svg.append(el("path", { d, fill: "none", stroke: s.color, "stroke-width": 2,
      "stroke-linejoin": "round", "stroke-linecap": "round" }));
    const end = s.points[s.points.length - 1];
    svg.append(el("circle", { cx: xTo(end.x), cy: yTo(end.y), r: 4, fill: s.color,
      stroke: css("--surface"), "stroke-width": 2 }));
  }
  container.append(svg);
}

function niceStep(raw: number): number {
  const mag = 10 ** Math.floor(Math.log10(raw));
  for (const m of [1, 2, 5, 10]) if (m * mag >= raw) return m * mag;
  return 10 * mag;
}
