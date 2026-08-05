import "./style.css";
import { Tokenizer } from "./engine/tokenizer";
import { drawChart, type Series } from "./ui/chart";
import type { WorkerCommand, WorkerEvent } from "./ui/worker";

const $ = <T extends HTMLElement = HTMLElement>(sel: string): T => {
  const node = document.querySelector<T>(sel);
  if (!node) throw new Error(`missing element: ${sel}`);
  return node;
};

const css = (name: string): string =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/* ---------------- tokenizer playground ---------------- */
const tokenizer = new Tokenizer();
const tokInput = $<HTMLInputElement>("#tok-input");
const tokOut = $("#tok-output");

function renderTokens(): void {
  const text = tokInput.value;
  tokOut.textContent = "";
  for (const ch of text) {
    const chip = document.createElement("span");
    try {
      const id = tokenizer.idOf(ch);
      chip.className = "tok-chip";
      chip.innerHTML = `<span class="tok-ch">${ch}</span><span class="tok-id">${id}</span>`;
    } catch {
      chip.className = "tok-chip tok-bad";
      chip.innerHTML = `<span class="tok-ch">${ch}</span><span class="tok-id">?</span>`;
    }
    tokOut.append(chip);
  }
}
tokInput.addEventListener("input", renderTokens);
renderTokens();

/* ---------------- training worker ---------------- */
const worker = new Worker(new URL("./ui/worker.ts", import.meta.url), { type: "module" });

interface LossPoint { x: number; y: number }
const lossPoints: LossPoint[] = [];
const addAccPoints: LossPoint[] = [];
const subAccPoints: LossPoint[] = [];
let phase2Start: number | null = null;
let running = false;
let predictionCounter = 0;

const statusEl = $("#train-status");
const startBtn = $<HTMLButtonElement>("#btn-start");
const phase2Btn = $<HTMLButtonElement>("#btn-phase2");
const samplesEl = $("#train-samples");

function redraw(): void {
  const annotations = phase2Start !== null
    ? [{ x: phase2Start, label: "− joins the data" }]
    : [];
  const lossSeries: Series[] = [
    { name: "loss", color: css("--series-1"), points: thin(lossPoints) },
  ];
  drawChart($("#loss-chart"), lossSeries, { logY: true, annotations });
  const accSeries: Series[] = [
    { name: "addition", color: css("--series-2"), points: addAccPoints },
  ];
  if (subAccPoints.length > 0) {
    accSeries.push({ name: "subtraction", color: css("--series-3"), points: subAccPoints });
  }
  drawChart($("#acc-chart"), accSeries, {
    yDomain: [0, 1],
    yFormat: (v) => `${Math.round(v * 100)}%`,
    annotations,
  });
  const legend = accSeries
    .map((s) => `<span class="key"><span class="swatch" style="background:${s.color}"></span>${s.name}</span>`)
    .join("");
  $("#acc-legend").innerHTML = legend;
}

function thin(pts: LossPoint[], max = 800): LossPoint[] {
  if (pts.length <= max) return pts;
  const stride = Math.ceil(pts.length / max);
  return pts.filter((_, i) => i % stride === 0 || i === pts.length - 1);
}

worker.onmessage = (msg: MessageEvent<WorkerEvent>) => {
  const e = msg.data;
  if (e.type === "ready") {
    statusEl.textContent = `model ready — ${e.numParams.toLocaleString()} parameters, untrained`;
    startBtn.disabled = false;
  } else if (e.type === "metrics") {
    if (e.step % 50 === 0 || lossPoints.length < 20) {
      lossPoints.push({ x: e.step, y: e.loss });
      redraw();
    }
    statusEl.textContent =
      `step ${e.step.toLocaleString()} — loss ${e.loss.toFixed(4)} — ` +
      `${Math.round(e.stepsPerSec)} steps/s${e.phase === 2 ? " — phase 2" : ""}`;
  } else if (e.type === "eval") {
    addAccPoints.push({ x: e.step, y: e.addAcc });
    if (e.subAcc !== null) subAccPoints.push({ x: e.step, y: e.subAcc });
    redraw();
    samplesEl.innerHTML = e.samples
      .map(
        (s) =>
          `<div class="sample ${s.ok ? "ok" : "bad"}">${s.prompt}<strong>${s.pred}</strong>` +
          `${s.ok ? "" : ` <span class="truth">(should be ${s.truth})</span>`}</div>`,
      )
      .join("");
  } else if (e.type === "prediction") {
    const verdict =
      e.truth === null
        ? `<span class="muted">can't check that one</span>`
        : String(e.truth) === e.answer
          ? `<span class="ok-text">correct</span>`
          : `<span class="bad-text">wrong — it's ${e.truth}</span>`;
    $("#ask-result").innerHTML = `<strong>${e.prompt}${e.answer}</strong> ${verdict}`;
  } else {
    statusEl.textContent = `paused at step ${e.step.toLocaleString()}`;
  }
};

function send(cmd: WorkerCommand): void {
  worker.postMessage(cmd);
}

startBtn.addEventListener("click", () => {
  if (running) {
    send({ type: "pause" });
    running = false;
    startBtn.textContent = "Resume training";
  } else {
    send({ type: "start", phase: phase2Start !== null ? 2 : 1 });
    running = true;
    startBtn.textContent = "Pause";
    phase2Btn.disabled = phase2Start !== null;
  }
});

phase2Btn.addEventListener("click", () => {
  phase2Start = lossPoints.length > 0 ? lossPoints[lossPoints.length - 1].x : 0;
  send({ type: "start", phase: 2 });
  running = true;
  startBtn.textContent = "Pause";
  phase2Btn.disabled = true;
  $("#phase2-note").classList.remove("hidden");
  redraw();
});

/* ---------------- ask the model ---------------- */
const askInput = $<HTMLInputElement>("#ask-input");
const askBtn = $<HTMLButtonElement>("#ask-btn");

function ask(): void {
  let prompt = askInput.value.trim().replace(/\s+/g, "");
  if (prompt.length === 0) return;
  if (!prompt.endsWith("=")) prompt += "=";
  predictionCounter += 1;
  $("#ask-result").innerHTML = `<span class="muted">thinking…</span>`;
  send({ type: "predict", id: predictionCounter, prompt });
}
askBtn.addEventListener("click", ask);
askInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") ask();
});

redraw();
