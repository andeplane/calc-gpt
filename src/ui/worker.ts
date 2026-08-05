/// <reference lib="webworker" />
/** Training runs in a worker so the page stays responsive. The worker owns the
 * model; the page sends control messages and receives metrics. */
import {
  AdditionGenerator,
  Batcher,
  MixtureGenerator,
  SubtractionGenerator,
} from "../engine/data";
import { CalcGPT } from "../engine/model";
import { Adam } from "../engine/optimizer";
import { Mulberry32 } from "../engine/rng";
import { Tokenizer } from "../engine/tokenizer";
import { ArraySink, Trainer, makeEvalSet } from "../engine/trainer";

export type WorkerCommand =
  | { type: "start"; phase: 1 | 2 }
  | { type: "pause" }
  | { type: "predict"; id: number; prompt: string };

export type WorkerEvent =
  | { type: "ready"; numParams: number }
  | { type: "metrics"; step: number; loss: number; phase: 1 | 2; stepsPerSec: number }
  | {
      type: "eval";
      step: number;
      phase: 1 | 2;
      addAcc: number;
      subAcc: number | null;
      samples: { prompt: string; pred: string; truth: string; ok: boolean }[];
    }
  | { type: "prediction"; id: number; prompt: string; answer: string; truth: number | null }
  | { type: "paused"; step: number };

const tokenizer = new Tokenizer();
const MAX_OPERAND = 99;
const BLOCK_SIZE = 12;
const EVAL_EVERY = 250;
const CHUNK = 20;

// setTimeout(0) is clamped to ~4ms in workers; a MessageChannel macrotask
// yields to onmessage without the clamp, so training runs at full speed.
const wake = new MessageChannel();
const scheduleLoop = () => wake.port2.postMessage(null);

const model = new CalcGPT(
  { vocabSize: tokenizer.size, blockSize: BLOCK_SIZE, nLayer: 2, nHead: 4, nEmbd: 32 },
  new Mulberry32(1337),
);

const sink = new ArraySink();
const addGen = () => new AdditionGenerator(new Mulberry32(1), MAX_OPERAND);
const mixedGen = () =>
  new MixtureGenerator(new Mulberry32(2), [
    { generator: new AdditionGenerator(new Mulberry32(3), MAX_OPERAND), weight: 1 },
    { generator: new SubtractionGenerator(new Mulberry32(4), MAX_OPERAND), weight: 1 },
  ]);

const trainer = new Trainer({
  model,
  optimizer: new Adam(3e-3),
  generator: addGen(),
  batcher: new Batcher(tokenizer, BLOCK_SIZE),
  tokenizer,
  sink,
  batchSize: 32,
});

const addEval = makeEvalSet(new AdditionGenerator(new Mulberry32(101), MAX_OPERAND), 60);
const subEval = makeEvalSet(new SubtractionGenerator(new Mulberry32(102), MAX_OPERAND), 60);

let running = false;
let phase: 1 | 2 = 1;

function post(e: WorkerEvent): void {
  (self as unknown as Worker).postMessage(e);
}

function truthOf(prompt: string): number | null {
  const m = prompt.match(/^(\d+)([+\-])(\d+)=$/);
  if (!m) return null;
  const a = Number(m[1]);
  const b = Number(m[3]);
  return m[2] === "+" ? a + b : a - b;
}

function runEval(): void {
  const add = trainer.evaluate(addEval);
  const sub = phase === 2 ? trainer.evaluate(subEval) : null;
  const sampleCases = [...addEval.slice(0, 3), ...(phase === 2 ? subEval.slice(0, 3) : [])];
  const samples = sampleCases.map((c) => {
    const pred = model.greedyAnswer(tokenizer, c.prompt);
    return { prompt: c.prompt, pred, truth: c.answer, ok: pred === c.answer };
  });
  post({
    type: "eval",
    step: trainer.step,
    phase,
    addAcc: add.accuracy,
    subAcc: sub ? sub.accuracy : null,
    samples,
  });
}

function loop(): void {
  if (!running) return;
  const t0 = performance.now();
  let loss = 0;
  for (let i = 0; i < CHUNK; i++) {
    loss = trainer.trainStep();
    if (trainer.step % EVAL_EVERY === 0) runEval();
  }
  const dt = (performance.now() - t0) / 1000;
  post({ type: "metrics", step: trainer.step, loss, phase, stepsPerSec: CHUNK / dt });
  scheduleLoop();
}
wake.port1.onmessage = loop;

self.onmessage = (msg: MessageEvent<WorkerCommand>) => {
  const cmd = msg.data;
  if (cmd.type === "start") {
    if (cmd.phase === 2 && phase === 1) {
      phase = 2;
      trainer.setGenerator(mixedGen());
    }
    if (!running) {
      running = true;
      loop();
    }
  } else if (cmd.type === "pause") {
    running = false;
    post({ type: "paused", step: trainer.step });
  } else {
    let answer: string;
    try {
      answer = model.greedyAnswer(tokenizer, cmd.prompt);
    } catch {
      answer = "(invalid prompt)";
    }
    post({ type: "prediction", id: cmd.id, prompt: cmd.prompt, answer, truth: truthOf(cmd.prompt) });
  }
};

post({ type: "ready", numParams: model.numParams });
