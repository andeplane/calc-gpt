import type { Batch, Batcher, Problem, ProblemGenerator } from "./data";
import type { Optimizer } from "./optimizer";
import type { Tokenizer } from "./tokenizer";
import type { Param } from "./model";

export interface StepEvent {
  step: number;
  loss: number;
}

export interface EvalResult {
  accuracy: number;
  perOp: Record<string, number>;
}

/** Injectable metrics destination (UI chart adapter in the browser,
 * a plain array in tests). */
export interface MetricsSink {
  onStep(e: StepEvent): void;
}

export class ArraySink implements MetricsSink {
  readonly steps: StepEvent[] = [];

  onStep(e: StepEvent): void {
    this.steps.push(e);
  }
}

/** The minimal surface the trainer needs from a model — CalcGPT satisfies it,
 * and tests can substitute a stub. */
export interface TrainableModel {
  zeroGrads(): void;
  lossBackward(batch: Batch): number;
  greedyAnswer(tokenizer: Tokenizer, prompt: string): string;
  parameters(): Param[];
}

export interface TrainerDeps {
  model: TrainableModel;
  optimizer: Optimizer;
  generator: ProblemGenerator;
  batcher: Batcher;
  tokenizer: Tokenizer;
  sink: MetricsSink;
  batchSize: number;
}

/** Orchestrates the loop: sample problems → encode → loss+grads → update.
 * Every collaborator is constructor-injected; switching curriculum phase is
 * just `setGenerator` — the model keeps its weights and continues learning. */
export class Trainer {
  private stepCount = 0;

  constructor(private readonly deps: TrainerDeps) {}

  get step(): number {
    return this.stepCount;
  }

  get generator(): ProblemGenerator {
    return this.deps.generator;
  }

  /** Curriculum phase switch: same model, same optimizer state, new data. */
  setGenerator(generator: ProblemGenerator): void {
    this.deps.generator = generator;
  }

  trainStep(): number {
    const { model, optimizer, generator, batcher, sink, batchSize } = this.deps;
    const problems: Problem[] = [];
    for (let i = 0; i < batchSize; i++) problems.push(generator.sample());
    const batch = batcher.encode(problems);
    model.zeroGrads();
    const loss = model.lossBackward(batch);
    optimizer.step(model.parameters());
    this.stepCount += 1;
    sink.onStep({ step: this.stepCount, loss });
    return loss;
  }

  /** Exact-match accuracy via greedy decode, split per operator. */
  evaluate(cases: Problem[]): EvalResult {
    const { model, tokenizer } = this.deps;
    let correct = 0;
    const perOp: Record<string, { ok: number; n: number }> = {};
    for (const { prompt, answer } of cases) {
      const op = prompt.slice(1).match(/[+\-*/]/)![0]; // operator after first digit
      const pred = model.greedyAnswer(tokenizer, prompt);
      const ok = pred === answer;
      if (ok) correct += 1;
      perOp[op] ??= { ok: 0, n: 0 };
      perOp[op].ok += ok ? 1 : 0;
      perOp[op].n += 1;
    }
    const out: Record<string, number> = {};
    for (const [op, { ok, n }] of Object.entries(perOp)) out[op] = ok / n;
    return { accuracy: correct / cases.length, perOp: out };
  }
}

/** Fixed held-out problem set from a generator (deterministic if the
 * generator's rng is seeded). */
export function makeEvalSet(generator: ProblemGenerator, n: number): Problem[] {
  const cases: Problem[] = [];
  for (let i = 0; i < n; i++) cases.push(generator.sample());
  return cases;
}
