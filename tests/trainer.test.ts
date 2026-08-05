import { describe, expect, it } from "vitest";
import {
  AdditionGenerator,
  Batcher,
  MixtureGenerator,
  SubtractionGenerator,
  type Problem,
} from "../src/engine/data";
import { CalcGPT, type Param } from "../src/engine/model";
import { Adam, type Optimizer } from "../src/engine/optimizer";
import { Mulberry32 } from "../src/engine/rng";
import { Tokenizer } from "../src/engine/tokenizer";
import {
  ArraySink,
  Trainer,
  makeEvalSet,
  type TrainableModel,
} from "../src/engine/trainer";

const tok = new Tokenizer();

describe("Trainer with stubbed collaborators (dependency injection)", () => {
  function makeStubs() {
    const calls = { zeroGrads: 0, lossBackward: 0, optimizerSteps: 0 };
    const answers: Record<string, string> = { "1+2=": "3", "5+5=": "10", "9-4=": "5" };
    const model: TrainableModel = {
      zeroGrads: () => void (calls.zeroGrads += 1),
      lossBackward: () => {
        calls.lossBackward += 1;
        return 1 / calls.lossBackward;
      },
      greedyAnswer: (_t, prompt) => answers[prompt] ?? "?",
      parameters: () => [] as Param[],
    };
    const optimizer: Optimizer = {
      lr: 0.1,
      step: () => void (calls.optimizerSteps += 1),
    };
    return { calls, model, optimizer };
  }

  it("runs the loop: zero grads → backward → optimizer step → sink event", () => {
    const { calls, model, optimizer } = makeStubs();
    const sink = new ArraySink();
    const trainer = new Trainer({
      model,
      optimizer,
      generator: new AdditionGenerator(new Mulberry32(1)),
      batcher: new Batcher(tok, 12),
      tokenizer: tok,
      sink,
      batchSize: 4,
    });
    const l1 = trainer.trainStep();
    const l2 = trainer.trainStep();
    expect(l1).toBe(1);
    expect(l2).toBe(0.5);
    expect(calls).toEqual({ zeroGrads: 2, lossBackward: 2, optimizerSteps: 2 });
    expect(sink.steps).toEqual([
      { step: 1, loss: 1 },
      { step: 2, loss: 0.5 },
    ]);
    expect(trainer.step).toBe(2);
  });

  it("evaluates exact-match accuracy per operator", () => {
    const { model, optimizer } = makeStubs();
    const trainer = new Trainer({
      model,
      optimizer,
      generator: new AdditionGenerator(new Mulberry32(1)),
      batcher: new Batcher(tok, 12),
      tokenizer: tok,
      sink: new ArraySink(),
      batchSize: 1,
    });
    const cases: Problem[] = [
      { prompt: "1+2=", answer: "3" }, // right
      { prompt: "5+5=", answer: "11" }, // wrong (stub says 10)
      { prompt: "9-4=", answer: "5" }, // right
    ];
    const result = trainer.evaluate(cases);
    expect(result.accuracy).toBeCloseTo(2 / 3, 6);
    expect(result.perOp["+"]).toBeCloseTo(0.5, 6);
    expect(result.perOp["-"]).toBeCloseTo(1.0, 6);
  });

  it("swaps generators without touching the model (phase switch)", () => {
    const { model, optimizer } = makeStubs();
    const gen1 = new AdditionGenerator(new Mulberry32(1));
    const gen2 = new SubtractionGenerator(new Mulberry32(1));
    const trainer = new Trainer({
      model,
      optimizer,
      generator: gen1,
      batcher: new Batcher(tok, 12),
      tokenizer: tok,
      sink: new ArraySink(),
      batchSize: 1,
    });
    expect(trainer.generator).toBe(gen1);
    trainer.setGenerator(gen2);
    expect(trainer.generator).toBe(gen2);
  });
});

describe("makeEvalSet", () => {
  it("draws n problems deterministically from a seeded generator", () => {
    const a = makeEvalSet(new AdditionGenerator(new Mulberry32(9)), 5);
    const b = makeEvalSet(new AdditionGenerator(new Mulberry32(9)), 5);
    expect(a).toHaveLength(5);
    expect(a).toEqual(b);
  });
});

describe("integration: a real model learns addition, then subtraction", () => {
  // Yield to the event loop between step bursts so coverage-instrumented CI
  // runs don't starve the test worker's RPC channel.
  async function run(trainer: Trainer, steps: number): Promise<void> {
    for (let i = 0; i < steps; i++) {
      trainer.trainStep();
      if (i % 25 === 24) await new Promise((r) => setTimeout(r, 0));
    }
  }

  it("phase 1 learns 1-digit addition; phase 2 continues into subtraction", async () => {
    const rng = new Mulberry32(1337);
    const model = new CalcGPT(
      { vocabSize: tok.size, blockSize: 8, nLayer: 1, nHead: 2, nEmbd: 24 },
      rng,
    );
    const paramsBefore = model.parameters();
    const trainer = new Trainer({
      model,
      optimizer: new Adam(3e-3),
      generator: new AdditionGenerator(new Mulberry32(1), 9),
      batcher: new Batcher(tok, 8),
      tokenizer: tok,
      sink: new ArraySink(),
      batchSize: 24,
    });

    await run(trainer, 400);
    const addEval = makeEvalSet(new AdditionGenerator(new Mulberry32(99), 9), 40);
    expect(trainer.evaluate(addEval).accuracy).toBeGreaterThanOrEqual(0.8);

    // Phase 2: same model object, new data mix.
    trainer.setGenerator(
      new MixtureGenerator(new Mulberry32(2), [
        { generator: new AdditionGenerator(new Mulberry32(3), 9), weight: 1 },
        { generator: new SubtractionGenerator(new Mulberry32(4), 9), weight: 1 },
      ]),
    );
    await run(trainer, 400);
    expect(model.parameters()).toEqual(paramsBefore); // same tensors, continued

    const subEval = makeEvalSet(new SubtractionGenerator(new Mulberry32(98), 9), 40);
    expect(trainer.evaluate(subEval).accuracy).toBeGreaterThanOrEqual(0.4);
    expect(trainer.evaluate(addEval).accuracy).toBeGreaterThanOrEqual(0.6);
  }, 120_000);
});
