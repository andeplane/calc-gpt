import { describe, expect, it } from "vitest";
import { Batcher } from "../src/engine/data";
import { CalcGPT, Param, type ModelConfig } from "../src/engine/model";
import { Adam } from "../src/engine/optimizer";
import { Mulberry32 } from "../src/engine/rng";
import { Tokenizer } from "../src/engine/tokenizer";

const tok = new Tokenizer();

const tinyCfg: ModelConfig = {
  vocabSize: tok.size,
  blockSize: 8,
  nLayer: 1,
  nHead: 2,
  nEmbd: 8,
};

function tinyBatch(blockSize = 8) {
  const batcher = new Batcher(tok, blockSize);
  return batcher.encode([
    { prompt: "1+2=", answer: "3" },
    { prompt: "9+9=", answer: "18" },
  ]);
}

describe("Param", () => {
  it("starts at zero without an initializer", () => {
    const p = new Param("p", 2, 3);
    expect(p.size).toBe(6);
    expect(Array.from(p.data)).toEqual([0, 0, 0, 0, 0, 0]);
  });
});

describe("CalcGPT construction", () => {
  it("rejects nEmbd not divisible by nHead", () => {
    expect(() => new CalcGPT({ ...tinyCfg, nEmbd: 9 }, new Mulberry32(1))).toThrow(
      /divisible/,
    );
  });

  it("counts parameters across all tensors", () => {
    const model = new CalcGPT(tinyCfg, new Mulberry32(1));
    expect(model.parameters().length).toBe(2 + 12 * tinyCfg.nLayer + 3);
    const sum = model.parameters().reduce((s, p) => s + p.size, 0);
    expect(model.numParams).toBe(sum);
  });

  it("rejects sequences longer than the block size", () => {
    const model = new CalcGPT(tinyCfg, new Mulberry32(1));
    const x = new Int32Array(9);
    expect(() => model.forward(x, 1, 9)).toThrow(/block size/);
  });
});

describe("forward & loss", () => {
  it("produces logits of the right shape", () => {
    const model = new CalcGPT(tinyCfg, new Mulberry32(1));
    const f = model.forward(Int32Array.from([1, 10, 2, 14]), 1, 4);
    expect(f.logits.length).toBe(4 * tok.size);
    expect(Array.from(f.logits).every(Number.isFinite)).toBe(true);
  });

  it("starts near the uniform-distribution loss ln(V)", () => {
    const model = new CalcGPT(tinyCfg, new Mulberry32(1));
    const loss = model.lossBackward(tinyBatch());
    expect(loss).toBeGreaterThan(Math.log(16) - 0.4);
    expect(loss).toBeLessThan(Math.log(16) + 0.4);
  });

  it("zeroGrads clears accumulated gradients", () => {
    const model = new CalcGPT(tinyCfg, new Mulberry32(1));
    model.lossBackward(tinyBatch());
    const anyNonZero = model.parameters().some((p) => p.grad.some((g) => g !== 0));
    expect(anyNonZero).toBe(true);
    model.zeroGrads();
    const allZero = model.parameters().every((p) => p.grad.every((g) => g === 0));
    expect(allZero).toBe(true);
  });
});

describe("analytical gradients match numerical gradients", () => {
  // The most important test in the repo: perturb weights one at a time and
  // compare (loss(w+e) - loss(w-e)) / 2e with the hand-derived backward pass.
  const model = new CalcGPT(tinyCfg, new Mulberry32(1234));
  const batch = tinyBatch();
  model.zeroGrads();
  model.lossBackward(batch);
  const analytic = model.parameters().map((p) => Float32Array.from(p.grad));
  const eps = 1e-3;

  it.each(model.parameters().map((p, idx) => [p.name, idx] as const))(
    "gradient check: %s",
    (_name, idx) => {
      const p = model.parameters()[idx];
      const nSamples = Math.min(6, p.size);
      let checkedNonZero = 0;
      for (let s = 0; s < nSamples; s++) {
        const i = Math.floor((s * p.size) / nSamples);
        const old = p.data[i];
        p.data[i] = old + eps;
        const up = model.lossBackward(batch);
        p.data[i] = old - eps;
        const down = model.lossBackward(batch);
        p.data[i] = old;
        const numeric = (up - down) / (2 * eps);
        const a = analytic[idx][i];
        expect(Math.abs(a - numeric)).toBeLessThanOrEqual(
          2e-3 + 0.1 * Math.max(Math.abs(a), Math.abs(numeric)),
        );
        if (Math.abs(a) > 1e-6) checkedNonZero += 1;
      }
      expect(checkedNonZero).toBeGreaterThan(0);
    },
  );
});

describe("optimization end-to-end", () => {
  it("overfits a fixed micro-batch and answers it greedily", () => {
    const cfg: ModelConfig = { ...tinyCfg, nEmbd: 32, blockSize: 12 };
    const model = new CalcGPT(cfg, new Mulberry32(7));
    const batcher = new Batcher(tok, cfg.blockSize);
    const problems = [
      { prompt: "12+34=", answer: "46" },
      { prompt: "7+8=", answer: "15" },
      { prompt: "90+9=", answer: "99" },
    ];
    const batch = batcher.encode(problems);
    const opt = new Adam(0.01);
    let loss = Infinity;
    for (let i = 0; i < 400; i++) {
      model.zeroGrads();
      loss = model.lossBackward(batch);
      opt.step(model.parameters());
    }
    expect(loss).toBeLessThan(0.05);
    for (const { prompt, answer } of problems) {
      expect(model.greedyAnswer(tok, prompt)).toBe(answer);
    }
  });

  it("greedy decode stops when the context window fills up", () => {
    const model = new CalcGPT(tinyCfg, new Mulberry32(1));
    // Prompt of blockSize-1 tokens leaves room for exactly one generated token.
    const answer = model.greedyAnswer(tok, "1+1=1+1");
    expect(answer.length).toBeLessThanOrEqual(1);
  });
});
