import { describe, expect, it } from "vitest";
import {
  AdditionGenerator,
  Batcher,
  MixtureGenerator,
  SubtractionGenerator,
  type Problem,
  type ProblemGenerator,
} from "../src/engine/data";
import { Mulberry32, type Rng } from "../src/engine/rng";
import { Tokenizer } from "../src/engine/tokenizer";

describe("AdditionGenerator", () => {
  it("produces well-formed, correct addition problems within range", () => {
    const gen = new AdditionGenerator(new Mulberry32(1), 99);
    for (let i = 0; i < 500; i++) {
      const { prompt, answer } = gen.sample();
      const m = prompt.match(/^(\d+)\+(\d+)=$/);
      expect(m).not.toBeNull();
      const [a, b] = [Number(m![1]), Number(m![2])];
      expect(a).toBeLessThanOrEqual(99);
      expect(b).toBeLessThanOrEqual(99);
      expect(Number(answer)).toBe(a + b);
    }
  });
});

describe("SubtractionGenerator", () => {
  it("produces correct subtraction problems, including negative answers", () => {
    const gen = new SubtractionGenerator(new Mulberry32(2), 99);
    let sawNegative = false;
    for (let i = 0; i < 500; i++) {
      const { prompt, answer } = gen.sample();
      const m = prompt.match(/^(\d+)-(\d+)=$/);
      expect(m).not.toBeNull();
      expect(Number(answer)).toBe(Number(m![1]) - Number(m![2]));
      if (answer.startsWith("-")) sawNegative = true;
    }
    expect(sawNegative).toBe(true);
  });
});

describe("MixtureGenerator", () => {
  const constGen = (p: Problem): ProblemGenerator => ({ sample: () => p });
  const add: Problem = { prompt: "1+1=", answer: "2" };
  const sub: Problem = { prompt: "1-1=", answer: "0" };

  it("throws on an empty mixture", () => {
    expect(() => new MixtureGenerator(new Mulberry32(1), [])).toThrow(/at least one/);
  });

  it("selects parts according to their weights", () => {
    const rig = (v: number): Rng => ({ next: () => v });
    const parts = [
      { generator: constGen(add), weight: 1 },
      { generator: constGen(sub), weight: 1 },
    ];
    expect(new MixtureGenerator(rig(0.0), parts).sample()).toEqual(add);
    expect(new MixtureGenerator(rig(0.75), parts).sample()).toEqual(sub);
  });

  it("roughly respects weights over many samples", () => {
    const gen = new MixtureGenerator(new Mulberry32(5), [
      { generator: constGen(add), weight: 3 },
      { generator: constGen(sub), weight: 1 },
    ]);
    let adds = 0;
    for (let i = 0; i < 1000; i++) if (gen.sample() === add) adds += 1;
    expect(adds / 1000).toBeGreaterThan(0.68);
    expect(adds / 1000).toBeLessThan(0.82);
  });

  it("falls back to the last part when weights sum to zero", () => {
    const gen = new MixtureGenerator(new Mulberry32(1), [
      { generator: constGen(sub), weight: 0 },
    ]);
    expect(gen.sample()).toEqual(sub);
  });
});

describe("Batcher", () => {
  const tok = new Tokenizer();
  const batcher = new Batcher(tok, 12);

  it("encodes a known example with the exact expected x, y and mask", () => {
    const { x, y, mask, batchSize } = batcher.encode([{ prompt: "2+2=", answer: "4" }]);
    expect(batchSize).toBe(1);
    // sequence: 2 + 2 = 4 ;  then EOS padding
    expect(Array.from(x)).toEqual([2, 10, 2, 14, 4, 15, 15, 15, 15, 15, 15, 15]);
    expect(Array.from(y)).toEqual([10, 2, 14, 4, 15, 15, 15, 15, 15, 15, 15, 15]);
    // graded positions: targets '4' (index 3) and ';' (index 4) only
    expect(Array.from(mask)).toEqual([0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]);
  });

  it("stacks multiple problems", () => {
    const batch = batcher.encode([
      { prompt: "1+1=", answer: "2" },
      { prompt: "99+99=", answer: "198" },
    ]);
    expect(batch.batchSize).toBe(2);
    expect(batch.x.length).toBe(24);
    // second row starts with '9'
    expect(batch.x[12]).toBe(9);
  });

  it("throws when a sequence exceeds the block size", () => {
    const tiny = new Batcher(tok, 4);
    expect(() => tiny.encode([{ prompt: "99+99=", answer: "198" }])).toThrow(/block size/);
  });
});
