import { describe, expect, it } from "vitest";
import { Mulberry32, randInt, randn, type Rng } from "../src/engine/rng";

describe("Mulberry32", () => {
  it("is deterministic for the same seed", () => {
    const a = new Mulberry32(42);
    const b = new Mulberry32(42);
    for (let i = 0; i < 100; i++) expect(a.next()).toBe(b.next());
  });

  it("differs across seeds", () => {
    const a = new Mulberry32(1);
    const b = new Mulberry32(2);
    const seqA = Array.from({ length: 10 }, () => a.next());
    const seqB = Array.from({ length: 10 }, () => b.next());
    expect(seqA).not.toEqual(seqB);
  });

  it("emits values in [0, 1)", () => {
    const rng = new Mulberry32(7);
    for (let i = 0; i < 1000; i++) {
      const v = rng.next();
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });
});

describe("randInt", () => {
  it("stays in [lo, hi) and hits both endpoints eventually", () => {
    const rng = new Mulberry32(3);
    const seen = new Set<number>();
    for (let i = 0; i < 2000; i++) {
      const v = randInt(rng, 0, 10);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(10);
      seen.add(v);
    }
    expect(seen.size).toBe(10);
  });
});

describe("randn", () => {
  it("has roughly standard-normal statistics", () => {
    const rng = new Mulberry32(11);
    const n = 5000;
    let sum = 0;
    let sumSq = 0;
    for (let i = 0; i < n; i++) {
      const v = randn(rng);
      expect(Number.isFinite(v)).toBe(true);
      sum += v;
      sumSq += v * v;
    }
    expect(Math.abs(sum / n)).toBeLessThan(0.1);
    expect(Math.abs(sumSq / n - 1)).toBeLessThan(0.1);
  });

  it("guards against log(0) when the rng emits exactly 0", () => {
    const zeroRng: Rng = { next: () => 0 };
    expect(Number.isFinite(randn(zeroRng))).toBe(true);
  });
});
