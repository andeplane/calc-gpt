import { describe, expect, it } from "vitest";
import { matmul, matmulATB, matmulABT } from "../src/engine/linalg";

const f32 = (xs: number[]) => Float32Array.from(xs);

describe("matmul", () => {
  it("computes a known 2x3 @ 3x2 product", () => {
    const a = f32([1, 2, 3, 4, 5, 6]); // 2x3
    const b = f32([7, 8, 9, 10, 11, 12]); // 3x2
    const out = new Float32Array(4);
    matmul(a, b, 2, 3, 2, out);
    expect(Array.from(out)).toEqual([58, 64, 139, 154]);
  });

  it("overwrites the output on repeated calls", () => {
    const a = f32([1, 0]); // 1x2 with a zero (exercises the skip branch)
    const b = f32([2, 3]); // 2x1
    const out = new Float32Array(1);
    matmul(a, b, 1, 2, 1, out);
    matmul(a, b, 1, 2, 1, out);
    expect(out[0]).toBe(2);
  });
});

describe("matmulATB", () => {
  it("accumulates aT @ b into the output", () => {
    const a = f32([1, 2, 3, 4]); // 2x2
    const b = f32([5, 6, 7, 8]); // 2x2
    const out = new Float32Array(4);
    matmulATB(a, b, 2, 2, 2, out);
    // aT@b = [[1,3],[2,4]] @ [[5,6],[7,8]] = [[26,30],[38,44]]
    expect(Array.from(out)).toEqual([26, 30, 38, 44]);
    matmulATB(a, b, 2, 2, 2, out);
    expect(Array.from(out)).toEqual([52, 60, 76, 88]); // += semantics
  });

  it("skips zero entries without changing the result", () => {
    const a = f32([0, 2]); // 1x2
    const b = f32([3]); // 1x1
    const out = new Float32Array(2);
    matmulATB(a, b, 1, 2, 1, out);
    expect(Array.from(out)).toEqual([0, 6]);
  });
});

describe("matmulABT", () => {
  it("accumulates a @ bT into the output", () => {
    const a = f32([1, 2, 3, 4]); // 2x2
    const b = f32([5, 6, 7, 8]); // 2x2 (interpreted as k x n = 2x2)
    const out = new Float32Array(4);
    matmulABT(a, b, 2, 2, 2, out);
    // a@bT = [[1*5+2*6, 1*7+2*8],[3*5+4*6, 3*7+4*8]] = [[17,23],[39,53]]
    expect(Array.from(out)).toEqual([17, 23, 39, 53]);
    matmulABT(a, b, 2, 2, 2, out);
    expect(Array.from(out)).toEqual([34, 46, 78, 106]); // += semantics
  });
});
