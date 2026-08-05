import { describe, expect, it } from "vitest";
import { Param } from "../src/engine/model";
import { Adam, Sgd } from "../src/engine/optimizer";

function paramWithGrad(value: number, grad: number): Param {
  const p = new Param("p", 1, 1, () => value);
  p.grad[0] = grad;
  return p;
}

describe("Sgd", () => {
  it("applies p -= lr * grad exactly", () => {
    const p = paramWithGrad(1.0, 0.5);
    new Sgd(0.2).step([p]);
    expect(p.data[0]).toBeCloseTo(0.9, 6);
  });
});

describe("Adam", () => {
  it("moves by ~lr per step under a constant unit gradient (bias-corrected)", () => {
    // With g=1 every step: mHat = 1 and vHat = 1 exactly after bias
    // correction, so each update is lr / (1 + eps) ≈ lr.
    const p = paramWithGrad(1.0, 1.0);
    const opt = new Adam(0.1);
    opt.step([p]);
    expect(p.data[0]).toBeCloseTo(0.9, 5);
    p.grad[0] = 1.0;
    opt.step([p]);
    expect(p.data[0]).toBeCloseTo(0.8, 5);
  });

  it("reuses moment state across steps for the same param", () => {
    // After many steps with g=1 then one step with g=0, momentum keeps the
    // parameter moving — proving m/v persist per param.
    const p = paramWithGrad(0.0, 1.0);
    const opt = new Adam(0.1);
    for (let i = 0; i < 5; i++) {
      p.grad[0] = 1.0;
      opt.step([p]);
    }
    const before = p.data[0];
    p.grad[0] = 0.0;
    opt.step([p]);
    expect(p.data[0]).toBeLessThan(before); // still descending on momentum
  });

  it("respects a changed learning rate mid-training", () => {
    const p = paramWithGrad(1.0, 1.0);
    const opt = new Adam(0.1);
    opt.step([p]);
    const firstMove = 1.0 - p.data[0];
    opt.lr = 0.01;
    p.grad[0] = 1.0;
    const before = p.data[0];
    opt.step([p]);
    const secondMove = before - p.data[0];
    expect(secondMove).toBeLessThan(firstMove / 5);
  });
});
