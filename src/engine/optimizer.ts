import type { Param } from "./model";

/** Injectable update rule: given params with fresh gradients, update in place. */
export interface Optimizer {
  lr: number;
  step(params: Param[]): void;
}

export class Sgd implements Optimizer {
  constructor(public lr: number) {}

  step(params: Param[]): void {
    for (const p of params) {
      for (let i = 0; i < p.data.length; i++) p.data[i] -= this.lr * p.grad[i];
    }
  }
}

export class Adam implements Optimizer {
  private readonly m = new Map<Param, Float32Array>();
  private readonly v = new Map<Param, Float32Array>();
  private t = 0;

  constructor(
    public lr: number,
    private readonly beta1 = 0.9,
    private readonly beta2 = 0.999,
    private readonly eps = 1e-8,
  ) {}

  step(params: Param[]): void {
    this.t += 1;
    const bc1 = 1 - Math.pow(this.beta1, this.t);
    const bc2 = 1 - Math.pow(this.beta2, this.t);
    for (const p of params) {
      let m = this.m.get(p);
      let v = this.v.get(p);
      if (!m || !v) {
        m = new Float32Array(p.data.length);
        v = new Float32Array(p.data.length);
        this.m.set(p, m);
        this.v.set(p, v);
      }
      for (let i = 0; i < p.data.length; i++) {
        const g = p.grad[i];
        m[i] = this.beta1 * m[i] + (1 - this.beta1) * g;
        v[i] = this.beta2 * v[i] + (1 - this.beta2) * g * g;
        const mHat = m[i] / bc1;
        const vHat = v[i] / bc2;
        p.data[i] -= (this.lr * mHat) / (Math.sqrt(vHat) + this.eps);
      }
    }
  }
}
