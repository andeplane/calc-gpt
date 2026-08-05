import { matmul, matmulATB, matmulABT } from "./linalg";
import type { Rng } from "./rng";
import { randn } from "./rng";
import type { Tokenizer } from "./tokenizer";
import type { Batch } from "./data";

export interface ModelConfig {
  vocabSize: number;
  blockSize: number;
  nLayer: number;
  nHead: number;
  nEmbd: number;
}

/** A trainable tensor: values + gradient accumulator. */
export class Param {
  readonly data: Float32Array;
  readonly grad: Float32Array;

  constructor(
    readonly name: string,
    readonly rows: number,
    readonly cols: number,
    init?: (i: number) => number,
  ) {
    this.data = new Float32Array(rows * cols);
    this.grad = new Float32Array(rows * cols);
    if (init) for (let i = 0; i < this.data.length; i++) this.data[i] = init(i);
  }

  get size(): number {
    return this.data.length;
  }
}

interface LayerParams {
  ln1g: Param; ln1b: Param;
  wqkv: Param; bqkv: Param;
  wproj: Param; bproj: Param;
  ln2g: Param; ln2b: Param;
  w1: Param; b1: Param;
  w2: Param; b2: Param;
}

interface LayerCache {
  hIn: Float32Array;      // [BT×C] residual stream entering the layer
  n1: Float32Array;       // [BT×C] ln1 output
  x1hat: Float32Array;    // [BT×C] ln1 normalized (pre gamma/beta)
  inv1: Float32Array;     // [BT]   ln1 1/std
  qkv: Float32Array;      // [BT×3C]
  probs: Float32Array;    // [B×H×T×T] attention weights
  attOut: Float32Array;   // [BT×C] concatenated head outputs
  hMid: Float32Array;     // [BT×C] after attention residual
  n2: Float32Array;       // [BT×C] ln2 output
  x2hat: Float32Array;    // [BT×C]
  inv2: Float32Array;     // [BT]
  pre1: Float32Array;     // [BT×4C] mlp pre-activation
  act1: Float32Array;     // [BT×4C] relu(pre1)
}

interface ForwardCache {
  B: number;
  T: number;
  h0: Float32Array;
  layers: LayerCache[];
  hLast: Float32Array;
  nf: Float32Array;
  xfhat: Float32Array;
  invf: Float32Array;
  logits: Float32Array; // [BT×V]
}

/** ln forward: writes out = xhat*g + b, keeps xhat and 1/std for backward. */
function layerNormForward(
  x: Float32Array, g: Float32Array, b: Float32Array,
  rows: number, C: number,
  out: Float32Array, xhat: Float32Array, inv: Float32Array,
): void {
  for (let r = 0; r < rows; r++) {
    const off = r * C;
    let mean = 0;
    for (let c = 0; c < C; c++) mean += x[off + c];
    mean /= C;
    let variance = 0;
    for (let c = 0; c < C; c++) {
      const d = x[off + c] - mean;
      variance += d * d;
    }
    const invStd = 1 / Math.sqrt(variance / C + 1e-5);
    inv[r] = invStd;
    for (let c = 0; c < C; c++) {
      const xh = (x[off + c] - mean) * invStd;
      xhat[off + c] = xh;
      out[off + c] = xh * g[c] + b[c];
    }
  }
}

/** ln backward: adds dx into dxOut; accumulates dg, db. */
function layerNormBackward(
  dy: Float32Array, xhat: Float32Array, inv: Float32Array,
  g: Float32Array, rows: number, C: number,
  dxOut: Float32Array, dg: Float32Array, db: Float32Array,
): void {
  for (let r = 0; r < rows; r++) {
    const off = r * C;
    let meanDxhat = 0;
    let meanDxhatXhat = 0;
    for (let c = 0; c < C; c++) {
      const d = dy[off + c];
      const dxh = d * g[c];
      dg[c] += d * xhat[off + c];
      db[c] += d;
      meanDxhat += dxh;
      meanDxhatXhat += dxh * xhat[off + c];
    }
    meanDxhat /= C;
    meanDxhatXhat /= C;
    for (let c = 0; c < C; c++) {
      const dxh = dy[off + c] * g[c];
      dxOut[off + c] += inv[r] * (dxh - meanDxhat - xhat[off + c] * meanDxhatXhat);
    }
  }
}

/** A small GPT: token+position embeddings, N pre-norm transformer blocks
 * (causal self-attention + ReLU MLP), final layernorm, linear head.
 * Forward AND backward are written out by hand — no autograd anywhere. */
export class CalcGPT {
  readonly wte: Param;
  readonly wpe: Param;
  readonly layers: LayerParams[];
  readonly lnfg: Param;
  readonly lnfb: Param;
  readonly wout: Param;

  constructor(
    readonly cfg: ModelConfig,
    rng: Rng,
  ) {
    if (cfg.nEmbd % cfg.nHead !== 0) {
      throw new Error("nEmbd must be divisible by nHead");
    }
    const init = () => 0.02 * randn(rng);
    const zeros = () => 0;
    const ones = () => 1;
    const { vocabSize: V, blockSize: T, nEmbd: C, nLayer } = cfg;
    this.wte = new Param("wte", V, C, init);
    this.wpe = new Param("wpe", T, C, init);
    this.layers = [];
    for (let l = 0; l < nLayer; l++) {
      this.layers.push({
        ln1g: new Param(`l${l}.ln1g`, 1, C, ones),
        ln1b: new Param(`l${l}.ln1b`, 1, C, zeros),
        wqkv: new Param(`l${l}.wqkv`, C, 3 * C, init),
        bqkv: new Param(`l${l}.bqkv`, 1, 3 * C, zeros),
        wproj: new Param(`l${l}.wproj`, C, C, init),
        bproj: new Param(`l${l}.bproj`, 1, C, zeros),
        ln2g: new Param(`l${l}.ln2g`, 1, C, ones),
        ln2b: new Param(`l${l}.ln2b`, 1, C, zeros),
        w1: new Param(`l${l}.w1`, C, 4 * C, init),
        b1: new Param(`l${l}.b1`, 1, 4 * C, zeros),
        w2: new Param(`l${l}.w2`, 4 * C, C, init),
        b2: new Param(`l${l}.b2`, 1, C, zeros),
      });
    }
    this.lnfg = new Param("lnf.g", 1, C, ones);
    this.lnfb = new Param("lnf.b", 1, C, zeros);
    this.wout = new Param("wout", C, V, init);
  }

  parameters(): Param[] {
    const ps: Param[] = [this.wte, this.wpe];
    for (const l of this.layers) {
      ps.push(l.ln1g, l.ln1b, l.wqkv, l.bqkv, l.wproj, l.bproj,
              l.ln2g, l.ln2b, l.w1, l.b1, l.w2, l.b2);
    }
    ps.push(this.lnfg, this.lnfb, this.wout);
    return ps;
  }

  get numParams(): number {
    return this.parameters().reduce((s, p) => s + p.size, 0);
  }

  zeroGrads(): void {
    for (const p of this.parameters()) p.grad.fill(0);
  }

  /** Forward pass over B sequences of length T (T ≤ blockSize). */
  forward(x: Int32Array, B: number, T: number): ForwardCache {
    const { nEmbd: C, nHead: H, vocabSize: V } = this.cfg;
    if (T > this.cfg.blockSize) throw new Error("sequence exceeds block size");
    const hs = C / H;
    const scale = 1 / Math.sqrt(hs);
    const BT = B * T;

    const h0 = new Float32Array(BT * C);
    for (let bt = 0; bt < BT; bt++) {
      const tok = x[bt];
      const t = bt % T;
      for (let c = 0; c < C; c++) {
        h0[bt * C + c] = this.wte.data[tok * C + c] + this.wpe.data[t * C + c];
      }
    }

    const layers: LayerCache[] = [];
    let h = h0;
    for (const lp of this.layers) {
      const cache: LayerCache = {
        hIn: h,
        n1: new Float32Array(BT * C),
        x1hat: new Float32Array(BT * C),
        inv1: new Float32Array(BT),
        qkv: new Float32Array(BT * 3 * C),
        probs: new Float32Array(B * H * T * T),
        attOut: new Float32Array(BT * C),
        hMid: new Float32Array(BT * C),
        n2: new Float32Array(BT * C),
        x2hat: new Float32Array(BT * C),
        inv2: new Float32Array(BT),
        pre1: new Float32Array(BT * 4 * C),
        act1: new Float32Array(BT * 4 * C),
      };
      layerNormForward(h, lp.ln1g.data, lp.ln1b.data, BT, C, cache.n1, cache.x1hat, cache.inv1);
      matmul(cache.n1, lp.wqkv.data, BT, C, 3 * C, cache.qkv);
      for (let bt = 0; bt < BT; bt++) {
        for (let j = 0; j < 3 * C; j++) cache.qkv[bt * 3 * C + j] += lp.bqkv.data[j];
      }
      // causal attention, head by head
      for (let b = 0; b < B; b++) {
        for (let head = 0; head < H; head++) {
          const qOff = head * hs;
          const kOff = C + head * hs;
          const vOff = 2 * C + head * hs;
          for (let i = 0; i < T; i++) {
            const qi = (b * T + i) * 3 * C + qOff;
            const pRow = ((b * H + head) * T + i) * T;
            let maxScore = -Infinity;
            for (let j = 0; j <= i; j++) {
              const kj = (b * T + j) * 3 * C + kOff;
              let s = 0;
              for (let d = 0; d < hs; d++) s += cache.qkv[qi + d] * cache.qkv[kj + d];
              s *= scale;
              cache.probs[pRow + j] = s;
              if (s > maxScore) maxScore = s;
            }
            let sum = 0;
            for (let j = 0; j <= i; j++) {
              const e = Math.exp(cache.probs[pRow + j] - maxScore);
              cache.probs[pRow + j] = e;
              sum += e;
            }
            for (let j = 0; j <= i; j++) cache.probs[pRow + j] /= sum;
            const oOff = (b * T + i) * C + head * hs;
            for (let d = 0; d < hs; d++) {
              let acc = 0;
              for (let j = 0; j <= i; j++) {
                acc += cache.probs[pRow + j] * cache.qkv[(b * T + j) * 3 * C + vOff + d];
              }
              cache.attOut[oOff + d] = acc;
            }
          }
        }
      }
      // attention projection + residual
      const proj = new Float32Array(BT * C);
      matmul(cache.attOut, lp.wproj.data, BT, C, C, proj);
      for (let bt = 0; bt < BT; bt++) {
        for (let c = 0; c < C; c++) {
          cache.hMid[bt * C + c] = cache.hIn[bt * C + c] + proj[bt * C + c] + lp.bproj.data[c];
        }
      }
      // MLP + residual
      layerNormForward(cache.hMid, lp.ln2g.data, lp.ln2b.data, BT, C, cache.n2, cache.x2hat, cache.inv2);
      matmul(cache.n2, lp.w1.data, BT, C, 4 * C, cache.pre1);
      for (let bt = 0; bt < BT; bt++) {
        for (let j = 0; j < 4 * C; j++) {
          const v = cache.pre1[bt * 4 * C + j] + lp.b1.data[j];
          cache.pre1[bt * 4 * C + j] = v;
          cache.act1[bt * 4 * C + j] = v > 0 ? v : 0;
        }
      }
      const mlpOut = new Float32Array(BT * C);
      matmul(cache.act1, lp.w2.data, BT, 4 * C, C, mlpOut);
      const hOut = new Float32Array(BT * C);
      for (let bt = 0; bt < BT; bt++) {
        for (let c = 0; c < C; c++) {
          hOut[bt * C + c] = cache.hMid[bt * C + c] + mlpOut[bt * C + c] + lp.b2.data[c];
        }
      }
      layers.push(cache);
      h = hOut;
    }

    const nf = new Float32Array(BT * C);
    const xfhat = new Float32Array(BT * C);
    const invf = new Float32Array(BT);
    layerNormForward(h, this.lnfg.data, this.lnfb.data, BT, C, nf, xfhat, invf);
    const logits = new Float32Array(BT * V);
    matmul(nf, this.wout.data, BT, C, V, logits);

    return { B, T, h0, layers, hLast: h, nf, xfhat, invf, logits };
  }

  /** Masked cross-entropy loss + full backward pass (accumulates gradients). */
  lossBackward(batch: Batch): number {
    const { nEmbd: C, nHead: H, vocabSize: V } = this.cfg;
    const B = batch.batchSize;
    const T = batch.x.length / B;
    const BT = B * T;
    const hs = C / H;
    const scale = 1 / Math.sqrt(hs);

    const f = this.forward(batch.x, B, T);

    let maskSum = 0;
    for (let bt = 0; bt < BT; bt++) maskSum += batch.mask[bt];

    // softmax + CE, and dLogits = mask/maskSum * (p - onehot(y))
    const dLogits = new Float32Array(BT * V);
    let loss = 0;
    for (let bt = 0; bt < BT; bt++) {
      const m = batch.mask[bt];
      const off = bt * V;
      let maxLogit = -Infinity;
      for (let v = 0; v < V; v++) if (f.logits[off + v] > maxLogit) maxLogit = f.logits[off + v];
      let sum = 0;
      for (let v = 0; v < V; v++) sum += Math.exp(f.logits[off + v] - maxLogit);
      const logZ = maxLogit + Math.log(sum);
      const y = batch.y[bt];
      if (m > 0) {
        loss += m * (logZ - f.logits[off + y]);
        const w = m / maskSum;
        for (let v = 0; v < V; v++) {
          const p = Math.exp(f.logits[off + v] - logZ);
          dLogits[off + v] = w * (p - (v === y ? 1 : 0));
        }
      }
    }
    loss /= maskSum;

    // head
    matmulATB(f.nf, dLogits, BT, C, V, this.wout.grad);
    const dnf = new Float32Array(BT * C);
    matmulABT(dLogits, this.wout.data, BT, V, C, dnf);
    let dh = new Float32Array(BT * C);
    layerNormBackward(dnf, f.xfhat, f.invf, this.lnfg.data, BT, C, dh, this.lnfg.grad, this.lnfb.grad);

    for (let l = this.layers.length - 1; l >= 0; l--) {
      const lp = this.layers[l];
      const cache = f.layers[l];
      // ---- MLP backward: hOut = hMid + act1 @ w2 + b2
      const dhMid = new Float32Array(BT * C);
      dhMid.set(dh); // residual path
      for (let bt = 0; bt < BT; bt++) {
        for (let c = 0; c < C; c++) lp.b2.grad[c] += dh[bt * C + c];
      }
      matmulATB(cache.act1, dh, BT, 4 * C, C, lp.w2.grad);
      const dAct1 = new Float32Array(BT * 4 * C);
      matmulABT(dh, lp.w2.data, BT, C, 4 * C, dAct1);
      for (let i = 0; i < BT * 4 * C; i++) {
        if (cache.pre1[i] <= 0) dAct1[i] = 0; // relu gate
      }
      for (let bt = 0; bt < BT; bt++) {
        for (let j = 0; j < 4 * C; j++) lp.b1.grad[j] += dAct1[bt * 4 * C + j];
      }
      matmulATB(cache.n2, dAct1, BT, C, 4 * C, lp.w1.grad);
      const dn2 = new Float32Array(BT * C);
      matmulABT(dAct1, lp.w1.data, BT, 4 * C, C, dn2);
      layerNormBackward(dn2, cache.x2hat, cache.inv2, lp.ln2g.data, BT, C, dhMid, lp.ln2g.grad, lp.ln2b.grad);

      // ---- attention backward: hMid = hIn + attOut @ wproj + bproj
      const dhIn = new Float32Array(BT * C);
      dhIn.set(dhMid); // residual path
      for (let bt = 0; bt < BT; bt++) {
        for (let c = 0; c < C; c++) lp.bproj.grad[c] += dhMid[bt * C + c];
      }
      matmulATB(cache.attOut, dhMid, BT, C, C, lp.wproj.grad);
      const dAttOut = new Float32Array(BT * C);
      matmulABT(dhMid, lp.wproj.data, BT, C, C, dAttOut);

      const dQkv = new Float32Array(BT * 3 * C);
      for (let b = 0; b < B; b++) {
        for (let head = 0; head < H; head++) {
          const qOff = head * hs;
          const kOff = C + head * hs;
          const vOff = 2 * C + head * hs;
          for (let i = 0; i < T; i++) {
            const pRow = ((b * H + head) * T + i) * T;
            const doOff = (b * T + i) * C + head * hs;
            // dV and dP
            let dpDotP = 0;
            const dp = new Float32Array(i + 1);
            for (let j = 0; j <= i; j++) {
              const vj = (b * T + j) * 3 * C + vOff;
              let d = 0;
              for (let dd = 0; dd < hs; dd++) {
                d += dAttOut[doOff + dd] * cache.qkv[vj + dd];
                dQkv[vj + dd] += cache.probs[pRow + j] * dAttOut[doOff + dd];
              }
              dp[j] = d;
              dpDotP += d * cache.probs[pRow + j];
            }
            // softmax backward, then into q and k
            const qi = (b * T + i) * 3 * C + qOff;
            for (let j = 0; j <= i; j++) {
              const ds = cache.probs[pRow + j] * (dp[j] - dpDotP) * scale;
              const kj = (b * T + j) * 3 * C + kOff;
              for (let dd = 0; dd < hs; dd++) {
                dQkv[qi + dd] += ds * cache.qkv[kj + dd];
                dQkv[kj + dd] += ds * cache.qkv[qi + dd];
              }
            }
          }
        }
      }
      for (let bt = 0; bt < BT; bt++) {
        for (let j = 0; j < 3 * C; j++) lp.bqkv.grad[j] += dQkv[bt * 3 * C + j];
      }
      matmulATB(cache.n1, dQkv, BT, C, 3 * C, lp.wqkv.grad);
      const dn1 = new Float32Array(BT * C);
      matmulABT(dQkv, lp.wqkv.data, BT, 3 * C, C, dn1);
      layerNormBackward(dn1, cache.x1hat, cache.inv1, lp.ln1g.data, BT, C, dhIn, lp.ln1g.grad, lp.ln1b.grad);

      dh = dhIn;
    }

    // embeddings
    for (let bt = 0; bt < BT; bt++) {
      const tok = batch.x[bt];
      const t = bt % T;
      for (let c = 0; c < C; c++) {
        this.wte.grad[tok * C + c] += dh[bt * C + c];
        this.wpe.grad[t * C + c] += dh[bt * C + c];
      }
    }

    return loss;
  }

  /** Greedy decode: feed `prompt`, append argmax tokens until ';' or the
   * context fills up. Returns just the generated answer (without ';'). */
  greedyAnswer(tokenizer: Tokenizer, prompt: string): string {
    const eos = tokenizer.eosId;
    const ids: number[] = Array.from(tokenizer.encode(prompt));
    const out: number[] = [];
    while (ids.length < this.cfg.blockSize) {
      const x = Int32Array.from(ids);
      const f = this.forward(x, 1, ids.length);
      const off = (ids.length - 1) * this.cfg.vocabSize;
      let best = 0;
      for (let v = 1; v < this.cfg.vocabSize; v++) {
        if (f.logits[off + v] > f.logits[off + best]) best = v;
      }
      if (best === eos) break;
      ids.push(best);
      out.push(best);
    }
    return tokenizer.decode(out);
  }
}
