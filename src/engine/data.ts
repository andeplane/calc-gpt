import type { Rng } from "./rng";
import { randInt } from "./rng";
import type { Tokenizer } from "./tokenizer";

/** One training example: the model reads `prompt` and must produce `answer;`. */
export interface Problem {
  prompt: string; // e.g. "23+45="
  answer: string; // e.g. "68"
}

/** Injectable data source — swapping the generator is how curriculum phases
 * work: the model object never changes, only what it gets to read. */
export interface ProblemGenerator {
  sample(): Problem;
}

export class AdditionGenerator implements ProblemGenerator {
  constructor(
    private readonly rng: Rng,
    private readonly maxOperand = 99,
  ) {}

  sample(): Problem {
    const a = randInt(this.rng, 0, this.maxOperand + 1);
    const b = randInt(this.rng, 0, this.maxOperand + 1);
    return { prompt: `${a}+${b}=`, answer: `${a + b}` };
  }
}

export class SubtractionGenerator implements ProblemGenerator {
  constructor(
    private readonly rng: Rng,
    private readonly maxOperand = 99,
  ) {}

  sample(): Problem {
    const a = randInt(this.rng, 0, this.maxOperand + 1);
    const b = randInt(this.rng, 0, this.maxOperand + 1);
    return { prompt: `${a}-${b}=`, answer: `${a - b}` };
  }
}

/** Weighted mixture of generators (e.g. 50% addition + 50% subtraction). */
export class MixtureGenerator implements ProblemGenerator {
  private readonly totalWeight: number;

  constructor(
    private readonly rng: Rng,
    private readonly parts: { generator: ProblemGenerator; weight: number }[],
  ) {
    if (parts.length === 0) throw new Error("MixtureGenerator needs at least one part");
    this.totalWeight = parts.reduce((s, p) => s + p.weight, 0);
  }

  sample(): Problem {
    let r = this.rng.next() * this.totalWeight;
    for (const part of this.parts) {
      r -= part.weight;
      if (r < 0) return part.generator.sample();
    }
    return this.parts[this.parts.length - 1].generator.sample(); // r == totalWeight edge
  }
}

/** A tokenized training batch. All sequences are padded with EOS to blockSize.
 * `mask` is 1 only where the target token is part of the answer (incl. the
 * terminating ';') — the model is graded on the answer, not on predicting
 * the random question. */
export interface Batch {
  x: Int32Array; // [batch * blockSize] inputs
  y: Int32Array; // [batch * blockSize] next-token targets
  mask: Float32Array; // [batch * blockSize] loss mask
  batchSize: number;
}

export class Batcher {
  constructor(
    private readonly tokenizer: Tokenizer,
    readonly blockSize: number,
  ) {}

  encode(problems: Problem[]): Batch {
    const B = problems.length;
    const T = this.blockSize;
    const eos = this.tokenizer.eosId;
    const x = new Int32Array(B * T).fill(eos);
    const y = new Int32Array(B * T).fill(eos);
    const mask = new Float32Array(B * T);
    for (let b = 0; b < B; b++) {
      const { prompt, answer } = problems[b];
      const full = this.tokenizer.encode(prompt + answer + ";");
      if (full.length > T + 1) {
        throw new Error(`sequence longer than block size: ${prompt}${answer};`);
      }
      for (let t = 0; t < full.length - 1; t++) x[b * T + t] = full[t];
      for (let t = 0; t < full.length - 1; t++) y[b * T + t] = full[t + 1];
      // targets y[t] = full[t+1]; grade positions whose target lies after '='
      for (let t = prompt.length - 1; t < full.length - 1; t++) mask[b * T + t] = 1;
    }
    return { x, y, mask, batchSize: B };
  }
}
