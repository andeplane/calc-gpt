/** Injectable randomness. Everything that needs random numbers takes an Rng,
 * so tests can pass a seeded (or fake) source and stay deterministic. */
export interface Rng {
  /** Uniform float in [0, 1). */
  next(): number;
}

/** Small, fast, seedable PRNG (mulberry32). */
export class Mulberry32 implements Rng {
  private state: number;

  constructor(seed: number) {
    this.state = seed >>> 0;
  }

  next(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let t = this.state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
}

/** Uniform integer in [lo, hi). */
export function randInt(rng: Rng, lo: number, hi: number): number {
  return lo + Math.floor(rng.next() * (hi - lo));
}

/** Standard normal via Box–Muller. */
export function randn(rng: Rng): number {
  let u = rng.next();
  if (u === 0) u = 1e-12; // log(0) guard
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng.next());
}
