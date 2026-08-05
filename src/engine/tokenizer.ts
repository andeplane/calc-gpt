/** Character-level tokenizer over a fixed alphabet.
 *
 * The default vocabulary reserves slots for all four operators even though
 * phase 1 only trains on '+': adding subtraction later is then purely a data
 * change — the model's embedding table already has a row waiting for '-'. */
export const VOCAB = "0123456789+-*/=;";

export class Tokenizer {
  private readonly stoi = new Map<string, number>();

  constructor(readonly vocab: string = VOCAB) {
    for (let i = 0; i < vocab.length; i++) this.stoi.set(vocab[i], i);
  }

  get size(): number {
    return this.vocab.length;
  }

  /** Token id of ';' — used as both end-of-sequence and padding. */
  get eosId(): number {
    return this.idOf(";");
  }

  idOf(ch: string): number {
    const id = this.stoi.get(ch);
    if (id === undefined) throw new Error(`unknown character: ${JSON.stringify(ch)}`);
    return id;
  }

  encode(text: string): Int32Array {
    const out = new Int32Array(text.length);
    for (let i = 0; i < text.length; i++) out[i] = this.idOf(text[i]);
    return out;
  }

  decode(ids: ArrayLike<number>): string {
    let out = "";
    for (let i = 0; i < ids.length; i++) {
      const id = ids[i];
      if (id < 0 || id >= this.vocab.length) throw new Error(`invalid token id: ${id}`);
      out += this.vocab[id];
    }
    return out;
  }
}
