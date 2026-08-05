import { describe, expect, it } from "vitest";
import { Tokenizer, VOCAB } from "../src/engine/tokenizer";

describe("Tokenizer", () => {
  const tok = new Tokenizer();

  it("has the 16-character calculator vocabulary by default", () => {
    expect(tok.size).toBe(16);
    expect(VOCAB).toBe("0123456789+-*/=;");
  });

  it("maps characters to their vocabulary index", () => {
    expect(tok.idOf("0")).toBe(0);
    expect(tok.idOf("9")).toBe(9);
    expect(tok.idOf("+")).toBe(10);
    expect(tok.idOf("-")).toBe(11);
    expect(tok.eosId).toBe(15);
  });

  it("round-trips encode/decode", () => {
    const s = "23+45=68;";
    expect(tok.decode(tok.encode(s))).toBe(s);
  });

  it("encodes to the expected ids", () => {
    expect(Array.from(tok.encode("2+2="))).toEqual([2, 10, 2, 14]);
  });

  it("throws on characters outside the vocabulary", () => {
    expect(() => tok.encode("2 + 2")).toThrow(/unknown character/);
  });

  it("throws on invalid token ids when decoding", () => {
    expect(() => tok.decode([0, 99])).toThrow(/invalid token id/);
    expect(() => tok.decode([-1])).toThrow(/invalid token id/);
  });

  it("supports a custom vocabulary", () => {
    const custom = new Tokenizer("ab");
    expect(custom.size).toBe(2);
    expect(Array.from(custom.encode("ba"))).toEqual([1, 0]);
  });
});
