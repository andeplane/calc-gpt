# calc-gpt

An interactive blog post that trains a tiny GPT on the language of arithmetic —
from scratch, live, in your browser tab.

**Live site: https://andeplane.github.io/calc-gpt/**

- Phase 1: the model learns 2-digit **addition** (`23+45=68;`)
- Phase 2: the **same model** — same weights, same tokenizer — continues training
  with subtraction mixed in. The `-` token was reserved in the vocabulary from
  day one, so the curriculum switch is purely a data change.

## What's inside

- `src/engine/` — a dependency-injected training engine in plain TypeScript:
  char-level tokenizer, seeded RNG, problem generators, a 2-layer pre-norm
  transformer (~60k params) with a **hand-written backward pass** (no autograd,
  no ML libraries), Adam, and a trainer that composes it all.
- `tests/` — 62 tests with a **100% coverage gate** (statements, branches,
  functions, lines) on the engine. The centerpiece: numerical gradient checking
  of every parameter tensor against the analytic backward pass.
- `index.html` + `src/ui/` — the blog itself: KaTeX math, a tokenizer
  playground, live training in a web worker with loss/accuracy charts, an
  ask-the-model box, and the phase-2 curriculum button.
- `python/calc_gpt_micro.py` — the same model, line-for-line, in numpy.
  Run it to gradient-check and watch it overfit: `python python/calc_gpt_micro.py`

## Develop

```bash
npm install
npm run dev        # local dev server
npm run coverage   # tests + 100% coverage gate
npm run build      # typecheck + production build
```

Deploys to GitHub Pages via Actions on every push to `main` (tests must pass
first).

Inspired by [Karpathy's microgpt](https://karpathy.github.io/2026/02/12/microgpt/).
