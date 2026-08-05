"""calc_gpt_micro: the calc-gpt engine, translated to numpy.

Line-for-line equivalent of the TypeScript engine at
https://github.com/andeplane/calc-gpt — same architecture (pre-LN transformer,
ReLU MLP, causal attention), same manual backward pass, same masked loss,
same Adam. Run this file to gradient-check the backward pass and watch the
model overfit a few problems:

    python calc_gpt_micro.py
"""

import numpy as np

VOCAB = "0123456789+-*/=;"
STOI = {ch: i for i, ch in enumerate(VOCAB)}
EOS = STOI[";"]


def encode(s):
    return np.array([STOI[ch] for ch in s], dtype=np.int64)


def decode(ids):
    return "".join(VOCAB[i] for i in ids)


# ----------------------------------------------------------------------------
# data

def make_addition_batch(rng, n, block_size, max_operand=99):
    x = np.full((n, block_size), EOS, dtype=np.int64)
    y = np.full((n, block_size), EOS, dtype=np.int64)
    mask = np.zeros((n, block_size), dtype=np.float64)
    for i in range(n):
        a, b = rng.integers(0, max_operand + 1), rng.integers(0, max_operand + 1)
        s = f"{a}+{b}={a + b};"
        toks = encode(s)
        prompt_len = s.index("=") + 1
        x[i, : len(toks) - 1] = toks[:-1]
        y[i, : len(toks) - 1] = toks[1:]
        mask[i, prompt_len - 1 : len(toks) - 1] = 1.0
    return x, y, mask


# ----------------------------------------------------------------------------
# model

def layernorm_forward(x, g, b):
    mean = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + 1e-5)
    xhat = (x - mean) * inv
    return xhat * g + b, (xhat, inv)


def layernorm_backward(dy, cache, g):
    xhat, inv = cache
    dg = (dy * xhat).sum(axis=tuple(range(dy.ndim - 1)))
    db = dy.sum(axis=tuple(range(dy.ndim - 1)))
    dxhat = dy * g
    dx = inv * (
        dxhat
        - dxhat.mean(-1, keepdims=True)
        - xhat * (dxhat * xhat).mean(-1, keepdims=True)
    )
    return dx, dg, db


class CalcGPT:
    def __init__(self, rng, vocab_size=16, block_size=12, n_layer=2, n_head=4, n_embd=48):
        self.n_layer, self.n_head, self.n_embd = n_layer, n_head, n_embd
        self.block_size, self.vocab_size = block_size, vocab_size
        init = lambda *shape: 0.02 * rng.standard_normal(shape)
        self.params = {"wte": init(vocab_size, n_embd), "wpe": init(block_size, n_embd)}
        for l in range(n_layer):
            self.params |= {
                f"l{l}.ln1g": np.ones(n_embd), f"l{l}.ln1b": np.zeros(n_embd),
                f"l{l}.wqkv": init(n_embd, 3 * n_embd), f"l{l}.bqkv": np.zeros(3 * n_embd),
                f"l{l}.wproj": init(n_embd, n_embd), f"l{l}.bproj": np.zeros(n_embd),
                f"l{l}.ln2g": np.ones(n_embd), f"l{l}.ln2b": np.zeros(n_embd),
                f"l{l}.w1": init(n_embd, 4 * n_embd), f"l{l}.b1": np.zeros(4 * n_embd),
                f"l{l}.w2": init(4 * n_embd, n_embd), f"l{l}.b2": np.zeros(n_embd),
            }
        self.params |= {"lnf.g": np.ones(n_embd), "lnf.b": np.zeros(n_embd),
                        "wout": init(n_embd, vocab_size)}
        self.grads = {k: np.zeros_like(v) for k, v in self.params.items()}

    def zero_grads(self):
        for g in self.grads.values():
            g.fill(0.0)

    def forward(self, x):
        p = self.params
        B, T = x.shape
        H, C = self.n_head, self.n_embd
        hs = C // H
        cache = {"x": x, "layers": []}
        h = p["wte"][x] + p["wpe"][:T]
        cache["h0"] = h
        causal = np.tril(np.ones((T, T), dtype=bool))
        for l in range(self.n_layer):
            lc = {"h_in": h}
            n1, lc["ln1"] = layernorm_forward(h, p[f"l{l}.ln1g"], p[f"l{l}.ln1b"])
            lc["n1"] = n1
            qkv = n1 @ p[f"l{l}.wqkv"] + p[f"l{l}.bqkv"]
            q, k, v = np.split(qkv, 3, axis=-1)
            # [B, H, T, hs]
            q = q.reshape(B, T, H, hs).transpose(0, 2, 1, 3)
            k = k.reshape(B, T, H, hs).transpose(0, 2, 1, 3)
            v = v.reshape(B, T, H, hs).transpose(0, 2, 1, 3)
            scores = q @ k.swapaxes(-1, -2) / np.sqrt(hs)
            scores = np.where(causal, scores, -np.inf)
            probs = np.exp(scores - scores.max(-1, keepdims=True))
            probs /= probs.sum(-1, keepdims=True)
            att = (probs @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
            lc |= {"q": q, "k": k, "v": v, "probs": probs, "att": att}
            h = h + att @ p[f"l{l}.wproj"] + p[f"l{l}.bproj"]
            lc["h_mid"] = h
            n2, lc["ln2"] = layernorm_forward(h, p[f"l{l}.ln2g"], p[f"l{l}.ln2b"])
            lc["n2"] = n2
            pre = n2 @ p[f"l{l}.w1"] + p[f"l{l}.b1"]
            act = np.maximum(pre, 0.0)
            lc |= {"pre": pre, "act": act}
            h = h + act @ p[f"l{l}.w2"] + p[f"l{l}.b2"]
            cache["layers"].append(lc)
        nf, cache["lnf"] = layernorm_forward(h, p["lnf.g"], p["lnf.b"])
        cache["h_last"], cache["nf"] = h, nf
        return nf @ p["wout"], cache

    def loss_backward(self, x, y, mask):
        """Masked cross-entropy + full backward pass. Returns the loss."""
        p, g = self.params, self.grads
        B, T = x.shape
        H, C = self.n_head, self.n_embd
        hs = C // H
        logits, cache = self.forward(x)

        z = logits - logits.max(-1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
        msum = mask.sum()
        loss = -(mask * np.take_along_axis(logp, y[..., None], -1)[..., 0]).sum() / msum

        dlogits = np.exp(logp)
        rows = np.arange(B)[:, None], np.arange(T)[None, :]
        onehot = np.zeros_like(dlogits)
        onehot[rows[0], rows[1], y] = 1.0
        dlogits = (dlogits - onehot) * (mask / msum)[..., None]

        g["wout"] += cache["nf"].reshape(-1, C).T @ dlogits.reshape(-1, self.vocab_size)
        dnf = dlogits @ p["wout"].T
        dh, dgf, dbf = layernorm_backward(dnf, cache["lnf"], p["lnf.g"])
        g["lnf.g"] += dgf
        g["lnf.b"] += dbf

        for l in reversed(range(self.n_layer)):
            lc = cache["layers"][l]
            # MLP
            g[f"l{l}.b2"] += dh.sum((0, 1))
            g[f"l{l}.w2"] += lc["act"].reshape(-1, 4 * C).T @ dh.reshape(-1, C)
            dact = dh @ p[f"l{l}.w2"].T
            dact[lc["pre"] <= 0] = 0.0
            g[f"l{l}.b1"] += dact.sum((0, 1))
            g[f"l{l}.w1"] += lc["n2"].reshape(-1, C).T @ dact.reshape(-1, 4 * C)
            dn2 = dact @ p[f"l{l}.w1"].T
            dmid, dg2, db2 = layernorm_backward(dn2, lc["ln2"], p[f"l{l}.ln2g"])
            g[f"l{l}.ln2g"] += dg2
            g[f"l{l}.ln2b"] += db2
            dmid = dmid + dh  # residual
            # attention
            g[f"l{l}.bproj"] += dmid.sum((0, 1))
            g[f"l{l}.wproj"] += lc["att"].reshape(-1, C).T @ dmid.reshape(-1, C)
            datt = (dmid @ p[f"l{l}.wproj"].T).reshape(B, T, H, hs).transpose(0, 2, 1, 3)
            dprobs = datt @ lc["v"].swapaxes(-1, -2)
            dv = lc["probs"].swapaxes(-1, -2) @ datt
            dscores = lc["probs"] * (dprobs - (dprobs * lc["probs"]).sum(-1, keepdims=True))
            dscores /= np.sqrt(hs)
            dq = dscores @ lc["k"]
            dk = dscores.swapaxes(-1, -2) @ lc["q"]
            dqkv = np.concatenate(
                [d.transpose(0, 2, 1, 3).reshape(B, T, C) for d in (dq, dk, dv)], axis=-1
            )
            g[f"l{l}.bqkv"] += dqkv.sum((0, 1))
            g[f"l{l}.wqkv"] += lc["n1"].reshape(-1, C).T @ dqkv.reshape(-1, 3 * C)
            dn1 = dqkv @ p[f"l{l}.wqkv"].T
            dh_in, dg1, db1 = layernorm_backward(dn1, lc["ln1"], p[f"l{l}.ln1g"])
            g[f"l{l}.ln1g"] += dg1
            g[f"l{l}.ln1b"] += db1
            dh = dh_in + dmid  # residual

        np.add.at(g["wte"], x, dh)
        g["wpe"][:T] += dh.sum(0)
        return loss

    def greedy_answer(self, prompt):
        ids = list(encode(prompt))
        out = []
        while len(ids) < self.block_size:
            logits, _ = self.forward(np.array([ids]))
            nxt = int(logits[0, -1].argmax())
            if nxt == EOS:
                break
            ids.append(nxt)
            out.append(nxt)
        return decode(out)


class Adam:
    def __init__(self, params, lr=2e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads):
        self.t += 1
        for k, w in self.params.items():
            gk = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * gk
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * gk * gk
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            w -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


# ----------------------------------------------------------------------------

def gradient_check():
    rng = np.random.default_rng(0)
    model = CalcGPT(rng, block_size=8, n_layer=1, n_head=2, n_embd=8)
    x, y, mask = make_addition_batch(rng, 2, 8, max_operand=9)
    model.zero_grads()
    model.loss_backward(x, y, mask)
    analytic_grads = {k: v.copy() for k, v in model.grads.items()}
    eps, worst = 1e-5, 0.0
    for name, w in model.params.items():
        flat = w.reshape(-1)
        for i in range(0, flat.size, max(1, flat.size // 5)):
            old = flat[i]
            flat[i] = old + eps
            up = model.loss_backward(x, y, mask)
            flat[i] = old - eps
            down = model.loss_backward(x, y, mask)
            flat[i] = old
            numeric = (up - down) / (2 * eps)
            analytic = analytic_grads[name].reshape(-1)[i]
            rel = abs(analytic - numeric) / max(1e-8, abs(analytic) + abs(numeric))
            worst = max(worst, rel)
    print(f"gradient check: worst relative error {worst:.2e} (float64, expect < 1e-5)")
    assert worst < 1e-5


def overfit_demo():
    rng = np.random.default_rng(1)
    model = CalcGPT(rng)
    opt = Adam(model.params, lr=1e-2)
    x, y, mask = make_addition_batch(rng, 4, model.block_size)
    for step in range(1, 301):
        model.zero_grads()
        loss = model.loss_backward(x, y, mask)
        opt.step(model.grads)
        if step % 50 == 0:
            print(f"step {step} | loss {loss:.4f}")
    prompt = decode(x[0][: list(y[0]).index(EOS) + 1]).split("=")[0] + "="
    print(f"{prompt}{model.greedy_answer(prompt)}")


if __name__ == "__main__":
    gradient_check()
    overfit_demo()
