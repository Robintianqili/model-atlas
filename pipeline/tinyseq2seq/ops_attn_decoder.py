"""Python mirror of op 'attn_decoder' (site/src/lib/explainers/tinyseq2seq/ops/attn_decoder.ts): a GRU decoder with
Bahdanau (additive) attention, decoding greedily for `steps` steps from the encoder states h [T, H]:
    s = h_{T-1}, y = bos;  key_j = W_k h_j + b_k (once);  for each step:
        query = W_q s;  e_j = v . tanh(query + key_j);  a = softmax(e);  c = sum_j a_j h_j
        s = GRUCell([emb(y); c], s);  logits = W_o [s; c] + b_o;  y = argmax(logits)
Output: logits [steps, V]. Sub-steps keys [T, A], query [steps, A], scores and weights [steps, T], context and state
[steps, H], next [steps] (the chosen ids as floats). export.py checks it against model.GreedyProbs (AK103).
"""
import torch

from ops_gru import cell

OP = "attn_decoder"
VERSION = 1
CATEGORY = "attention"
LABEL = ("Decoder with attention", "带注意力的解码器")
ATTRS = {"steps": {"type": "int", "required": True}, "bos": {"type": "int", "default": 0}}
ALIASES: dict = {}
WEIGHTS = [{"slot": s} for s in ("emb", "wq", "wk", "bk", "v", "weight_ih", "weight_hh", "bias_ih", "bias_hh", "wo", "bo")]
SUBS = ["keys", "query", "scores", "weights", "context", "state", "next"]


def reference(xs, params, w):
    enc = xs[0]
    steps, y = int(params["steps"]), int(params.get("bos", 0))
    keys = enc @ w["wk"].T + w["bk"]
    s = enc[-1]
    rows = {k: [] for k in SUBS[1:] + ["logits"]}
    for _ in range(steps):
        q = w["wq"] @ s
        e = torch.tanh(q[None, :] + keys) @ w["v"].reshape(-1)
        a = torch.softmax(e, dim=0)
        c = a @ enc
        _, _, _, s = cell(torch.cat([w["emb"][y], c]), s, w["weight_ih"], w["weight_hh"], w["bias_ih"], w["bias_hh"])
        logits = w["wo"] @ torch.cat([s, c]) + w["bo"]
        y = int(torch.argmax(logits))
        for k, v in zip(rows, (q, e, a, c, s, torch.tensor(float(y), dtype=enc.dtype), logits)):
            rows[k].append(v)
    out = {k: torch.stack(v) for k, v in rows.items()}
    return out.pop("logits"), {"keys": keys, **out}
