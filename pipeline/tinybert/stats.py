"""tinybert: measured numbers for facts.json, on the TinyStories validation split (never trained on).

    npm run atlas -- py pipeline/tinybert/stats.py        # writes pipeline/tinybert/stats.json (export.py reads it)

GPU, float32, TF32 off. Two measurements:

1. Training-style MLM (data.py's 80/10/10 masking, windows of 1..61 tokens, half of them full): accuracy and top-5 on
   every chosen position, and on the chosen positions that show [MASK] only.
2. One hidden token, as on the page: windows of 8..40 tokens of a validation story, one token c (1 <= c < n) hidden.
   - both sides:   TinyBERT reads [CLS] w0 .. w(c-1) [MASK] w(c+1) .. [SEP]          (the page's setting)
   - left only:    TinyBERT reads [CLS] w0 .. w(c-1) [MASK] [SEP]                    (nothing after the gap)
   - TinyGPT:      the shipped TinyGPT reads w0 .. w(c-1) and predicts the next token (its own task)
   Same windows, same hidden tokens; top-1 accuracy of each.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

import data
from model import MASK, TinyBERT

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def tinygpt():
    """The shipped TinyGPT (pipeline/tinygpt/runs/final_d76L4/final.pt, checked against static/models/tinygpt/weights.bin)."""
    spec = importlib.util.spec_from_file_location("tinygpt_model", ROOT / "pipeline" / "tinygpt" / "model.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tinygpt_model"] = mod  # dataclasses look the module up
    spec.loader.exec_module(mod)
    ck = torch.load(ROOT / "pipeline" / "tinygpt" / "runs" / "final_d76L4" / "final.pt", map_location="cpu")
    m = mod.GPT(mod.GPTConfig(**ck["cfg"]))
    m.load_state_dict(ck["model"])
    site = json.loads((ROOT / "site" / "static" / "models" / "tinygpt" / "spec.json").read_text(encoding="utf-8"))
    t = next(x for x in site["tensors"] if x["name"] == "tok_emb.weight")
    w = np.fromfile(ROOT / "site" / "static" / "models" / "tinygpt" / "weights.bin", dtype="<f4")[t["offset"]:t["offset"] + t["length"]]
    assert np.abs(w.reshape(t["shape"]) - m.tok_emb.weight.detach().numpy()).max() < 1e-6, "final.pt is not the shipped TinyGPT"
    return m.to(DEV).eval()


def pad_batch(rows: list[tuple[list[int], list[int]]]):
    T = max(len(r[0]) for r in rows)
    ids = np.zeros((len(rows), T), np.int64)
    seg = np.zeros((len(rows), T), np.int64)
    pad = np.ones((len(rows), T), bool)
    for b, (i, s) in enumerate(rows):
        ids[b, :len(i)], seg[b, :len(s)], pad[b, :len(i)] = i, s, False
    return [torch.from_numpy(x).to(DEV) for x in (ids, seg, pad)]


@torch.no_grad()
def mlm_eval(model, batches: int = 100, B: int = 512) -> dict:
    val = data.Batches("val")
    n = right = top5 = nm = right_m = 0
    loss = 0.0
    for i in range(batches):
        inputs, seg, pad, chosen, ids = [torch.from_numpy(x).to(DEV) for x in val.mlm(np.random.default_rng([4321, i]), B)]
        h = model.hidden(inputs, seg, pad)
        logits = h[chosen] @ model.tok.weight.t()
        tgt = ids[chosen]
        loss += torch.nn.functional.cross_entropy(logits, tgt, reduction="sum").item()
        hit = logits.argmax(-1) == tgt
        right += hit.sum().item()
        top5 += (logits.topk(5, -1).indices == tgt[:, None]).any(-1).sum().item()
        n += tgt.numel()
        shown_mask = inputs[chosen] == MASK
        nm += shown_mask.sum().item()
        right_m += (hit & shown_mask).sum().item()
    return {"valAcc": right / n, "valTop5": top5 / n, "valLoss": loss / n, "valN": n, "valAccMasked": right_m / nm, "valMaskedN": nm}


@torch.no_grad()
def gap_eval(model, gpt, N: int = 20000, B: int = 500) -> dict:
    val = data.Batches("val")
    rng = np.random.default_rng(2024)
    hit = {"both": 0, "left": 0, "gpt": 0}
    done = 0
    while done < N:
        win, n = val.windows(rng, B, 8, 40, 0.0)
        keep = n >= 8
        win, n = win[keep], n[keep]
        rows_both, rows_left, gpt_rows, at_both, at_left, cs, tgts = [], [], [], [], [], [], []
        for b in range(len(n)):
            if done + len(tgts) >= N:
                break
            content = [int(x) for x in win[b, :n[b]]]
            c = int(rng.integers(1, len(content)))
            i1, s1, w1 = data.layout(content, {c})
            i2, s2, w2 = data.layout(content[:c + 1], {c})
            rows_both.append((i1, s1)), rows_left.append((i2, s2)), gpt_rows.append(content)
            at_both.append(w1[c]), at_left.append(w2[c]), cs.append(c), tgts.append(content[c])
        tg = torch.tensor(tgts, device=DEV)
        ar = torch.arange(len(tgts), device=DEV)
        for key, rows, at in (("both", rows_both, at_both), ("left", rows_left, at_left)):
            ids, seg, pad = pad_batch(rows)
            pred = model(ids, seg, pad)[ar, torch.tensor(at, device=DEV)].argmax(-1)
            hit[key] += (pred == tg).sum().item()
        T = max(len(r) for r in gpt_rows)
        g = torch.zeros((len(gpt_rows), T), dtype=torch.long, device=DEV)
        for b, r in enumerate(gpt_rows):
            g[b, :len(r)] = torch.tensor(r, device=DEV)
        logits, _ = gpt(g)   # causal: row c-1 depends only on w0..w(c-1), so right padding changes nothing
        pred = logits[ar, torch.tensor(cs, device=DEV) - 1].argmax(-1)
        hit["gpt"] += (pred == tg).sum().item()
        done += len(tgts)
    return {"accBoth": hit["both"] / done, "accLeft": hit["left"] / done, "accGpt": hit["gpt"] / done, "cmpN": done}


def main():
    model = TinyBERT()
    model.load_state_dict(torch.load(HERE / "model_final.pt", map_location="cpu"))
    model = model.to(DEV).eval()
    out = {**mlm_eval(model), **gap_eval(model, tinygpt())}
    (HERE / "stats.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
