"""Export a trained TinyGPT run to the site's model folder (site/static/models/tinygpt/).

Writes: spec.json, weights.bin (fp32), tokenizer.json, reference.json (parity data),
checkpoints/index.json + step_XXXXXX.bin (fp16) + curve.json, replay.json.

Run: python export.py --run final_d72L4
"""
from __future__ import annotations

import argparse
import base64
import datetime
import json
import math
import os
import shutil

import numpy as np
import torch

import bpe
from model import GPT, GPTConfig

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
DATA = os.path.join(HERE, "..", "data", "tinystories")
SITE = os.path.join(ROOT, "site", "static", "models", "tinygpt")
M32 = 0xFFFFFFFF

PROBES = [
    "Once upon a time, there was a little",
    "The cat sat on the",
    "Lily wanted to play with her",
    "One day, Tom went to the",
    "The sun was",
    "She was very",
]
REFERENCE_PROMPTS = [
    "Once upon a time, there was a little",
    "The cat sat on the mat. 我爱 🐶!",
    # a longer one (about 40 tokens)
    "Lily and Ben went to the park. They saw a big red ball under a tree. Ben said, \"Can we play with it?\" Lily smiled and said, \"Yes!\"",
]
SAMPLE_PROMPTS = [
    "Once upon a time, there was a little",
    "One day, a girl named Mia found a",
    "Tom and his dog went to the",
    "The little bird was sad because",
    "Once upon a time, in a big forest,",
]
SAMPLED = {"temperature": 0.8, "topK": 0, "topP": 1.0, "seed": 1234}


# ------------------------------------------------------------------------------------------------
# sampling, identical to site/src/lib/models/tinygpt.ts
# ------------------------------------------------------------------------------------------------

def mulberry32(seed: int):
    a = seed & M32

    def imul(x, y):
        return (x * y) & M32

    def nxt():
        nonlocal a
        a = (a + 0x6D2B79F5) & M32
        t = imul(a ^ (a >> 15), 1 | a)
        t = ((t + imul(t ^ (t >> 7), 61 | t)) & M32) ^ t
        return ((t ^ (t >> 14)) & M32) / 4294967296.0

    return nxt


def sampling_distribution(logits, temperature, top_k=0, top_p=1.0):
    V = len(logits)
    z = [float(v) / temperature for v in logits]
    m = max(z)
    p = [math.exp(v - m) for v in z]
    s = 0.0
    for v in p:
        s += v
    p = [v / s for v in p]
    if (0 < top_k < V) or top_p < 1:
        idx = sorted(range(V), key=lambda i: (-p[i], i))
        keep = top_k if 0 < top_k < V else V
        if top_p < 1:
            cum, n = 0.0, 0
            while n < keep:
                cum += p[idx[n]]
                n += 1
                if cum >= top_p:
                    break
            keep = max(1, n)
        for r in range(keep, V):
            p[idx[r]] = 0.0
        zz = 0.0
        for v in p:
            zz += v
        p = [v / zz for v in p]
    return p


@torch.no_grad()
def last_logits(model, ids):
    x = torch.tensor([ids[-model.cfg.context:]], dtype=torch.long)
    logits, _ = model(x)
    return logits[0, -1].float().numpy()


@torch.no_grad()
def generate(model, ids, max_new, temperature, top_k=0, top_p=1.0, seed=42, eot=None, stop_at_eot=True):
    rand = mulberry32(seed)
    out = list(ids)
    for _ in range(max_new):
        lg = last_logits(model, out)
        if temperature <= 0:
            tok = int(np.argmax(lg))
        else:
            p = sampling_distribution(lg, temperature, top_k, top_p)
            u = rand()
            tok, cum = -1, 0.0
            for i, v in enumerate(p):
                if v == 0:
                    continue
                cum += v
                tok = i
                if cum > u:
                    break
        out.append(tok)
        if stop_at_eot and tok == eot:
            break
    return out[len(ids):]


# ------------------------------------------------------------------------------------------------

def load_model(path, fp16_round=False):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ck["cfg"])
    model = GPT(cfg)
    sd = ck["model"]
    if fp16_round:
        sd = {k: v.half().float() for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    return model, ck


def b64f32(t) -> dict:
    a = np.ascontiguousarray(np.asarray(t, dtype="<f4"))
    return {"shape": list(a.shape), "dtype": "float32", "b64": base64.b64encode(a.tobytes()).decode("ascii")}


def tensor_table(named):
    table, off = [], 0
    for n, t in named:
        table.append({"name": n, "shape": list(t.shape), "offset": off, "length": int(t.numel())})
        off += int(t.numel())
    return table, off


def write_flat(named, path, dtype):
    flat = np.concatenate([t.numpy().reshape(-1) for _, t in named]).astype(dtype)
    flat.astype("<f4" if dtype == np.float32 else "<f2").tofile(path)
    return flat


@torch.no_grad()
def heldout_loss(model, seqs):
    x = torch.tensor(seqs, dtype=torch.long)
    logits, loss = model(x[:, :-1], x[:, 1:])
    return float(loss)


@torch.no_grad()
def full_val_loss(model, data, T, device="cuda"):
    """Mean loss over all non-overlapping T-token windows of `data` (each token predicted once)."""
    model = model.to(device)
    n = (len(data) - 1) // T
    x = torch.from_numpy(data[: n * T].astype(np.int64)).view(n, T)
    y = torch.from_numpy(data[1: n * T + 1].astype(np.int64)).view(n, T)
    tot, cnt = 0.0, 0
    for i in range(0, n, 1024):
        xb, yb = x[i:i + 1024].to(device), y[i:i + 1024].to(device)
        logits, _ = model(xb)
        l = torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.size(-1)), yb.reshape(-1), reduction="sum")
        tot += float(l)
        cnt += yb.numel()
    model.to("cpu")
    return tot / cnt, cnt


def layer_specs(cfg: GPTConfig):
    D, H = cfg.d_model, cfg.n_head
    T = "T"
    L = []

    def lay(id_, op, en, zh, ins, shape, params=None):
        d = {"id": id_, "op": op, "label": {"en": en, "zh": zh}, "in": ins, "outShape": shape}
        if params:
            d["params"] = params
        L.append(d)

    lay("tok_emb", "embedding", "Token embedding", "词元嵌入", ["input"], [T, D], {"weight": "tok_emb.weight"})
    lay("pos_emb", "position_embedding", "Position embedding", "位置嵌入", ["input"], [T, D], {"weight": "pos_emb.weight"})
    lay("embed", "add", "Token + position", "词元 + 位置", ["tok_emb", "pos_emb"], [T, D])
    prev = "embed"
    for i in range(cfg.n_layer):
        p, n = f"L{i}.", i + 1
        en, zh = f"Block {n} · ", f"第 {n} 块 · "
        lay(p + "ln1", "layernorm", en + "LayerNorm 1", zh + "层归一化 1", [prev], [T, D],
            {"weight": p + "ln1.weight", "bias": p + "ln1.bias", "eps": cfg.ln_eps})
        for x, xen, xzh in [("q", "Query Q", "查询 Q"), ("k", "Key K", "键 K"), ("v", "Value V", "值 V")]:
            lay(p + "attn." + x, "linear", en + xen, zh + xzh, [p + "ln1"], [T, D],
                {"weight": p + f"attn.{x}.weight", "bias": p + f"attn.{x}.bias", "nHead": H, "dHead": D // H})
        lay(p + "attn.scores", "attention_scores", en + "Scores QKᵀ/√d", zh + "注意力分数 QKᵀ/√d",
            [p + "attn.q", p + "attn.k"], [H, T, T], {"nHead": H, "dHead": D // H, "scale": 1 / math.sqrt(D // H)})
        lay(p + "attn.masked", "causal_mask", en + "Causal mask", zh + "因果掩码", [p + "attn.scores"], [H, T, T],
            {"fill": "-Infinity above the diagonal"})
        lay(p + "attn.weights", "softmax", en + "Attention weights", zh + "注意力权重", [p + "attn.masked"], [H, T, T],
            {"axis": -1})
        lay(p + "attn.heads", "attention_mix", en + "Weighted values (heads joined)", zh + "值的加权和（多头拼接）",
            [p + "attn.weights", p + "attn.v"], [T, D], {"nHead": H, "dHead": D // H})
        lay(p + "attn.proj", "linear", en + "Attention output", zh + "注意力输出投影", [p + "attn.heads"], [T, D],
            {"weight": p + "attn.proj.weight", "bias": p + "attn.proj.bias"})
        lay(p + "resid1", "add", en + "Residual add 1", zh + "残差相加 1", [prev, p + "attn.proj"], [T, D])
        lay(p + "ln2", "layernorm", en + "LayerNorm 2", zh + "层归一化 2", [p + "resid1"], [T, D],
            {"weight": p + "ln2.weight", "bias": p + "ln2.bias", "eps": cfg.ln_eps})
        lay(p + "mlp.fc", "linear", en + "MLP expand (4×)", zh + "MLP 扩展（4 倍）", [p + "ln2"], [T, 4 * D],
            {"weight": p + "mlp.fc.weight", "bias": p + "mlp.fc.bias"})
        lay(p + "mlp.gelu", "gelu_tanh", en + "GELU", zh + "GELU 激活", [p + "mlp.fc"], [T, 4 * D])
        lay(p + "mlp.proj", "linear", en + "MLP project back", zh + "MLP 投影回原维度", [p + "mlp.gelu"], [T, D],
            {"weight": p + "mlp.proj.weight", "bias": p + "mlp.proj.bias"})
        lay(p + "resid2", "add", en + "Residual add 2", zh + "残差相加 2", [p + "resid1", p + "mlp.proj"], [T, D])
        prev = p + "resid2"
    lay("ln_f", "layernorm", "Final LayerNorm", "最终层归一化", [prev], [T, D],
        {"weight": "ln_f.weight", "bias": "ln_f.bias", "eps": cfg.ln_eps})
    lay("logits", "linear_tied", "Logits (tied to token embedding)", "Logits（与词元嵌入共享权重）", ["ln_f"],
        [T, cfg.vocab_size], {"weight": "tok_emb.weight", "transposed": True})
    lay("probs", "softmax", "Next-token probabilities", "下一个词元的概率", ["logits"], [T, cfg.vocab_size], {"axis": -1})
    return L


CKPT_LABELS = {0: ("random weights", "随机初始权重")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = os.path.join(HERE, "runs", args.run)
    os.makedirs(os.path.join(SITE, "checkpoints"), exist_ok=True)
    tok = bpe.BPETokenizer.load(os.path.join(HERE, "out", "tokenizer.json"))
    eot = tok.special["<|endoftext|>"]
    with open(os.path.join(run, "curve.json")) as f:
        curve_run = json.load(f)
    with open(os.path.join(HERE, "out", "data_stats.json")) as f:
        stats = json.load(f)
    targs = curve_run["args"]

    model, ck = load_model(os.path.join(run, "final.pt"))
    cfg = model.cfg
    named = model.export_tensors()
    table, total = tensor_table(named)
    write_flat(named, os.path.join(SITE, "weights.bin"), np.float32)
    shutil.copy(os.path.join(HERE, "out", "tokenizer.json"), os.path.join(SITE, "tokenizer.json"))
    print(f"weights.bin: {total} floats, {total * 4} bytes")

    # --- honest final metrics over the whole validation split and a same-size slice of train ---
    val = np.fromfile(os.path.join(DATA, "val.bin"), dtype=np.uint16)
    train = np.fromfile(os.path.join(DATA, "train.bin"), dtype=np.uint16)
    vl, vn = full_val_loss(model, val, cfg.context)
    tl, tn = full_val_loss(model, train[: len(val)], cfg.context)
    # bits per UTF-8 byte: mean loss per token scaled to all val tokens (incl. <|endoftext|>) over all val bytes
    bpb = vl * len(val) / math.log(2) / stats["val_bytes"]
    metrics = {
        "val_loss": round(vl, 4), "val_perplexity": round(math.exp(vl), 3), "val_tokens_scored": vn,
        "train_loss": round(tl, 4), "train_perplexity": round(math.exp(tl), 3), "train_tokens_scored": tn,
        "val_bits_per_byte": round(bpb, 4),
        "loss_unit": "nats per token (cross-entropy), mean over non-overlapping 64-token windows",
        "uniform_baseline_loss": round(math.log(cfg.vocab_size), 4),
    }
    print("metrics", metrics)

    # --- 5 sample generations ---
    samples = []
    for i, pr in enumerate(SAMPLE_PROMPTS):
        ids = tok.encode(pr)
        new = generate(model, ids, 120, 0.8, seed=100 + i, eot=eot)
        samples.append({"prompt": pr, "temperature": 0.8, "seed": 100 + i, "maxNew": 120,
                        "continuation": tok.decode([t for t in new if t != eot]), "endedWithEot": bool(new and new[-1] == eot)})

    # --- reference.json (parity data) ---
    ref = {"format": "Each tensor is {shape, dtype: float32, b64: base64 of little-endian float32}. "
                     "-Infinity in attn.masked is stored as is (it is a float32 value).",
           "model": "weights.bin (fp32)", "prompts": []}
    for pr in REFERENCE_PROMPTS:
        ids = tok.encode(pr)
        order, tr = model.trace(ids)
        ref["prompts"].append({"text": pr, "ids": ids, "order": order,
                               "tensors": {k: b64f32(tr[k].numpy()) for k in order}})
    long_ids = tok.encode(" ".join(PROBES) + " " + REFERENCE_PROMPTS[2] + " " + PROBES[0])
    assert len(long_ids) > cfg.context, len(long_ids)
    ref["truncation"] = {"ids": long_ids, "context": cfg.context, "keptFrom": len(long_ids) - cfg.context,
                         "lastLogits": b64f32(last_logits(model, long_ids))}
    r = mulberry32(42)
    ref["prng"] = {"seed": 42, "values": [r() for _ in range(10)]}
    sp = tok.encode(PROBES[0])
    ref["sampling"] = []
    for sset in [dict(temperature=0.8, top_k=0, top_p=1.0, seed=7), dict(temperature=1.0, top_k=20, top_p=1.0, seed=8),
                 dict(temperature=1.2, top_k=0, top_p=0.9, seed=9), dict(temperature=0.0, top_k=0, top_p=1.0, seed=1)]:
        new = generate(model, sp, 40, sset["temperature"], sset["top_k"], sset["top_p"], sset["seed"], eot=eot)
        ref["sampling"].append({"ids": sp, "maxNew": 40, "temperature": sset["temperature"], "topK": sset["top_k"],
                                "topP": sset["top_p"], "seed": sset["seed"], "newIds": new})

    # --- checkpoints (fp16) + replay ---
    replay_files = sorted(f for f in os.listdir(os.path.join(run, "replay")) if f.endswith(".pt"))
    curve_by_step = {c["step"]: c for c in curve_run["curve"]}
    ck_index = {"dtype": "float16", "tensors": table, "checkpoints": []}
    rng = np.random.default_rng(0)
    # 200 held-out sequences: the first 65 tokens of 200 validation stories, evenly spaced
    starts = [0] + [int(i) + 1 for i in np.nonzero(val == eot)[0][:-1]]
    good = [s for s in starts if s + 65 <= len(val) and eot not in val[s:s + 65]]
    pick = [good[int(i)] for i in np.linspace(0, len(good) - 1, 200)]
    seqs = [val[s:s + 65].astype(np.int64).tolist() for s in pick]
    probe_ids = [tok.encode(p) for p in PROBES]
    replay = {
        "probes": PROBES, "probeIds": probe_ids,
        "settings": {"topN": 10, "greedy": {"maxNew": 24, "temperature": 0, "stopAtEot": True},
                     "sampled": {"maxNew": 24, **SAMPLED, "stopAtEot": True,
                                 "note": "same mulberry32 PRNG and sampling steps as tinygpt.ts generate()"},
                     "heldOut": "200 validation stories (evenly spaced), first 65 tokens each: mean loss of the "
                                "64 next-token predictions, in nats per token",
                     "weights": "every number here was computed from the float16 checkpoint files (as shipped)"},
        "checkpoints": [],
    }
    ref_ckpt_done = False
    for fn in replay_files:
        step = int(fn.split("_")[1].split(".")[0])
        m16, _ = load_model(os.path.join(run, "replay", fn), fp16_round=True)
        m32, _ = load_model(os.path.join(run, "replay", fn), fp16_round=False)
        nm = m16.export_tensors()
        out_fn = f"step_{step:06d}.bin"
        write_flat(nm, os.path.join(SITE, "checkpoints", out_fn), np.float16)
        cc = curve_by_step.get(step, {})
        lab = CKPT_LABELS.get(step)
        info = {"step": step, "file": out_fn, "tokens": step * targs["batch"] * targs["context"],
                "trainLoss": round(cc.get("train_loss", float("nan")), 4), "valLoss": round(cc.get("val_loss", float("nan")), 4)}
        if step == targs["steps"]:
            lab = ("final", "最终")
        if lab:
            info["label"] = {"en": lab[0], "zh": lab[1]}
        ck_index["checkpoints"].append(info)
        hl = heldout_loss(m16, seqs)
        entry = {**info, "heldOut": {"loss": round(hl, 4), "perplexity": round(math.exp(hl), 3), "n": len(seqs)}, "probes": []}
        for ids in probe_ids:
            lg = last_logits(m16, ids)
            p = torch.softmax(torch.from_numpy(lg).double(), -1).numpy()
            top = sorted(range(len(p)), key=lambda i: (-p[i], i))[:10]
            g = generate(m16, ids, 24, 0.0, eot=eot)
            s = generate(m16, ids, 24, SAMPLED["temperature"], SAMPLED["topK"], SAMPLED["topP"], SAMPLED["seed"], eot=eot)
            entry["probes"].append({
                "top": [{"id": int(i), "text": tok.token_to_display(int(i)), "p": round(float(p[i]), 6)} for i in top],
                "greedy": {"ids": g, "text": tok.decode(g)},
                "sampled": {"ids": s, "text": tok.decode(s)},
            })
        replay["checkpoints"].append(entry)
        print(f"checkpoint {step}: heldOut {hl:.4f}; greedy: {entry['probes'][0]['greedy']['text']!r}")
        if not ref_ckpt_done and step >= 1000:
            ids = tok.encode(REFERENCE_PROMPTS[0])
            _, t16 = m16.trace(ids)
            _, t32 = m32.trace(ids)
            ref["checkpoint"] = {"step": step, "file": out_fn, "ids": ids,
                                 "logitsFp16Weights": b64f32(t16["logits"].numpy()),
                                 "logitsFp32Weights": b64f32(t32["logits"].numpy()),
                                 "note": "logitsFp16Weights: PyTorch fp32 math on the fp16-rounded weights (what the "
                                         "site loads); logitsFp32Weights: the original fp32 checkpoint."}
            ref_ckpt_done = True

    with open(os.path.join(SITE, "checkpoints", "index.json"), "w") as f:
        json.dump(ck_index, f)
    with open(os.path.join(SITE, "checkpoints", "curve.json"), "w") as f:
        json.dump({"unit": "nats per token", "batchTokens": targs["batch"] * targs["context"],
                   "evalNote": f"train/val loss on {targs['eval_batches']} fixed batches of {targs['batch']} random "
                               f"64-token windows each (train split / official validation split)",
                   "points": [{"step": c["step"], "tokens": c["tokens"], "train": round(c["train_loss"], 4),
                               "val": round(c["val_loss"], 4), "lr": c["lr"]} for c in curve_run["curve"]]}, f)
    with open(os.path.join(SITE, "replay.json"), "w", encoding="utf-8") as f:
        json.dump(replay, f, ensure_ascii=False)
    with open(os.path.join(SITE, "reference.json"), "w", encoding="utf-8") as f:
        json.dump(ref, f, ensure_ascii=False)

    # --- spec.json ---
    old_parity = None
    sp_path = os.path.join(SITE, "spec.json")
    if os.path.exists(sp_path):
        with open(sp_path, encoding="utf-8") as f:
            old_parity = json.load(f).get("parity")
    spec = {
        "slug": "tinygpt",
        "name": {"en": "TinyGPT", "zh": "TinyGPT（微型 GPT）"},
        "version": datetime.date.today().isoformat(),
        "params": int(total),
        "precision": "fp32",
        "input": {"kind": "text", "shape": ["T"],
                  "notes": f"Token ids from tokenizer.json (byte-level BPE, {tok.vocab_size} tokens). At most "
                           f"{cfg.context} tokens: longer input keeps the last {cfg.context} (forward) and the cut is reported."},
        "vocabSize": cfg.vocab_size, "context": cfg.context, "nLayer": cfg.n_layer, "nHead": cfg.n_head,
        "dModel": cfg.d_model, "dHead": cfg.d_model // cfg.n_head, "dMLP": 4 * cfg.d_model, "lnEps": cfg.ln_eps,
        "eotId": eot, "tokenizer": "tokenizer.json", "tiedHead": True, "activation": "gelu_tanh",
        "tensors": table,
        "layers": layer_specs(cfg),
        "training": {
            "dataset": "TinyStories (Eldan & Li 2023, CDLA-Sharing-1.0): first 100 MiB of TinyStoriesV2-GPT4-train; "
                       "validation = official TinyStoriesV2-GPT4-valid",
            "trainTokens": stats["train_tokens"], "valTokens": stats["val_tokens"],
            "bytesPerToken": round(stats["val_bytes_per_token"], 3),
            "steps": targs["steps"], "batch": targs["batch"], "tokensSeen": targs["steps"] * targs["batch"] * targs["context"],
            "epochs": round(targs["steps"] * targs["batch"] * targs["context"] / stats["train_tokens"], 2),
            "optimizer": f"AdamW (betas 0.9/0.95, weight decay {targs['wd']}), lr {targs['lr']}, "
                         f"{targs['warmup']} warm-up steps then cosine to {targs['min_lr_frac']}x, grad clip 1.0, bf16 autocast",
            "hardware": "RTX 5090 Laptop GPU",
            "metrics": metrics,
            "samples": samples,
        },
        "parity": old_parity or {"against": "pytorch-fp32", "max_abs_diff": None, "checked": None},
        "files": {"weights": "weights.bin", "tokenizer": "tokenizer.json", "reference": "reference.json",
                  "checkpoints": "checkpoints/index.json", "curve": "checkpoints/curve.json", "replay": "replay.json"},
    }
    with open(sp_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=1)
    print("samples:")
    for s in samples:
        print(" ", repr(s["prompt"] + s["continuation"]))
    for fn in sorted(os.listdir(SITE)):
        p = os.path.join(SITE, fn)
        if os.path.isfile(p):
            print(f"{fn}: {os.path.getsize(p)} bytes")


if __name__ == "__main__":
    main()
