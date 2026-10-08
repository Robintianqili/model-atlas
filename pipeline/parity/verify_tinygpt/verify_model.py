"""Verification of the shipped TinyGPT assets, written independently of export.py.

1. weights.bin + spec.json -> PyTorch GPT; compare with runs/final_d76L4/final.pt; tensor table checks
2. validation / train-slice loss recomputed from the exported fp32 weights (fp32 and fp64)
3. fresh prompts -> PyTorch traces (fp32 and fp64) for the TS forward check
4. greedy and sampled continuations with an independent Python loop (for the TS generate check)
5. surprisal reference on a long text (same sliding windows as tinygpt.ts)
6. fp16 checkpoints: decode, compare with the training replay .pt files, step 0 vs a fresh init,
   near-uniform check, held-out loss recomputed, replay.json / curve.json / index.json completeness
7. file sizes against the contract, licenses.json

Writes pipeline/parity/verify_tinygpt/verify_ref.json for verify_model.check.ts.
Run: python pipeline/parity/verify_tinygpt/verify_model.py
"""
import base64
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "pipeline", "tinygpt"))
import bpe  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

SITE = os.path.join(ROOT, "site", "static", "models", "tinygpt")
RUN = os.path.join(ROOT, "pipeline", "tinygpt", "runs", "final_d76L4")
DATA = os.path.join(ROOT, "pipeline", "data", "tinystories")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
problems = []


def check(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        problems.append(msg)


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


spec = jload(os.path.join(SITE, "spec.json"))
tok = bpe.BPETokenizer.load(os.path.join(SITE, "tokenizer.json"))
EOT = spec["eotId"]
cfg = GPTConfig(vocab_size=spec["vocabSize"], context=spec["context"], n_layer=spec["nLayer"], n_head=spec["nHead"],
                d_model=spec["dModel"], ln_eps=spec["lnEps"])
D, L, C, V = cfg.d_model, cfg.n_layer, cfg.context, cfg.vocab_size

# ------------------------------------------------------------------------------------------------
print("1) tensor table, weights.bin, final.pt")
table = spec["tensors"]
off = 0
contiguous = True
for e in table:
    if e["offset"] != off or int(np.prod(e["shape"])) != e["length"]:
        contiguous = False
    off += e["length"]
wbytes = os.path.getsize(os.path.join(SITE, "weights.bin"))
check(contiguous, f"spec.tensors contiguous, shape products = lengths ({len(table)} tensors)")
check(off == spec["params"] == wbytes // 4 and wbytes % 4 == 0, f"sum of lengths {off} = params {spec['params']} = weights.bin {wbytes} / 4")
idx = jload(os.path.join(SITE, "checkpoints", "index.json"))
check(idx["tensors"] == table, "checkpoints/index.json tensor table identical to spec.tensors")
n_expected = 2 + L * 16 + 2
check(len(table) == n_expected and len({e['name'] for e in table}) == n_expected, f"{n_expected} distinct tensor names")


def model_from_flat(flat):
    """Build a GPT from a flat float array laid out like spec.tensors (mapping written here, not export.py)."""
    W = {e["name"]: torch.from_numpy(np.asarray(flat[e["offset"]:e["offset"] + e["length"]], dtype=np.float32).reshape(e["shape"]).copy()) for e in table}
    m = GPT(cfg)
    sd = {"tok_emb.weight": W["tok_emb.weight"], "pos_emb.weight": W["pos_emb.weight"],
          "ln_f.weight": W["ln_f.weight"], "ln_f.bias": W["ln_f.bias"]}
    for i in range(L):
        p, b = f"L{i}.", f"blocks.{i}."
        sd[b + "ln1.weight"], sd[b + "ln1.bias"] = W[p + "ln1.weight"], W[p + "ln1.bias"]
        sd[b + "ln2.weight"], sd[b + "ln2.bias"] = W[p + "ln2.weight"], W[p + "ln2.bias"]
        sd[b + "qkv.weight"] = torch.cat([W[p + "attn.q.weight"], W[p + "attn.k.weight"], W[p + "attn.v.weight"]], 0)
        sd[b + "qkv.bias"] = torch.cat([W[p + "attn.q.bias"], W[p + "attn.k.bias"], W[p + "attn.v.bias"]], 0)
        sd[b + "attn_proj.weight"], sd[b + "attn_proj.bias"] = W[p + "attn.proj.weight"], W[p + "attn.proj.bias"]
        sd[b + "fc.weight"], sd[b + "fc.bias"] = W[p + "mlp.fc.weight"], W[p + "mlp.fc.bias"]
        sd[b + "mlp_proj.weight"], sd[b + "mlp_proj.bias"] = W[p + "mlp.proj.weight"], W[p + "mlp.proj.bias"]
    m.load_state_dict(sd, strict=True)
    m.eval()
    return m


flat32 = np.fromfile(os.path.join(SITE, "weights.bin"), dtype="<f4")
model = model_from_flat(flat32)
ck = torch.load(os.path.join(RUN, "final.pt"), map_location="cpu", weights_only=False)
d_final = max(float((model.state_dict()[k].float() - v.float()).abs().max()) for k, v in ck["model"].items())
check(d_final == 0.0, f"weights.bin == runs/final_d76L4/final.pt exactly (max |diff| {d_final})")
check(ck["step"] == 30000, f"final.pt step {ck['step']}")
nonfinite = int((~np.isfinite(flat32)).sum())
check(nonfinite == 0, f"weights.bin all finite ({nonfinite} non-finite)")
print(f"  weights: min {flat32.min():.4f} max {flat32.max():.4f} mean|w| {np.abs(flat32).mean():.4f}")

# ------------------------------------------------------------------------------------------------
print("2) loss recomputed from weights.bin (all non-overlapping 64-token windows)")
val = np.fromfile(os.path.join(DATA, "val.bin"), dtype=np.uint16)
train = np.fromfile(os.path.join(DATA, "train.bin"), dtype=np.uint16)
check(len(val) == spec["training"]["valTokens"] and len(train) == spec["training"]["trainTokens"],
      f"val.bin {len(val)} tokens, train.bin {len(train)} tokens match spec.training")
stories_val = int((val == EOT).sum())
stories_train = int((train == EOT).sum())
print(f"  stories (EOT count): val {stories_val}, train {stories_train}")


@torch.no_grad()
def window_loss(m, data, dtype, dev="cuda"):
    m = m.to(dev).to(dtype)
    n = (len(data) - 1) // C
    x = torch.from_numpy(data[: n * C].astype(np.int64)).view(n, C)
    y = torch.from_numpy(data[1: n * C + 1].astype(np.int64)).view(n, C)
    tot = torch.zeros((), dtype=torch.float64, device=dev)
    for i in range(0, n, 2048):
        lg, _ = m(x[i:i + 2048].to(dev))
        tot += F.cross_entropy(lg.double().reshape(-1, V), y[i:i + 2048].to(dev).reshape(-1), reduction="sum")
    m.to("cpu").float()
    return float(tot) / (n * C), n * C


vl32, vn = window_loss(model, val, torch.float32)
vl64, _ = window_loss(model, val, torch.float64)
tl32, tn = window_loss(model, train[: len(val)], torch.float32)
met = spec["training"]["metrics"]
print(f"  val loss fp32 {vl32:.5f}  fp64 {vl64:.5f}  (claimed {met['val_loss']}), tokens {vn} (claimed {met['val_tokens_scored']})")
print(f"  train-slice loss fp32 {tl32:.5f} (claimed {met['train_loss']}); ppl val {math.exp(vl64):.3f} train {math.exp(tl32):.3f}")
check(abs(vl64 - met["val_loss"]) < 6e-4 and vn == met["val_tokens_scored"], "val loss matches claim within rounding")
check(abs(tl32 - met["train_loss"]) < 6e-4, "train loss matches claim within rounding")
stats = jload(os.path.join(ROOT, "pipeline", "tinygpt", "out", "data_stats.json"))
bpb = vl64 * len(val) / math.log(2) / stats["val_bytes"]
check(abs(bpb - met["val_bits_per_byte"]) < 6e-4, f"bits per byte {bpb:.4f} (claimed {met['val_bits_per_byte']})")
# an alternative view: loss by position inside the 64-token window
n_alt = 2000
starts = np.linspace(0, len(val) - C - 2, n_alt).astype(np.int64)
with torch.no_grad():
    xw = torch.from_numpy(np.stack([val[s:s + C] for s in starts]).astype(np.int64))
    yw = torch.from_numpy(np.stack([val[s + 1:s + C + 1] for s in starts]).astype(np.int64))
    lg, _ = model(xw)
    per = F.cross_entropy(lg.reshape(-1, V), yw.reshape(-1), reduction="none").view(n_alt, C)
print(f"  loss by context position (2000 val windows): pos0 {per[:, 0].mean():.3f}, pos1-7 {per[:, 1:8].mean():.3f}, "
      f"pos32-63 {per[:, 32:].mean():.3f}")

# ------------------------------------------------------------------------------------------------
print("3) fresh-prompt traces")


def b64(a):
    a = np.ascontiguousarray(np.asarray(a, dtype="<f4"))
    return base64.b64encode(a.tobytes()).decode("ascii")


story = ("Pip the small brown mouse lived under the old red barn. Every morning he ran to the big field to look for "
         "seeds, and every evening he ran back home before the owl woke up. One windy day the barn door blew shut and "
         "Pip could not get in! He sat on a stone and thought hard. \"I can dig,\" he said.")
fresh = [
    {"name": "T1-eot", "ids": [EOT]},
    {"name": "T1-word", "ids": tok.encode(" dog")[:1]},
    {"name": "chinese-emoji", "ids": tok.encode("我爱自然语言处理 " + chr(0x1F916) + " and the robot said hello.")},
    {"name": "eot-inside", "ids": tok.encode("The end.<|endoftext|>Once upon a time, a tiny dragon named Zed")},
    {"name": "whitespace", "ids": tok.encode("  \n\n\t   Lily   said   \"hi\"   \n")},
    {"name": "T64", "ids": tok.encode(story)[:64]},
    {"name": "long-truncated", "ids": tok.encode(story + " " + story)},
]
model64 = model_from_flat(flat32).double()
ref = {"prompts": [], "generate": [], "surprisal": None}
worst32v64 = 0.0
for fp in fresh:
    ids = fp["ids"]
    kept = ids[-C:]
    order, t = model.trace(kept)
    _, t64 = model64.trace(kept)
    for k in order:
        a, b = t[k].double(), t64[k]
        fin = torch.isfinite(b)
        worst32v64 = max(worst32v64, float((a[fin] - b[fin]).abs().max()))
    ref["prompts"].append({"name": fp["name"], "ids": ids, "kept": len(kept), "order": order,
                           "t32": {k: {"shape": list(t[k].shape), "b64": b64(t[k].numpy())} for k in order},
                           "t64": {k: b64(t64[k].numpy()) for k in order}})
    print(f"  {fp['name']}: {len(ids)} ids, traced {len(kept)}, {len(order)} tensors")
print(f"  PyTorch fp32 vs fp64 trace, worst over all fresh tensors: {worst32v64:.2e}")
check(ref["prompts"][0]["order"] == [l["id"] for l in spec["layers"]], "trace order == spec.layers ids (66)")

# ------------------------------------------------------------------------------------------------
print("4) continuations (independent Python loop)")
M32 = 0xFFFFFFFF


def mulberry32(seed):
    st = [seed & M32]

    def nxt():
        st[0] = (st[0] + 0x6D2B79F5) & M32
        a = st[0]
        t = ((a ^ (a >> 15)) * (1 | a)) & M32
        t = (((t + (((t ^ (t >> 7)) * (61 | t)) & M32)) & M32) ^ t) & M32
        return ((t ^ (t >> 14)) & M32) / 4294967296.0
    return nxt


def sample_dist(lg, temp, top_k, top_p):
    z = lg.astype(np.float64) / temp
    p = np.exp(z - z.max())
    p = p / p.sum()
    if (0 < top_k < V) or top_p < 1:
        order = sorted(range(V), key=lambda i: (-p[i], i))
        keep = top_k if 0 < top_k < V else V
        if top_p < 1:
            cum, n = 0.0, 0
            while n < keep:
                cum += p[order[n]]
                n += 1
                if cum >= top_p:
                    break
            keep = max(1, n)
        q = np.zeros(V)
        for i in order[:keep]:
            q[i] = p[i]
        p = q / q.sum()
    return p


@torch.no_grad()
def continue_ids(m, ids, max_new, temp, top_k=0, top_p=1.0, seed=42, stop=True):
    rnd = mulberry32(seed)
    out = list(ids)
    margins = []
    for _ in range(max_new):
        lg = m(torch.tensor([out[-C:]]))[0][0, -1].numpy()
        s = np.sort(lg)[::-1]
        margins.append(float(s[0] - s[1]))
        if temp <= 0:
            nt = int(np.argmax(lg))
        else:
            p = sample_dist(lg, temp, top_k, top_p)
            u = rnd()
            cum, nt = 0.0, -1
            for i in range(V):
                if p[i] == 0:
                    continue
                cum += p[i]
                nt = i
                if cum > u:
                    break
        out.append(nt)
        if stop and nt == EOT:
            break
    return out[len(ids):], margins


gen_prompts = ["Once upon a time, there was a big", "The dog and the cat", "Mom said, \"", "Ben was sad because",
               "我爱自然语言处理", "One day, Lily found a strange box in the garden. She", story, "<|endoftext|>"]
settings = [dict(temp=0, seed=1), dict(temp=0.8, seed=11), dict(temp=1.0, top_k=40, seed=12), dict(temp=1.1, top_p=0.9, seed=13)]
min_margin = 1e9
for pr in gen_prompts:
    ids = tok.encode(pr)
    for sset in settings:
        new, marg = continue_ids(model, ids, 60, sset["temp"], sset.get("top_k", 0), sset.get("top_p", 1.0), sset["seed"], stop=False)
        if sset["temp"] == 0:
            min_margin = min(min_margin, min(marg))
        ref["generate"].append({"prompt": pr, "ids": ids, "maxNew": 60, "temperature": sset["temp"], "topK": sset.get("top_k", 0),
                                "topP": sset.get("top_p", 1.0), "seed": sset["seed"], "stopAtEot": False, "newIds": new})
    print(f"  greedy {pr[:30]!r} -> {tok.decode(ref['generate'][-4]['newIds'])[:90]!r}")
print(f"  smallest top-1/top-2 logit margin seen in greedy runs: {min_margin:.2e}")

# ------------------------------------------------------------------------------------------------
print("5) surprisal reference (stride 32 windows)")
sids = tok.encode(story + " " + story + " The end.")
stride = 32
bits = [None] * len(sids)
ranks = [None] * len(sids)
scored_upto, start = 1, 0
with torch.no_grad():
    while scored_upto < len(sids):
        win = sids[start:start + C]
        lp = torch.log_softmax(model(torch.tensor([win]))[0][0].double(), -1)
        end = start + len(win)
        for pos in range(scored_upto, end):
            row = lp[pos - 1 - start]
            bits[pos] = float(-row[sids[pos]] / math.log(2))
            ranks[pos] = int((row > row[sids[pos]]).sum()) + 1
        scored_upto = end
        if end >= len(sids):
            break
        start += stride
ref["surprisal"] = {"ids": sids, "bits": bits, "ranks": ranks, "stride": stride}
print(f"  {len(sids)} tokens, mean bits {np.mean([b for b in bits if b is not None]):.4f}")

# ------------------------------------------------------------------------------------------------
print("6) fp16 checkpoints, replay, curve")
cks = idx["checkpoints"]
steps = [c["step"] for c in cks]
check(steps == [0, 10, 30, 100, 200, 400, 1000, 3000, 10000, 30000], f"checkpoint steps {steps}")
replay = jload(os.path.join(SITE, "replay.json"))
curve = jload(os.path.join(SITE, "checkpoints", "curve.json"))
pts = curve["points"]
cs = [p["step"] for p in pts]
check(cs == sorted(set(cs)) and cs[0] == 0 and cs[-1] == 30000, f"curve.json: {len(pts)} points, unique, sorted, 0..30000")
check(all(all(math.isfinite(p[k]) for k in ("train", "val", "lr")) for p in pts), "curve.json values finite")
check(all(p["tokens"] == p["step"] * curve["batchTokens"] for p in pts), "curve tokens = step x batchTokens")
by = {p["step"]: p for p in pts}
check(all(c["step"] in by and abs(by[c["step"]]["train"] - c["trainLoss"]) < 1e-9 and abs(by[c["step"]]["val"] - c["valLoss"]) < 1e-9 for c in cks),
      "every checkpoint has a curve point with the same train/val loss")
check([c["step"] for c in replay["checkpoints"]] == steps, "replay.json has every checkpoint, same order")
probe_ok = len(replay["probes"]) == len(replay["probeIds"]) == 6 and all(tok.encode(p) == i for p, i in zip(replay["probes"], replay["probeIds"]))
check(probe_ok, "replay probes: 6, probeIds = encode(probe)")
complete = True
for c in replay["checkpoints"]:
    if len(c["probes"]) != len(replay["probeIds"]):
        complete = False
    for pr in c["probes"]:
        if len(pr["top"]) != 10 or any(pr["top"][i]["p"] < pr["top"][i + 1]["p"] for i in range(9)):
            complete = False
        if pr["greedy"]["text"] != tok.decode(pr["greedy"]["ids"]) or pr["sampled"]["text"] != tok.decode(pr["sampled"]["ids"]):
            complete = False
        if not (pr["greedy"]["ids"] and pr["sampled"]["ids"]):
            complete = False
        if any(t["text"] != tok.token_to_display(t["id"]) for t in pr["top"]):
            complete = False
check(complete, "every checkpoint x probe has 10 sorted top entries, greedy and sampled ids + matching text")

# held-out sequences, rebuilt here: first 65 tokens of 200 evenly spaced val stories (no EOT inside)
st = [0] + [int(i) + 1 for i in np.nonzero(val == EOT)[0][:-1]]
st = [s for s in st if s + 65 <= len(val) and not (val[s:s + 65] == EOT).any()]
pick = [st[int(i)] for i in np.linspace(0, len(st) - 1, 200)]
held = torch.from_numpy(np.stack([val[s:s + 65] for s in pick]).astype(np.int64))

for c in cks:
    path = os.path.join(SITE, "checkpoints", c["file"])
    size = os.path.getsize(path)
    f16 = np.fromfile(path, dtype="<f2")
    ok_size = size == 2 * spec["params"]
    m16 = model_from_flat(f16.astype(np.float32))
    pt = torch.load(os.path.join(RUN, "replay", f"step_{c['step']:06d}.pt"), map_location="cpu", weights_only=False)
    ref16 = {k: v.half().float() for k, v in pt["model"].items()}
    d_pt = max(float((m16.state_dict()[k] - v).abs().max()) for k, v in ref16.items())
    with torch.no_grad():
        lg, loss = m16(held[:, :-1], held[:, 1:])
        p = torch.softmax(lg.double(), -1)
        ent = float(-(p * p.clamp_min(1e-300).log()).sum(-1).mean())
        pmax = float(p.max())
    rp = next(r for r in replay["checkpoints"] if r["step"] == c["step"])
    dh = abs(float(loss) - rp["heldOut"]["loss"])
    line = (f"step {c['step']:6d}: size {size} ok={ok_size}, vs replay .pt (fp16-rounded) max|diff| {d_pt:.1e}, "
            f"heldOut {float(loss):.4f} (replay {rp['heldOut']['loss']}), mean entropy {ent:.3f} nats, max p {pmax:.4f}")
    print("  " + line)
    if not (ok_size and d_pt == 0.0 and dh < 1e-3):
        problems.append("checkpoint " + line)
    if c["step"] == 0:
        torch.manual_seed(1337)  # train.py: torch.manual_seed(args.seed) then GPT(cfg)
        init_model = GPT(cfg)
        d_init = max(float((m16.state_dict()[k] - v.half().float()).abs().max()) for k, v in init_model.state_dict().items())
        check(d_init == 0.0, f"step-0 checkpoint == fresh GPT init with seed 1337 (fp16-rounded), max |diff| {d_init:.1e}")
        check(abs(ent - math.log(V)) < 0.05 and pmax < 0.02, f"step 0 near-uniform: entropy {ent:.4f} vs ln(1024) {math.log(V):.4f}, max p {pmax:.4f}")
    if c["step"] == 30000:
        d_last = float(np.abs(f16.astype(np.float32) - flat32.astype(np.float16).astype(np.float32)).max())
        check(d_last == 0.0, f"step_030000.bin == weights.bin rounded to fp16 (max |diff| {d_last})")

# ------------------------------------------------------------------------------------------------
print("7) sizes and licenses")
total = 0
for dp, _, fs in os.walk(SITE):
    for fn in fs:
        s = os.path.getsize(os.path.join(dp, fn))
        total += s
        if s > 5_000_000:
            problems.append(f"file over 5 MB: {fn} {s}")
check(total <= 15_000_000, f"tinygpt folder total {total} bytes ({total / 1e6:.2f} MB, {total / 2**20:.2f} MiB) <= 15 MB")
big = max(((os.path.getsize(os.path.join(dp, fn)), fn) for dp, _, fs in os.walk(SITE) for fn in fs))
check(big[0] <= 5_000_000, f"largest file {big[1]} {big[0]} bytes <= 5 MB")
lic = jload(os.path.join(ROOT, "site", "static", "licenses.json"))
check(isinstance(lic, list) and all(set(("name", "url", "license", "usedFor", "attribution")) <= set(x) for x in lic),
      f"licenses.json: valid JSON array of {len(lic)} entries with all 5 keys")
ts_entries = [x for x in lic if "tinystories" in (x["name"] + x["url"]).lower()]
check(len(ts_entries) >= 1, f"TinyStories dataset entry present: {[(x['name'], x['license']) for x in ts_entries]}")

with open(os.path.join(HERE, "verify_ref.json"), "w", encoding="utf-8") as f:
    json.dump(ref, f, ensure_ascii=False)
print(f"wrote verify_ref.json ({os.path.getsize(os.path.join(HERE, 'verify_ref.json'))} bytes)")
print("PROBLEMS:\n  " + "\n  ".join(problems) if problems else "PYTHON-SIDE VERIFICATION OK")
