"""Independent checks of finetune.json / heldout.json / pca.json (verification stream).
- completeness: 3 methods x all ns x all seeds, mean/std consistent with acc lists
- re-runs a sample of (method, N, seed) cells from scratch with a base model rebuilt from weights.bin
- heldout.json train[:N] == seed-0 subset, test drawings come from the test split
- pca.json: shapes, orthonormal components, points re-projected from live features
usage: python pipeline/tinycnn/verify/verify_finetune.py
"""
import base64, json, os, sys
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "pipeline", "tinycnn"))
from common import TinyVGG, load_split, N_TRAIN  # noqa: E402
from classes import BASE, HELD_OUT  # noqa: E402

D = os.path.join(ROOT, "site", "static", "models", "tinycnn")
problems = []


def check(c, msg):
    print(("OK   " if c else "FAIL ") + msg)
    if not c:
        problems.append(msg)


ft = json.load(open(os.path.join(D, "finetune.json"), encoding="utf-8"))
ns, seeds = ft["ns"], ft["seeds"]
for meth in ["head", "full", "scratch"]:
    rows = ft["results"][meth]
    check([r["n"] for r in rows] == ns, f"{meth}: rows for every n {ns}")
    for r in rows:
        if len(r["acc"]) != len(seeds):
            check(False, f"{meth} n={r['n']}: {len(r['acc'])} seeds")
        if abs(np.mean(r["acc"]) - r["mean"]) > 6e-5 or abs(np.std(r["acc"]) - r["std"]) > 6e-5:
            check(False, f"{meth} n={r['n']}: mean/std inconsistent")
    m = ft["methods"][meth]
    check(bool(m["label"]["en"] and m["label"]["zh"] and m["desc"]["en"] and m["desc"]["zh"]), f"{meth}: bilingual label/desc")
H = {r["n"]: r["mean"] for r in ft["results"]["head"]}
S = {r["n"]: r["mean"] for r in ft["results"]["scratch"]}
Fu = {r["n"]: r["mean"] for r in ft["results"]["full"]}
for n in [1, 2, 4, 8, 16]:
    check(H[n] > S[n], f"N={n}: head {H[n]:.4f} > scratch {S[n]:.4f}")
check(ft["methods"]["head"]["trainableParams"] == 256 * 2 + 2, "head trainable params = 514")
check(ft["methods"]["full"]["trainableParams"] == 72 + 8 + 576 + 8 + 1152 + 16 + 2304 + 16 + 514, "full trainable params = 4666")

# ---------------- re-run a sample of cells
dev = "cuda"
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.deterministic = True
spec = json.load(open(os.path.join(D, "spec.json"), encoding="utf-8"))
W = np.fromfile(os.path.join(D, "weights.bin"), dtype="<f4")
sd = {t["name"]: torch.tensor(W[t["offset"]:t["offset"] + t["length"]].reshape(t["shape"]).copy()) for t in spec["tensors"]}
Xtr, ytr, _ = load_split(HELD_OUT, "train")
Xte, yte, ite = load_split(HELD_OUT, "test")
T = lambda X: torch.tensor(X, device=dev).unsqueeze(1).float() / 255
Xtr_t, Xte_t = T(Xtr), T(Xte)
ytr_t, yte_t = torch.tensor(ytr, device=dev), torch.tensor(yte, device=dev)
LR = ft["budget"]["lr"]
STEPS = ft["budget"]["steps"]
MAXN = max(ns)


def subset(seed, n):
    rng = np.random.default_rng(1000 + seed)
    pa = rng.permutation(N_TRAIN)[:MAXN]
    pb = rng.permutation(N_TRAIN)[:MAXN] + N_TRAIN
    return np.concatenate([pa[:n], pb[:n]])


def run(method, seed, n):
    torch.manual_seed(seed)
    if method == "scratch":
        m = TinyVGG((8, 8, 16, 16), 2)
    else:
        m = TinyVGG((8, 8, 16, 16), 10)
        m.load_state_dict(sd)
        m.dense = nn.Linear(256, 2)
    m = m.to(dev)
    if method == "head":
        for k, p in m.named_parameters():
            p.requires_grad = k.startswith("dense.")
    idx = torch.tensor(subset(seed, n), device=dev)
    x, y = Xtr_t[idx], ytr_t[idx]
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=LR[method])
    if method == "head":
        with torch.no_grad():
            f = m.features(x)
    for _ in range(STEPS):
        out = m.dense(f) if method == "head" else m(x)
        loss = F.cross_entropy(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        pred = torch.cat([m(Xte_t[i:i + 4096]).argmax(1) for i in range(0, len(Xte_t), 4096)])
    return (pred == yte_t).float().mean().item()


cells = [("head", 1), ("head", 16), ("full", 4), ("scratch", 1), ("scratch", 16)]
maxgap = 0.0
for meth, n in cells:
    row = next(r for r in ft["results"][meth] if r["n"] == n)
    got = [round(run(meth, s, n), 4) for s in seeds]
    gap = max(abs(a - b) for a, b in zip(got, row["acc"]))
    maxgap = max(maxgap, gap)
    print(f"  re-run {meth} N={n}: {got} mean {np.mean(got):.4f} | finetune.json {row['acc']} mean {row['mean']:.4f} | max gap {gap:.4f}")
check(maxgap <= 0.01, f"re-run cells within 1 point of finetune.json (max gap {maxgap:.4f})")

# ---------------- heldout.json
held = json.load(open(os.path.join(D, "heldout.json"), encoding="utf-8"))
s0 = subset(0, 32)
test_set = {(int(y), Xte[i].tobytes()) for i, y in enumerate(yte)}
for k, c in enumerate(held["classes"]):
    check(c["id"] == HELD_OUT[k] and c["zh"], f"heldout class {k} = {c['id']}")
    tr = [base64.b64decode(b) for b in c["train"]]
    want = [Xtr[i].tobytes() for i in s0[k * 32:(k + 1) * 32]]
    check(tr == want, f"heldout {c['id']} train[0:32] == seed-0 subset order")
    check(len(c["test"]) == 100 and all((k, base64.b64decode(b)) in test_set for b in c["test"]), f"heldout {c['id']}: 100 test drawings from the test split")
    check(not (set(tr) & {base64.b64decode(b) for b in c["test"]}), f"heldout {c['id']}: train and test disjoint")

# ---------------- pca.json
pca = json.load(open(os.path.join(D, "pca.json"), encoding="utf-8"))
check(len(pca["classes"]) == 12 and [c["id"] for c in pca["classes"]] == BASE + HELD_OUT, "pca classes = 10 base + 2 held out")
base = TinyVGG((8, 8, 16, 16), 10); base.load_state_dict(sd); base.eval()
for key in ["all", "heldOut"]:
    b = pca["bases"][key]
    C = np.array(b["components"]); mu = np.array(b["mean"])
    check(C.shape == (2, 256) and mu.shape == (256,), f"pca {key}: shapes")
    G = C @ C.T
    check(np.abs(G - np.eye(2)).max() < 1e-4, f"pca {key}: components orthonormal (max dev {np.abs(G - np.eye(2)).max():.2e})")
    check(len(b["points"]) == 300 and sorted({p["c"] for p in b["points"]}) == list(range(12)), f"pca {key}: 300 points, 12 classes")
    ev = b["explainedVarianceRatio"]
    check(0 < ev[1] <= ev[0] < 1, f"pca {key}: EVR {ev}")
# live projection of heldout.json test drawings lands on the right side for the heldOut basis
with torch.no_grad():
    feats, labs = [], []
    for k, c in enumerate(held["classes"]):
        X = np.stack([np.frombuffer(base64.b64decode(s), np.uint8) for s in c["test"]]).reshape(-1, 1, 28, 28)
        feats.append(base.features(torch.tensor(X).float() / 255).double().numpy()); labs += [k] * len(X)
F_ = np.concatenate(feats); labs = np.array(labs)
b = pca["bases"]["heldOut"]
P = (F_ - np.array(b["mean"])) @ np.array(b["components"]).T
pts = b["points"]
cen = {c: np.mean([[p["x"], p["y"]] for p in pts if p["c"] == c], 0) for c in (10, 11)}
near = np.array([np.argmin([np.sum((p - cen[10]) ** 2), np.sum((p - cen[11]) ** 2)]) for p in P])
print(f"  heldOut PCA: {np.mean(near == labs)*100:.1f}% of 200 heldout.json test drawings nearest their own class centre")
print("\nPROBLEMS:", len(problems))
for p in problems:
    print(" -", p)
