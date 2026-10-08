"""Fine-tuning precompute for the transfer-learning concept page.

Task: tell the 2 held-out classes apart (ladder vs envelope) with N labelled drawings per class.
Methods (same optimiser budget for all: full-batch Adam, STEPS steps, batch = all 2N examples, no augmentation):
  head    - pretrained conv layers frozen, new 2-way dense head on the 256 flatten features
  full    - pretrained init, all layers trainable, new 2-way head
  scratch - random init, same architecture, 2-way head, all layers trainable
LR per method was picked once on the held-out classes' VAL split (python finetune.py --explore), never on test.

Writes site/static/models/tinycnn/{finetune.json, pca.json, heldout.json}
and pipeline/tinycnn/out/ft_check.{bin,json} for the node check of tinycnn_finetune.ts.
"""
import argparse, base64, json, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from common import TinyVGG, load_split, N_TRAIN
from classes import BASE, HELD_OUT, ZH

ap = argparse.ArgumentParser()
ap.add_argument("--explore", action="store_true")
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "site", "static", "models", "tinycnn")
dev = "cuda"
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.deterministic = True

NS = [1, 2, 4, 8, 16, 32, 64, 128, 512, 2048]  # 512 and 2048 are extra points showing where full fine-tuning catches up
SEEDS = [0, 1, 2, 3, 4]
STEPS = 300
LR = {"head": 1e-2, "full": 1e-3, "scratch": 1e-3}  # chosen with --explore on the val split (see notes)

base = TinyVGG((8, 8, 16, 16), len(BASE))
base.load_state_dict(torch.load(os.path.join(HERE, "model_final.pt"), map_location="cpu"))
base.eval()

Xtr, ytr, itr = load_split(HELD_OUT, "train")
Xva, yva, _ = load_split(HELD_OUT, "val")
Xte, yte, ite = load_split(HELD_OUT, "test")
T = lambda X: torch.tensor(X, device=dev).unsqueeze(1).float() / 255
Xtr_t, Xva_t, Xte_t = T(Xtr), T(Xva), T(Xte)
ytr_t, yva_t, yte_t = (torch.tensor(v, device=dev) for v in (ytr, yva, yte))


def subset(seed, n):
    """Nested subsets: the first n of a fixed per-seed permutation, for each class."""
    rng = np.random.default_rng(1000 + seed)
    pa = rng.permutation(N_TRAIN)[:max(NS)]
    pb = rng.permutation(N_TRAIN)[:max(NS)] + N_TRAIN
    return np.concatenate([pa[:n], pb[:n]])


def make(method, seed):
    torch.manual_seed(seed)
    if method == "scratch":
        m = TinyVGG((8, 8, 16, 16), 2)
    else:
        m = TinyVGG((8, 8, 16, 16), len(BASE))
        m.load_state_dict(base.state_dict())
        m.dense = nn.Linear(256, 2)  # new head, default init under the seed
    m = m.to(dev)
    if method == "head":
        for n_, p in m.named_parameters():
            p.requires_grad = n_.startswith("dense.")
    return m


def run(method, seed, n, lr, Xev, yev):
    m = make(method, seed)
    idx = torch.tensor(subset(seed, n), device=dev)
    x, y = Xtr_t[idx], ytr_t[idx]
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    m.train()
    if method == "head":
        with torch.no_grad():
            feats = m.features(x)
    for _ in range(STEPS):
        out = m.dense(feats) if method == "head" else m(x)
        loss = F.cross_entropy(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        pred = torch.cat([m(Xev[i:i + 4096]).argmax(1) for i in range(0, len(Xev), 4096)])
    acc = (pred == yev).float().mean().item()
    return acc, loss.item(), sum(p.numel() for p in params), m


if a.explore:
    for method in ["head", "full", "scratch"]:
        for lr in [1e-3, 3e-3, 1e-2, 3e-2]:
            accs = {n: np.mean([run(method, s, n, lr, Xva_t, yva_t)[0] for s in SEEDS[:3]]) for n in [2, 16, 128]}
            print(method, lr, {k: round(v, 4) for k, v in accs.items()}, "mean", round(np.mean(list(accs.values())), 4), flush=True)
    raise SystemExit

t0 = time.time()
results = {}
trainable = {}
for method in ["head", "full", "scratch"]:
    rows = []
    for n in NS:
        accs = []
        for s in SEEDS:
            acc, loss, ntr, _ = run(method, s, n, LR[method], Xte_t, yte_t)
            accs.append(round(acc, 4))
            trainable[method] = ntr
        rows.append({"n": n, "acc": accs, "mean": round(float(np.mean(accs)), 4), "std": round(float(np.std(accs)), 4)})
        print(method, n, rows[-1]["mean"], rows[-1]["std"], f"{time.time()-t0:.0f}s", flush=True)
    results[method] = rows

# what the frozen 10-class model calls the held-out drawings (zero-shot guesses)
with torch.no_grad():
    bm = base.to(dev)
    P = torch.cat([torch.softmax(bm(Xte_t[i:i + 4096]), 1) for i in range(0, len(Xte_t), 4096)]).cpu().numpy()
guess = {}
for k, c in enumerate(HELD_OUT):
    pk = P[yte == k]
    counts = np.bincount(pk.argmax(1), minlength=len(BASE)) / len(pk)
    guess[c] = {"top1Share": {BASE[j]: round(float(counts[j]), 4) for j in np.argsort(-counts)[:4]},
                "meanProbs": [round(float(v), 4) for v in pk.mean(0)]}

fin = {
    "note": "Precomputed on the RTX 5090 from the same TinyCNN (pipeline/tinycnn/finetune.py). Task: tell the two held-out classes apart. Accuracy on their test split (1,500 drawings each, 3,000 total). Training examples are N drawings per class from their train split; subsets are nested (N=16 contains N=8) and differ per seed.",
    "task": {"classes": [{"id": c, "en": c, "zh": ZH[c]} for c in HELD_OUT], "testN": int(len(yte))},
    "budget": {"optimizer": "Adam", "steps": STEPS, "batch": "all 2N examples (full batch)", "augmentation": "none",
               "lr": LR, "lrChosenOn": "held-out classes' validation split (1,500 per class), seeds 0-2, N in {2,16,128}",
               "init": "PyTorch default init under torch.manual_seed(seed) for the new head (head, full) or the whole net (scratch)"},
    "methods": {
        "head": {"label": {"en": "Head only (frozen features)", "zh": "只训练输出头（冻结特征）"}, "trainableParams": trainable["head"],
                 "desc": {"en": "Freeze the pretrained conv layers. Train a new 2-way dense layer on the 256 flatten features.",
                          "zh": "冻结预训练的卷积层，只在 256 维展平特征上训练一个新的二分类全连接层。"}},
        "full": {"label": {"en": "Full fine-tune", "zh": "全量微调"}, "trainableParams": trainable["full"],
                 "desc": {"en": "Start from the pretrained weights with a new head; every layer is trained.",
                          "zh": "从预训练权重出发，换一个新输出头，所有层都参与训练。"}},
        "scratch": {"label": {"en": "From scratch", "zh": "从零训练"}, "trainableParams": trainable["scratch"],
                    "desc": {"en": "Same architecture, random initial weights, every layer trained.",
                             "zh": "同样的结构，随机初始权重，所有层都训练。"}},
    },
    "ns": NS, "seeds": SEEDS,
    "results": results,
    "baseModelGuesses": {"note": "What the frozen 10-class model predicts for the held-out test drawings (it has never seen these classes).",
                         "classes": BASE, "byHeldOutClass": guess},
}
json.dump(fin, open(os.path.join(OUT, "finetune.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# ---------------- PCA of flatten features
Xb, yb, _ = load_split(BASE, "test")
rng = np.random.default_rng(5)
selb = np.concatenate([rng.choice(np.where(yb == k)[0], 500, replace=False) for k in range(len(BASE))])
selh = np.concatenate([rng.choice(np.where(yte == k)[0], 500, replace=False) for k in range(len(HELD_OUT))])
with torch.no_grad():
    Fb = base.features(T(Xb[selb])).cpu().double().numpy()
    Fh = base.features(Xte_t[torch.tensor(selh, device=dev)]).cpu().double().numpy()
labs = np.concatenate([yb[selb], yte[selh] + len(BASE)])
Fall = np.concatenate([Fb, Fh])


def pca(Fm):
    mu = Fm.mean(0)
    U, S, Vt = np.linalg.svd(Fm - mu, full_matrices=False)
    var = S ** 2 / (len(Fm) - 1)
    comp = Vt[:2]
    # sign convention: largest-magnitude loading positive
    for i in range(2):
        if comp[i][np.argmax(np.abs(comp[i]))] < 0:
            comp[i] = -comp[i]
    return mu, comp, var[:2] / var.sum()


def f6(v):
    return [float(f"{x:.6g}") for x in np.asarray(v).ravel()]


bases = {}
for key, Fm, lab_sel in [("all", Fall, None), ("heldOut", Fh, None)]:
    mu, comp, evr = pca(Fm)
    bases[key] = {"mean": f6(mu.astype(np.float32)), "components": [f6(comp[0].astype(np.float32)), f6(comp[1].astype(np.float32))],
                  "explainedVarianceRatio": [round(float(v), 4) for v in evr]}
# 300 sample points: 25 per class (12 classes), projected with both bases
pts_idx = np.concatenate([np.where(labs == k)[0][:25] for k in range(len(BASE) + len(HELD_OUT))])
for key in bases:
    mu = np.array(bases[key]["mean"]); comp = np.array(bases[key]["components"])
    P2 = (Fall[pts_idx] - mu) @ comp.T
    bases[key]["points"] = [{"c": int(labs[i]), "x": round(float(p[0]), 4), "y": round(float(p[1]), 4)} for i, p in zip(pts_idx, P2)]
pca_json = {
    "note": "2D PCA of TinyCNN flatten features (256-d, frozen pretrained model). 'all' is fit on 500 test drawings of each of the 12 classes; 'heldOut' only on the 2 held-out classes (1,000 drawings). Project a live drawing with (features - mean) . components. Point 'c' indexes 'classes' (10 base classes, then the 2 held-out classes).",
    "classes": [{"id": c, "en": c, "zh": ZH[c], "heldOut": c in HELD_OUT} for c in BASE + HELD_OUT],
    "bases": bases,
}
json.dump(pca_json, open(os.path.join(OUT, "pca.json"), "w", encoding="utf-8"), ensure_ascii=False)

# ---------------- heldout.json: dataset drawings the page can use live
held = {"note": "Quick, Draw! drawings of the 2 held-out classes (CC BY 4.0, Quick, Draw! dataset by Google Creative Lab). base64 of 784 uint8 (28x28, white on black); model input = value/255. 'train' = the first 32 of seed 0's training order (so train[:N] is exactly the seed-0 subset of size N used in finetune.json); 'test' = 100 random test-split drawings.",
        "classes": []}
s0 = subset(0, 32)
trng = np.random.default_rng(9)
for k, c in enumerate(HELD_OUT):
    tr_idx = s0[k * 32:(k + 1) * 32]
    te_idx = trng.choice(np.where(yte == k)[0], 100, replace=False)
    held["classes"].append({"id": c, "en": c, "zh": ZH[c],
                            "train": [base64.b64encode(Xtr[i].tobytes()).decode() for i in tr_idx],
                            "test": [base64.b64encode(Xte[i].tobytes()).decode() for i in te_idx]})
json.dump(held, open(os.path.join(OUT, "heldout.json"), "w", encoding="utf-8"), ensure_ascii=False)

# ---------------- node check data: N=16 subsets for all seeds + the full held-out test split
os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
chunks = [Xtr[subset(s, 16)] for s in SEEDS] + [Xte]
np.concatenate([c.reshape(-1, 784) for c in chunks]).astype(np.uint8).tofile(os.path.join(HERE, "out", "ft_check.bin"))
json.dump({"seeds": SEEDS, "n": 16, "trainLabels": [int(v) for v in ytr[subset(0, 16)]], "testLabels": yte.tolist(),
           "pythonHeadAcc16": results["head"][NS.index(16)]["acc"]},
          open(os.path.join(HERE, "out", "ft_check.json"), "w"))
for f in ["finetune.json", "pca.json", "heldout.json"]:
    print(f, os.path.getsize(os.path.join(OUT, f)))
