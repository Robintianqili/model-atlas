"""Independent verification of the tinycnn assets (verification stream).
Rebuilds TinyVGG from site/static/models/tinycnn/weights.bin + spec.json (NOT from the checkpoint),
re-evaluates the saved test split, checks the contract and dumps a per-layer reference for the TS check.
usage (from anywhere): python pipeline/tinycnn/verify/verify_assets.py
"""
import base64, hashlib, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "pipeline", "tinycnn"))
from common import TinyVGG, load_split, load_class, split_indices  # noqa: E402
from classes import BASE, HELD_OUT  # noqa: E402

D = os.path.join(ROOT, "site", "static", "models", "tinycnn")
OUTDIR = os.path.join(HERE, "out")
os.makedirs(OUTDIR, exist_ok=True)
problems = []


def check(cond, msg):
    print(("OK   " if cond else "FAIL ") + msg)
    if not cond:
        problems.append(msg)


spec = json.load(open(os.path.join(D, "spec.json"), encoding="utf-8"))
# ---------------- contract fields
for k in ["slug", "name", "version", "params", "precision", "input", "tensors", "layers", "training", "parity"]:
    check(k in spec, f"spec has '{k}'")
check(spec["slug"] == "tinycnn", "slug == tinycnn")
check(isinstance(spec["name"], dict) and spec["name"].get("en") and spec["name"].get("zh"), "name bilingual")
check(spec["precision"] == "fp32", "precision fp32")
check(spec["input"]["shape"] == [1, 28, 28] and spec["input"]["kind"] == "drawing", "input shape/kind")
wb = os.path.getsize(os.path.join(D, "weights.bin"))
off = 0
for t in spec["tensors"]:
    n = int(np.prod(t["shape"]))
    check(n == t["length"] and t["offset"] == off, f"tensor {t['name']} shape {t['shape']} offset {t['offset']} length {t['length']} (expected offset {off})")
    off += t["length"]
check(off * 4 == wb, f"sum(length)*4 = {off*4} == weights.bin size {wb}")
check(off == spec["params"], f"params {spec['params']} == sum of tensor lengths {off}")
ids = [L["id"] for L in spec["layers"]]
want_ids = ["input", "conv1_1", "relu1_1", "conv1_2", "relu1_2", "pool1", "conv2_1", "relu2_1", "conv2_2", "relu2_2", "pool2", "flatten", "dense", "softmax"]
check(ids == want_ids, f"layer ids in order: {ids}")
for L in spec["layers"]:
    for k in ["id", "op", "label", "in", "outShape"]:
        if k not in L:
            check(False, f"layer {L.get('id')} lacks {k}")
    if not (L["label"].get("en") and L["label"].get("zh")):
        check(False, f"layer {L['id']} label not bilingual")
    if "desc" in L and not (L["desc"].get("en") and L["desc"].get("zh")):
        check(False, f"layer {L['id']} desc not bilingual")
check(all("desc" in L for L in spec["layers"]), "every layer has a bilingual desc")
for c in spec["classes"] + spec["heldOutClasses"]:
    if not (c.get("id") and c.get("en") and c.get("zh")):
        check(False, f"class {c} incomplete")
check([c["id"] for c in spec["classes"]] == BASE, "spec.classes == BASE")
check([c["id"] for c in spec["heldOutClasses"]] == HELD_OUT, "spec.heldOutClasses == HELD_OUT")
check(spec["training"].get("dataset") and "steps" in spec["training"] and "metrics" in spec["training"], "training has dataset/steps/metrics")

# ---------------- rebuild from weights.bin
W = np.fromfile(os.path.join(D, "weights.bin"), dtype="<f4")
ch = (spec["layers"][1]["outShape"][0], spec["layers"][3]["outShape"][0], spec["layers"][6]["outShape"][0], spec["layers"][8]["outShape"][0])
m = TinyVGG(ch, len(BASE))
sd = {}
for t in spec["tensors"]:
    sd[t["name"]] = torch.tensor(W[t["offset"]:t["offset"] + t["length"]].reshape(t["shape"]).copy())
m.load_state_dict(sd)
m.eval()
torch.set_grad_enabled(False)
print("channels", ch, "params", sum(p.numel() for p in m.parameters()))
for name in ["model_final.pt", "runs/c8e60/best.pt"]:
    p = os.path.join(ROOT, "pipeline", "tinycnn", name)
    if os.path.exists(p):
        ck = torch.load(p, map_location="cpu")
        md = max(float((ck[k].float() - sd[k]).abs().max()) for k in sd)
        check(md == 0.0, f"weights.bin == {name} (max |diff| {md})")

# ---------------- test accuracy on the saved split
tr, va, te = split_indices()
check(len(set(tr) & set(te)) == 0 and len(set(va) & set(te)) == 0 and len(set(tr) & set(va)) == 0, "split indices disjoint")
Xte, yte, ite = load_split(BASE, "test")
Xva, yva, _ = load_split(BASE, "val")


def logits(X):
    return torch.cat([m(torch.tensor(X[i:i + 4096]).unsqueeze(1).float() / 255) for i in range(0, len(X), 4096)])


Lte = logits(Xte)
pred = Lte.argmax(1).numpy()
acc = float((pred == yte).mean())
vacc = float((logits(Xva).argmax(1).numpy() == yva).mean())
C = len(BASE)
cm = np.zeros((C, C), int)
for t_, p_ in zip(yte, pred):
    cm[t_, p_] += 1
met = spec["training"]["metrics"]
print(f"test acc {acc:.6f} ({(pred == yte).sum()}/{len(yte)})  claimed {met['test_acc']}; val {vacc:.6f} claimed {met['val_acc']}")
check(abs(acc - met["test_acc"]) < 6e-5, "test acc matches spec (to 4 dp)")
check(abs(vacc - met["val_acc"]) < 6e-5, "val acc matches spec")
check(cm.tolist() == met["confusion"], "confusion matrix matches spec")
pc = {BASE[i]: round(float(cm[i, i] / cm[i].sum()), 4) for i in range(C)}
check(pc == met["per_class_test_acc"], f"per-class matches spec {pc}")
check(met.get("test_n") == len(yte), "test_n")
# GPU check too (different kernels)
if torch.cuda.is_available():
    mg = TinyVGG(ch, len(BASE)).cuda(); mg.load_state_dict(sd); mg.eval()
    Lg = torch.cat([mg(torch.tensor(Xte[i:i + 4096], device="cuda").unsqueeze(1).float() / 255).cpu() for i in range(0, len(Xte), 4096)])
    print(f"GPU test acc {float((Lg.argmax(1).numpy() == yte).mean()):.6f}")
# exact-duplicate leakage between train and test (same bytes)
leak = 0
for k, c in enumerate(BASE + HELD_OUT):
    a = load_class(c)
    hs_tr = {hashlib.md5(a[i].tobytes()).digest() for i in tr}
    leak_c = sum(hashlib.md5(a[i].tobytes()).digest() in hs_tr for i in te)
    leak += leak_c
print(f"exact train/test duplicate drawings (12 classes): {leak} of {len(te)*12}")

# ---------------- reference.json
ref = json.load(open(os.path.join(D, "reference.json")))
for r in ref["inputs"]:
    k = BASE.index(r["label"])
    raw = load_class(r["label"])[r["fileIndex"]]
    x = np.asarray(r["input"], np.float32)
    check(np.abs(x - raw.astype(np.float32) / 255).max() < 1e-7, f"reference input {r['label']}#{r['fileIndex']} = dataset row")
    check(r["fileIndex"] in set(te.tolist()), f"reference {r['label']} is from the test split")
    t = m.trace(torch.tensor(x).reshape(1, 1, 28, 28))
    md = max(float(np.abs(np.asarray(r["layers"][lid]["data"], np.float32) - t[lid][0].numpy().ravel()).max()) for lid in want_ids)
    check(md < 1e-5, f"reference layers reproduced from weights.bin (max |diff| {md:.2e})")

# ---------------- extra per-layer reference: 40 random test drawings for the TS per-layer parity check
rng = np.random.default_rng(99)
sel = rng.choice(len(Xte), 40, replace=False)
X = torch.tensor(Xte[sel]).unsqueeze(1).float() / 255
t = m.trace(X)
blob = {"labels": [int(v) for v in yte[sel]], "layers": {}}
parts = [X.numpy().astype("<f4").ravel()]
offs = 0
for lid in want_ids:
    a = t[lid].numpy().astype("<f4")
    blob["layers"][lid] = {"shape": list(a.shape[1:]), "offset": offs, "per": int(np.prod(a.shape[1:]))}
    parts.append(a.ravel())
    offs += a.size
inputs = parts[0]
np.concatenate([inputs] + parts[1:]).astype("<f4").tofile(os.path.join(OUTDIR, "extra_ref.bin"))
blob["inputOffset"] = 0
for lid in want_ids:
    blob["layers"][lid]["offset"] += inputs.size
json.dump(blob, open(os.path.join(OUTDIR, "extra_ref.json"), "w"))

# ---------------- file sizes
tot = 0
for f in sorted(os.listdir(D)):
    s = os.path.getsize(os.path.join(D, f)); tot += s
    check(s <= 5 * 1024 * 1024, f"{f} {s/1024:.1f} KB <= 5 MB")
check(tot <= 15 * 1024 * 1024, f"model total {tot/1024/1024:.2f} MB <= 15 MB")

# ---------------- presets
pre = json.load(open(os.path.join(D, "presets.json"), encoding="utf-8"))
check(len(pre["classes"]) == 10 and len(pre["heldOut"]) == 2, "presets: 10 classes + 2 held out")
grid = []
for group in ["classes", "heldOut"]:
    for g in pre[group]:
        row = []
        arr = load_class(g["id"])
        check(len(g["items"]) == 6 and g.get("zh"), f"presets {g['id']}: 6 items, zh")
        for it in g["items"]:
            b = np.frombuffer(base64.b64decode(it["b64"]), np.uint8)
            assert b.size == 784
            same = np.array_equal(b, arr[it["fileIndex"]])
            if not same:
                check(False, f"preset {g['id']}#{it['fileIndex']} != dataset row")
            if it["fileIndex"] not in set(te.tolist()):
                check(False, f"preset {g['id']}#{it['fileIndex']} not in test split")
            if group == "classes":
                p = torch.softmax(m(torch.tensor(b.reshape(1, 1, 28, 28)).float() / 255), 1)[0]
                if int(p.argmax()) != BASE.index(g["id"]) or float(p.max()) <= 0.9:
                    check(False, f"preset {g['id']}#{it['fileIndex']} predicted {BASE[int(p.argmax())]} p={float(p.max()):.3f}")
            row.append(b.reshape(28, 28))
        grid.append(row)
print("presets decoded and matched to dataset rows")
from PIL import Image  # noqa: E402
S = 4
img = np.zeros((12 * 30, 6 * 30), np.uint8) + 60
for r, row in enumerate(grid):
    for c, b in enumerate(row):
        img[r * 30 + 1:r * 30 + 29, c * 30 + 1:c * 30 + 29] = b
Image.fromarray(img).resize((img.shape[1] * S, img.shape[0] * S), Image.NEAREST).save(os.path.join(OUTDIR, "presets_grid.png"))
print("wrote", os.path.join(OUTDIR, "presets_grid.png"))
print("\nPROBLEMS:", len(problems))
for p in problems:
    print(" -", p)
