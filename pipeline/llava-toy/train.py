"""llava-toy: train the projector and the tiny language model on questions about Quick, Draw! drawings (GPU; fixed seed).

    npm run atlas -- py pipeline/llava-toy/train.py --run v1 [--epochs 30] [--seed 0]

The frozen TinyViT (pipeline/tinyvit/model_final.pt) is never trained: its 3 blocks only compute the 16 image features
of each drawing. Data: ak.data.quickdraw(CLASSES, split), TinyCNN's 10 classes and split (12,000 / 1,500 / 1,500 per
class) with TinyCNN's augmentation (shift, scale, rotation, stroke thickness), as TinyViT was trained.

Each drawing gets one question, drawn afresh every epoch: half "what is this ?" / "what is it ?" (answer: the class
word), half "is this a <class> ?" / "is it an <class> ?" with the true class or another one, half each (answer: yes or
no). Loss: cross-entropy on the answer word only (as LLaVA computes it on the answer tokens). AdamW (lr 1e-3, weight
decay 0.01 on matrices), batch 512, 1 warm-up epoch then a cosine decay. The checkpoint with the best validation score
(the mean of the two question types' accuracies) is evaluated on the CPU in float32 and copied to model_final.pt.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

try:
    import atlas_kit as ak
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import atlas_kit as ak

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import CLASSES, ID, N_IMG, VOWEL, LlavaToy, image_features, load_vit, n_params  # noqa: E402

C = len(CLASSES)
CLS_ID = torch.tensor([ID[c] for c in CLASSES])
ART = torch.tensor([ID["an"] if c in VOWEL else ID["a"] for c in CLASSES])


def tinycnn_augment():
    p = ak.root_path("pipeline", "tinycnn", "common.py")
    spec = importlib.util.spec_from_file_location("llava_toy_tinycnn_common", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.augment


def tensors(split: str):
    X, y, _ = ak.data.quickdraw(CLASSES, split)
    return torch.tensor(X).float().unsqueeze(1) / 255, torch.tensor(y)


def questions(y: torch.Tensor, g: torch.Generator, kind: torch.Tensor | None = None, asked: torch.Tensor | None = None):
    """For labels y [B]: ids [B, 5] (0 pads the 4-word questions), the answer row [B] (16 + length - 1), the answer id [B],
    the kind [B] (0: what, 1: yes/no). `kind` and `asked` may be given (validation); else they are drawn from g."""
    B = len(y)
    if kind is None:
        kind = torch.randint(0, 2, (B,), generator=g)
    subj = torch.where(torch.randint(0, 2, (B,), generator=g) == 0, ID["this"], ID["it"])
    if asked is None:
        other = (y + torch.randint(1, C, (B,), generator=g)) % C
        asked = torch.where(torch.randint(0, 2, (B,), generator=g) == 0, y, other)
    ids = torch.zeros(B, 5, dtype=torch.long)
    what = kind == 0
    ids[what, 0], ids[what, 1], ids[what, 2], ids[what, 3] = ID["what"], ID["is"], subj[what], ID["?"]
    yn = ~what
    ids[yn, 0], ids[yn, 1], ids[yn, 2], ids[yn, 3], ids[yn, 4] = ID["is"], subj[yn], ART[asked[yn]], CLS_ID[asked[yn]], ID["?"]
    row = torch.where(what, N_IMG + 3, N_IMG + 4)
    ans = torch.where(what, CLS_ID[y], torch.where(asked == y, ID["yes"], ID["no"]))
    return ids, row, ans, kind


@torch.no_grad()
def evaluate(vit, m, X, y, dev, bs=2000, seed=7):
    """{what, yesno}: accuracy of 'what is this ?' and of one yes/no question per drawing (true class or another one)."""
    m.eval()
    g = torch.Generator().manual_seed(seed)
    ok = {0: 0, 1: 0}
    for k in (0, 1):
        for i in range(0, len(X), bs):
            yy = y[i:i + bs]
            ids, row, ans, _ = questions(yy, g, kind=torch.full((len(yy),), k))
            logits = m(image_features(vit, X[i:i + bs].to(dev)), ids.to(dev))
            pick = logits[torch.arange(len(yy)), row.to(dev)].argmax(-1).cpu()
            ok[k] += int((pick == ans).sum())
    return {"what": ok[0] / len(X), "yesno": ok[1] / len(X)}


def param_groups(m, wd):
    decay, no = [], []
    for n, p in m.named_parameters():
        (decay if p.dim() >= 2 and not n.endswith("pos") else no).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no, "weight_decay": 0.0}]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="v1")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--bs", type=int, default=512)
    a = ap.parse_args()

    out = HERE / "runs" / a.run
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    Xtr, ytr = tensors("train")
    Xva, yva = tensors("val")
    Xtr_d = Xtr.to(dev)
    augment = tinycnn_augment()
    vit = load_vit().to(dev)
    m = LlavaToy().to(dev)
    opt = torch.optim.AdamW(param_groups(m, a.wd), lr=a.lr)
    spe = math.ceil(len(Xtr) / a.bs)
    total, warm = a.epochs * spe, spe
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm)))))
    gen = torch.Generator(device=dev).manual_seed(a.seed + 1)
    gq = torch.Generator().manual_seed(a.seed + 2)
    best, history = (-1.0, -1), []
    print(f"llava-toy: {n_params(m)} trainable params (TinyViT frozen) on {dev}; {len(Xtr)} train, {len(Xva)} val; {a.epochs} epochs x {spe} steps")
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        m.train()
        perm = torch.randperm(len(Xtr), device=dev, generator=gen)
        tot, n = 0.0, 0
        for i in range(0, len(perm), a.bs):
            idx = perm[i:i + a.bs]
            with torch.no_grad():
                feats = image_features(vit, augment(Xtr_d[idx], gen))
            ids, row, ans, _ = questions(ytr[idx.cpu()], gq)
            logits = m(feats, ids.to(dev))
            loss = F.cross_entropy(logits[torch.arange(len(idx), device=dev), row.to(dev)], ans.to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(idx)
            n += len(idx)
        va = evaluate(vit, m, Xva, yva, dev)
        score = (va["what"] + va["yesno"]) / 2
        history.append({"epoch": ep, "loss": round(tot / n, 5), **{f"val_{k}": round(v, 4) for k, v in va.items()}})
        if score > best[0]:
            best = (score, ep)
            torch.save(m.state_dict(), out / "best.pt")
        print(f"epoch {ep:3d}  loss {tot / n:.4f}  val what {va['what']:.4f} yes/no {va['yesno']:.4f}  best {best[0]:.4f}@{best[1]}  {time.time() - t0:.1f}s", flush=True)

    # final: the best checkpoint on the CPU in float32 (TF32 is off: atlas_kit switches it off on import)
    fin = LlavaToy()
    fin.load_state_dict(torch.load(out / "best.pt", map_location="cpu"))
    vit_cpu = load_vit()
    Xte, yte = tensors("test")
    te, va = evaluate(vit_cpu, fin, Xte, yte, "cpu"), evaluate(vit_cpu, fin, Xva, yva, "cpu")
    torch.save(fin.state_dict(), HERE / "model_final.pt")
    report = {"run": a.run, "params": n_params(fin), "epochs": a.epochs, "best_epoch": best[1], "seed": a.seed,
              "steps": total, "batch": a.bs, "lr": a.lr, "weight_decay": a.wd,
              "test_what": round(te["what"], 4), "test_yesno": round(te["yesno"], 4), "val_what": round(va["what"], 4),
              "val_yesno": round(va["yesno"], 4), "test_n": len(yte), "evaluated": "CPU float32, TF32 off", "history": history}
    (out / "metrics.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    with open(HERE / "train_metrics.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "history"}))


if __name__ == "__main__":
    main()
