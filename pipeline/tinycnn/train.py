"""Train TinyVGG on the 10 base Quick, Draw! classes.
usage: python train.py --ch 8,8,16,16 --epochs 40 --seed 0 --out runs/c8"""
import argparse, json, os, time
import numpy as np, torch, torch.nn.functional as F
from common import TinyVGG, load_split, augment, n_params
from classes import BASE

ap = argparse.ArgumentParser()
ap.add_argument("--ch", default="8,8,16,16"); ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--bs", type=int, default=256); ap.add_argument("--wd", type=float, default=5e-4)
ap.add_argument("--noaug", action="store_true"); ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = "cuda"
torch.manual_seed(a.seed); np.random.seed(a.seed)
ch = tuple(int(c) for c in a.ch.split(","))
Xtr, ytr, _ = load_split(BASE, "train"); Xva, yva, _ = load_split(BASE, "val"); Xte, yte, _ = load_split(BASE, "test")
Xtr = torch.tensor(Xtr, device=dev).unsqueeze(1); ytr = torch.tensor(ytr, device=dev)
def T(x): return torch.tensor(x, device=dev).unsqueeze(1).float() / 255
Xva, Xte = T(Xva), T(Xte); yva_t = torch.tensor(yva, device=dev)
m = TinyVGG(ch, len(BASE)).to(dev)
print("params", n_params(m), flush=True)
decay = [p for n, p in m.named_parameters() if n.endswith("weight")]
nodecay = [p for n, p in m.named_parameters() if not n.endswith("weight")]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": nodecay, "weight_decay": 0}], lr=a.lr)
spe = len(Xtr) // a.bs; steps = a.epochs * spe
sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.15)
gen = torch.Generator(device=dev); gen.manual_seed(a.seed)

def evaluate(X, y, aug=False):
    m.eval(); out = []
    with torch.no_grad():
        g = torch.Generator(device=dev); g.manual_seed(123)
        for i in range(0, len(X), 4096):
            x = X[i:i + 4096]
            if aug: x = augment(x, g)
            out.append(m(x))
    m.train(); L = torch.cat(out)
    return (L.argmax(1) == y).float().mean().item(), L

hist = []; best = (-1, None); t0 = time.time()
for e in range(a.epochs):
    perm = torch.randperm(len(Xtr), device=dev); tot = 0.0
    for i in range(spe):
        b = perm[i * a.bs:(i + 1) * a.bs]
        x = Xtr[b].float() / 255
        if not a.noaug: x = augment(x, gen)
        loss = F.cross_entropy(m(x), ytr[b])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step(); tot += loss.item()
    va, _ = evaluate(Xva, yva_t)
    hist.append({"epoch": e + 1, "train_loss": tot / spe, "val_acc": va})
    print(f"ep {e+1} loss {tot/spe:.4f} val {va:.4f} t {time.time()-t0:.0f}s", flush=True)
    if va > best[0]: best = (va, {k: v.detach().clone() for k, v in m.state_dict().items()}, e + 1)
torch.save(m.state_dict(), os.path.join(a.out, "last.pt"))
torch.save(best[1], os.path.join(a.out, "best.pt"))
m.load_state_dict(best[1])
yte_t = torch.tensor(yte, device=dev)
te, L = evaluate(Xte, yte_t); te_aug, _ = evaluate(Xte, yte_t, aug=True)
pred = L.argmax(1).cpu().numpy()
C = len(BASE); cm = np.zeros((C, C), int)
for t, p in zip(yte, pred): cm[t, p] += 1
res = {"ch": ch, "params": n_params(m), "epochs": a.epochs, "best_epoch": best[2], "val_acc": best[0],
       "test_acc": te, "test_acc_augmented": te_aug, "per_class_test_acc": (cm.diagonal() / cm.sum(1)).tolist(),
       "confusion": cm.tolist(), "history": hist, "args": vars(a)}
json.dump(res, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
print("TEST", te, "TEST_AUG", te_aug, "best_epoch", best[2])
