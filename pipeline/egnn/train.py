"""egnn: train the small EGNN (model.py) on the toy molecules of make_data.ts: energy and forces at once.

    node pipeline/egnn/make_data.ts                  # first: the sets (EGNN_DATA or pipeline/egnn/runs/data)
    npm run atlas -- py pipeline/egnn/train.py       # GPU when there is one; about a minute

Loss: mse(E) / var(E) + mse(F) / var(F) over the training set (both terms start near 1). Batches hold molecules with the
same number of atoms. Adam, cosine learning rate, seed 0; the best epoch on the validation set is kept (model_final.pt).
Writes train_metrics.json (stats.py adds the test-set numbers at export time).
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data  # noqa: E402
from model import EGNN, n_params  # noqa: E402

SEED, EPOCHS, BATCH, LR = 0, 60, 128, 2e-3


def batches(sets: dict, gen: torch.Generator, dev):
    order = []
    for n, s in sets.items():
        perm = torch.randperm(len(s["E"]), generator=gen)
        order += [(n, perm[k:k + BATCH]) for k in range(0, len(perm), BATCH)]
    for i in torch.randperm(len(order), generator=gen).tolist():
        n, idx = order[i]
        s = sets[n]
        yield s["pos"][idx].to(dev), s["kind"][idx].to(dev), s["E"][idx].to(dev), s["F"][idx].to(dev)


@torch.no_grad()
def evaluate(model, sets: dict, dev) -> dict:
    ae = af = cnt = cf = 0.0
    for s in sets.values():
        e, f = model(s["pos"].to(dev), s["kind"].to(dev))
        ae += (e.squeeze(-1) - s["E"].to(dev)).abs().sum().item()
        af += (f - s["F"].to(dev)).abs().sum().item()
        cnt += len(s["E"])
        cf += s["F"].numel()
    return {"energy_mae": ae / cnt, "force_mae": af / cf}


def main() -> None:
    torch.manual_seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va = data.load("train"), data.load("val")
    vE = tr["E"].var().item()
    vF = tr["F"][tr["F"] != 0].var().item()
    sets, vsets = data.by_size(tr), data.by_size(va)
    model = EGNN().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    steps = EPOCHS * sum(math.ceil(len(s["E"]) / BATCH) for s in sets.values())
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: 0.5 * (1 + math.cos(math.pi * min(k, steps) / steps)))
    gen = torch.Generator().manual_seed(SEED)
    best, hist, t0 = None, [], time.time()
    for ep in range(EPOCHS):
        model.train()
        tot = nb = 0.0
        for pos, kind, E, F in batches(sets, gen, dev):
            e, f = model(pos, kind)
            loss = ((e.squeeze(-1) - E) ** 2).mean() / vE + ((f - F) ** 2).mean() / vF
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot, nb = tot + loss.item(), nb + 1
        model.eval()
        v = evaluate(model, vsets, dev)
        hist.append({"epoch": ep + 1, "loss": tot / nb, **v})
        print(f"epoch {ep + 1}: loss {tot / nb:.4f}  val energy MAE {v['energy_mae']:.4f}  force MAE {v['force_mae']:.4f}", flush=True)
        score = v["energy_mae"] / math.sqrt(vE) + v["force_mae"] / math.sqrt(vF)
        if best is None or score < best[0]:
            best = (score, ep + 1, {k: t.detach().cpu().clone() for k, t in model.state_dict().items()})
    torch.save(best[2], HERE / "model_final.pt")
    out = {"seed": SEED, "epochs": EPOCHS, "batch": BATCH, "lr": LR, "steps": steps, "best_epoch": best[1], "params": n_params(model),
           "train_n": len(tr["E"]), "val_n": len(va["E"]), "var_E": vE, "var_F": vF, "seconds": round(time.time() - t0, 1),
           "device": str(dev), "history": hist}
    (HERE / "train_metrics.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"best epoch {best[1]}; {n_params(model)} params; {out['seconds']} s on {dev}")


if __name__ == "__main__":
    main()
