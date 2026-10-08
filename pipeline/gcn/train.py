"""gcn: train the GCN of model.py on Zachary's karate club with two labelled members, and measure the deep version.

    npm run atlas -- py pipeline/gcn/train.py          # < 10 s on the CPU; writes model_final.pt and train_metrics.json

Semi-supervised node classification (Kipf and Welling, 2017): the loss sees only the labelled members (Mr. Hi, node 0,
and the Officer, node 33); every other member gets its faction from the graph. Faction labels are networkx's 'club'
attribute (Zachary 1977), the graph is the UNWEIGHTED adjacency (ak.data.karate()). CPU float32, TF32 off, seed 0,
Adam lr 0.01, 200 full-graph steps (the same recipe as the browser trainer, trainer.ts).

Also measured (facts for the limits step): the same recipe with 2 and with 8 graph layers over 10 seeds (0-9).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import atlas_kit as ak
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import atlas_kit as ak

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import GCN, Deep  # noqa: E402

LABELLED = [0, 33]
STEPS, LR, SEED = 200, 0.01, 0


def data():
    k = ak.data.karate()
    return torch.tensor(k["A"]), torch.eye(k["n"]), torch.tensor(k["labels"]), k


def fit(model, adj, x, y, steps=STEPS, lr=LR):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    idx = torch.tensor(LABELLED)
    losses = []
    for _ in range(steps):
        loss = F.cross_entropy(model(adj, x)[idx], y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def unlabelled_acc(model, adj, x, y) -> tuple[float, list[int]]:
    with torch.no_grad():
        pred = model(adj, x).argmax(1)
    rest = [i for i in range(len(y)) if i not in LABELLED]
    wrong = [i for i in rest if pred[i] != y[i]]
    return 1 - len(wrong) / len(rest), wrong


def main() -> None:
    adj, x, y, _ = data()
    torch.manual_seed(SEED)
    model = GCN()
    losses = fit(model, adj, x, y)
    acc, wrong = unlabelled_acc(model, adj, x, y)
    torch.save(model.state_dict(), HERE / "model_final.pt")
    seeds = {}
    for name, make in (("shallow", GCN), ("deep", lambda: Deep(8))):
        accs = []
        for s in range(10):
            torch.manual_seed(s)
            m = make()
            fit(m, adj, x, y)
            accs.append(unlabelled_acc(m, adj, x, y)[0])
        seeds[name] = accs
    out = {"seed": SEED, "steps": STEPS, "lr": LR, "optimizer": "adam", "labelled": LABELLED, "loss_first": losses[0],
           "loss_final": losses[-1], "unlabelled_acc": acc, "wrong": wrong, "params": sum(p.numel() for p in model.parameters()),
           "seeds_shallow_acc": seeds["shallow"], "seeds_deep_acc": seeds["deep"]}
    (HERE / "train_metrics.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
