"""gnn-variants: train the four networks of model.py on Zachary's karate club with two labelled members.

    npm run atlas -- py pipeline/gnn-variants/train.py      # under a minute on the CPU: model_final.pt, train_metrics.json

The same task and recipe for every variant (and the same as the gcn package): semi-supervised node classification, the
loss sees only Mr. Hi (member 0) and the Officer (member 33), every other member gets its faction from the graph.
Faction labels are networkx's 'club' attribute (Zachary 1977); the graph is the UNWEIGHTED adjacency (ak.data.karate()).
CPU float32, TF32 off, Adam lr 0.01, 200 full-graph steps, seed 0 for the exported networks; seeds 0-9 for the means.
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
from model import VARIANTS, Net, Variants  # noqa: E402

LABELLED = [0, 33]
STEPS, LR, SEED, SEEDS = 200, 0.01, 0, 10


def data():
    k = ak.data.karate()
    return torch.tensor(k["A"]), torch.eye(k["n"]), torch.tensor(k["labels"])


def fit(model, adj, x, y):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    idx = torch.tensor(LABELLED)
    losses = []
    for _ in range(STEPS):
        loss = F.cross_entropy(model(adj, x)[idx], y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def unlabelled(model, adj, x, y):
    with torch.no_grad():
        pred = model(adj, x).argmax(1)
    rest = [i for i in range(len(y)) if i not in LABELLED]
    wrong = [i for i in rest if pred[i] != y[i]]
    return 1 - len(wrong) / len(rest), wrong


def main() -> None:
    adj, x, y = data()
    out = {"seed": SEED, "steps": STEPS, "lr": LR, "optimizer": "adam", "labelled": LABELLED, "seeds": SEEDS, "variants": {}}
    full = Variants()
    for k in VARIANTS:
        torch.manual_seed(SEED)
        m = Net(k)
        losses = fit(m, adj, x, y)
        acc, wrong = unlabelled(m, adj, x, y)
        getattr(full, k).load_state_dict(m.state_dict())
        accs = []
        for s in range(SEEDS):
            torch.manual_seed(s)
            ms = Net(k)
            fit(ms, adj, x, y)
            accs.append(unlabelled(ms, adj, x, y)[0])
        out["variants"][k] = {"params": sum(p.numel() for p in m.parameters()), "loss_first": losses[0], "loss_final": losses[-1],
                              "unlabelled_acc": acc, "wrong": wrong, "seeds_acc": accs, "seeds_mean": sum(accs) / len(accs),
                              "seeds_min": min(accs), "seeds_max": max(accs)}
        print(k, json.dumps(out["variants"][k]))
    torch.save(full.state_dict(), HERE / "model_final.pt")
    (HERE / "train_metrics.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
