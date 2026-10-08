"""tinymamba: measured numbers for the page (facts.json via export.py), on rows the model never trained on.

    npm run atlas -- py pipeline/tinymamba/stats.py          # writes stats.json next to this file (CPU, float32)

  accK<k>, exactK<k>   k = 1..8 letters (training had 1..4): the share of answers right / of rows with every answer
                       right, 2,000 rows each (bodies of max(6, k)..16 tokens, seed 30 + k, data.py's drawing rules)
  accGap<g>            one letter, then g dots, then one '?': the share right over the 8 letters x up to 6 leading
                       dots (training bodies had at most 15 dots after a letter)
  delta<Kind><b>       block b's step size delta, averaged over all channels, at dots / letters / question marks of
                       the 5,000 validation rows (seed 2)
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
from data import ASK, DOT, LETTERS, batch  # noqa: E402
from model import TinyMamba  # noqa: E402

metrics = json.loads((HERE / "train_metrics.json").read_text(encoding="utf-8"))
m = TinyMamba(**metrics["config"])
m.load_state_dict(torch.load(HERE / "model_final.pt", map_location="cpu"))
m.eval()


def with_k(rng, k):
    n = max(6, k) + rng.int(16 - max(6, k) + 1)
    pos = list(range(n))
    for i in range(k):
        j = i + rng.int(n - i)
        pos[i], pos[j] = pos[j], pos[i]
    places = sorted(pos[:k])
    letters = [1 + rng.int(LETTERS) for _ in range(k)]
    body = [DOT] * n
    for p, c in zip(places, letters):
        body[p] = c
    return body + [ASK] * k, [-1] * n + letters


@torch.no_grad()
def score(rows):
    right = total = exact = 0
    for ids, tg in rows:
        pred = m(torch.tensor(ids)).argmax(-1).tolist()
        ok = [pred[t] == y for t, y in enumerate(tg) if y >= 0]
        right, total, exact = right + sum(ok), total + len(ok), exact + all(ok)
    return right / total, exact / len(rows)


out = {}
for k in range(1, 9):
    rng = ak.Rng(30 + k)
    out[f"accK{k}"], out[f"exactK{k}"] = score([with_k(rng, k) for _ in range(2000)])
for g in (2, 4, 8, 12, 16, 20, 24, 30, 38):
    rows = [([DOT] * lead + [c] + [DOT] * g + [ASK], [-1] * (lead + 1 + g) + [c]) for c in range(1, LETTERS + 1) for lead in range(6) if lead + g + 2 <= 40]
    out[f"accGap{g}"] = score(rows)[0]
kinds = {"Dot": lambda i: i == DOT, "Letter": lambda i: 1 <= i <= LETTERS, "Ask": lambda i: i == ASK}
with torch.no_grad():
    sums = {(k, b): [0.0, 0] for k in kinds for b in range(len(m.blocks))}
    for ids, _ in batch(2, 5000):
        x = m.emb(torch.tensor(ids))
        for b, blk in enumerate(m.blocks):
            u, dt, Bm, Cm = blk.parts(x)
            delta = F.softplus(dt).mean(-1)
            for t, i in enumerate(ids):
                for k, f in kinds.items():
                    if f(i):
                        sums[(k, b)][0] += delta[t].item()
                        sums[(k, b)][1] += 1
            x = blk(x)
    for (k, b), (s, n) in sums.items():
        out[f"delta{k}{b + 1}"] = s / n
(HERE / "stats.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
print(json.dumps(out, indent=1))
