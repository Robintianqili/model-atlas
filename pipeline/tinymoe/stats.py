"""tinymoe: measured numbers for the page (facts.json) on held-out text, CPU float32 (TF32 off).

    npm run atlas -- py pipeline/tinymoe/stats.py      # after train.py for the three variants; writes stats.json

Data: the first 4,000 non-overlapping windows of 64 characters of TinyStoriesV2-GPT4-valid.txt (256,000 characters;
the stories the models never trained on). For 'moe' (balancing loss) and 'noaux' (none), per block:
  - share: the share of the token slots each expert takes (each character fills 2 slots), and its mean chance;
  - balance: Switch Transformer's balancing loss E * sum_e f_e * P_e over all windows (1 = perfectly balanced);
  - dropped: the share of slots dropped when each expert may take at most ceil(c * k * T / E) slots per window
    (capacity factor c; characters served first come, first served, both choices of a character in turn), the rule
    process.ts plays on the page;
  - top: for each expert, the characters it is chosen for most often (share of that character's slots, >= 300 slots).
Validation losses of the three variants come from train_metrics.json (train.py).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

try:
    import atlas_kit as ak
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import atlas_kit as ak

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import ALPHABET, CTX, E, K, TinyMoE, encode  # noqa: E402

CAPS = (1.0, 1.25, 1.5, 2.0)


def capacity(c: float, T: int) -> int:
    return math.ceil(c * K * T / E)


def dropped(gates: torch.Tensor, p: torch.Tensor, c: float) -> tuple[int, int]:
    """Slots dropped in one window (gates, p [T, E]) under capacity factor c: characters in order, each one's experts in
    order of their chance; an expert that is full drops the slot. Returns (dropped slots, characters that lost both)."""
    T = gates.shape[0]
    cap = capacity(c, T)
    load = [0] * E
    lost = both = 0
    for t in range(T):
        order = sorted((j for j in range(E) if gates[t, j] > 0), key=lambda j: (-float(p[t, j]), j))
        n = 0
        for j in order:
            if load[j] < cap:
                load[j] += 1
            else:
                lost += 1
                n += 1
        both += n == K
    return lost, both


@torch.no_grad()
def routing(m: TinyMoE, x: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
    out = []
    for s in range(0, len(x), 500):
        out.append(m.run(x[s:s + 500])[1])
    return [(torch.cat([o[b][0] for o in out]), torch.cat([o[b][1] for o in out])) for b in range(len(out[0]))]


def block_stats(x: torch.Tensor, p: torch.Tensor, g: torch.Tensor) -> dict:
    n, T, _ = p.shape
    f = (g > 0).float().reshape(-1, E).mean(0) / K
    P = p.reshape(-1, E).mean(0)
    res = {"share": [round(float(v), 5) for v in f], "chance": [round(float(v), 5) for v in P],
           "balance": round(float(E * (f * P).sum()), 5), "busiest": round(float(f.max()), 5), "quietest": round(float(f.min()), 5)}
    drops = {}
    for c in CAPS:
        lost = both = 0
        for w in range(n):
            a, b = dropped(g[w], p[w], c)
            lost += a
            both += b
        drops[str(c)] = {"slots": round(lost / (n * T * K), 5), "both": round(both / (n * T), 5), "cap": capacity(c, T)}
    res["dropped"] = drops
    ids = x.reshape(-1)
    chosen = (g > 0).reshape(-1, E)
    top = []
    for j in range(E):
        rows = []
        for cid in range(len(ALPHABET)):
            sel = ids == cid
            cnt = int(sel.sum())
            if cnt >= 300:
                rows.append((float(chosen[sel, j].float().mean()), cid, cnt))
        rows.sort(key=lambda r: (-r[0], r[1]))
        top.append([{"char": ALPHABET[c], "share": round(s, 4), "count": cnt} for s, c, cnt in rows[:6]])
    res["top"] = top
    return res


if __name__ == "__main__":
    va = torch.tensor(encode(ak.data.tinystories_chars("TinyStoriesV2-GPT4-valid.txt", n_chars=2_000_000)), dtype=torch.long)
    n = 4000
    x = va[: n * CTX].view(n, CTX)
    tm = json.loads((HERE / "train_metrics.json").read_text(encoding="utf-8"))
    out = {"windows": n, "window": CTX, "chars": n * CTX, "caps": list(CAPS),
           "val": {k: tm[k]["val_loss"] for k in ("moe", "noaux", "dense")},
           "params": {k: tm[k]["params"] for k in ("moe", "noaux", "dense")},
           "active": {k: tm[k]["active"] for k in ("moe", "noaux", "dense")}}
    for v in ("moe", "noaux"):
        m = TinyMoE()
        m.load_state_dict(torch.load(HERE / "runs" / v / "final.pt", map_location="cpu"))
        m.eval()
        out[v] = [block_stats(x, p.reshape(n, CTX, E), g.reshape(n, CTX, E)) for p, g in routing(m, x)]
    with open(HERE / "stats.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    for v in ("moe", "noaux"):
        for b, s in enumerate(out[v]):
            print(v, "block", b + 1, "share", s["share"], "balance", s["balance"], "dropped@1", s["dropped"]["1.0"])
            for j, t in enumerate(s["top"]):
                print("   expert", j, " ".join(f"{r['char']!r}:{r['share']:.2f}" for r in t[:5]))
    print("val", out["val"])
