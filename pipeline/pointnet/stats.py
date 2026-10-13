"""pointnet: the measured numbers behind the lesson (facts.json), on the CPU in float32 over make_data.ts's test clouds.

    npm run atlas -- py pipeline/pointnet/stats.py          # prints them; export.py writes them into facts.json

- testAcc, testN: accuracy of model_final.pt on the 5,000 test clouds (1,000 per shape, 16 to 200 points).
- critShare, critMean, deadMean: per test cloud, the points that win at least one feature of the max (the first such
  point on a tie, as the page's max_points op) among features whose max is above 0, as a share of the cloud's points
  and as a count; and the number of features that are 0 at every point (they have no winner). Averaged over clouds.
- halfFlip, halfAcc: remove a seeded half of each test cloud's points (make_data.ts 'half'): the share of clouds whose
  answer changes, and the accuracy then.
- strayFlip, strayAcc: add one stray point STRAY radii to the right of the centre (shapes.ts withStray, make_data.ts
  'stray'): the share of clouds whose answer changes, and the accuracy then.
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    import atlas_kit as ak  # noqa: F401  (switches TF32 off)
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import atlas_kit as ak  # noqa: F401

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data import load  # noqa: E402
from model import PointNet  # noqa: E402


@torch.no_grad()
def answers(m: PointNet, name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    X, y, n = load(name)
    return m(X).argmax(1), y, n, X


@torch.no_grad()
def measure(m: PointNet) -> dict:
    """{fact key: (value, unit)}"""
    m = m.cpu().eval()
    p, y, n, X = answers(m, "test")
    crit, dead = [], []
    for j in range(len(X)):
        h = m.point_features(X[j, :int(n[j])]).numpy()       # [n, F], the real points only (no repeats)
        top = h.max(0)
        live = top > 0
        crit.append(len(set(h.argmax(0)[live].tolist())) / int(n[j]))
        dead.append(int((~live).sum()))
    crit_n = [c * int(k) for c, k in zip(crit, n)]
    ph, yh, _, _ = answers(m, "half")
    ps, ys, _, _ = answers(m, "stray")
    return {
        "testAcc": (float((p == y).float().mean()), "share"),
        "testN": (len(y), "clouds"),
        "critShare": (float(np.mean(crit)), "share"),
        "critMean": (float(np.mean(crit_n)), "points"),
        "deadMean": (float(np.mean(dead)), "features"),
        "halfFlip": (float((ph != p).float().mean()), "share"),
        "halfAcc": (float((ph == yh).float().mean()), "share"),
        "strayFlip": (float((ps != p).float().mean()), "share"),
        "strayAcc": (float((ps == ys).float().mean()), "share"),
    }


if __name__ == "__main__":
    mdl = PointNet()
    mdl.load_state_dict(torch.load(HERE / "model_final.pt", map_location="cpu"))
    for k, v in measure(mdl).items():
        print(f"{k:10s} {v[0]:.4f} {v[1]}")
