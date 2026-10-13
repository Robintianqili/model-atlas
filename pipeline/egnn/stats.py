"""egnn: measured numbers for facts.json, on the 5,000 test molecules of make_data.ts (CPU, float32, TF32 off).

    energyMAE / forceMAE   mean |prediction - toy formula| (energy per molecule; force per coordinate)
    energyMAD              mean |E - mean E| of the test energies: the error of always answering the average energy
    forceMAD               mean |F| per coordinate: the error of always answering "no force"
    turnE / turnF          the largest |E(turned) - E| and |f(turned) - turned f| over every test molecule, each turned
                           by its own seeded angle (float32 rounding is all that is left)
    mirrorE / mirrorF      the same for the mirror image (x -> -x)
"""
from __future__ import annotations

import math

import torch

import data


def _turn(deg: float) -> torch.Tensor:
    a = math.radians(deg)
    return torch.tensor([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], dtype=torch.float32)


@torch.no_grad()
def measure(model) -> dict[str, tuple[float, str]]:
    model = model.cpu().eval()
    sets = data.by_size(data.load("test"))
    gen = torch.Generator().manual_seed(1)
    ae = af = mad_f = 0.0
    cnt = cf = 0
    turn_e = turn_f = mir_e = mir_f = 0.0
    es = []
    flip = torch.tensor([[-1.0, 0.0], [0.0, 1.0]])
    for s in sets.values():
        pos, kind = s["pos"], s["kind"]
        e, f = model(pos, kind)
        e = e.squeeze(-1)
        ae += (e - s["E"]).abs().sum().item()
        af += (f - s["F"]).abs().sum().item()
        mad_f += s["F"].abs().sum().item()
        cnt += len(e)
        cf += s["F"].numel()
        es.append(s["E"])
        for i in range(len(e)):
            R = _turn(360 * torch.rand(1, generator=gen).item())
            e2, f2 = model(pos[i] @ R.T, kind[i])
            turn_e = max(turn_e, (e2.squeeze() - e[i]).abs().item())
            turn_f = max(turn_f, (f2 - f[i] @ R.T).abs().max().item())
            e3, f3 = model(pos[i] @ flip.T, kind[i])
            mir_e = max(mir_e, (e3.squeeze() - e[i]).abs().item())
            mir_f = max(mir_f, (f3 - f[i] @ flip.T).abs().max().item())
    E = torch.cat(es)
    return {
        "testN": (cnt, "molecules"),
        "energyMAE": (round(ae / cnt, 6), "energy units (mean |error| per molecule)"),
        "forceMAE": (round(af / cf, 6), "force units (mean |error| per coordinate)"),
        "energyMAD": (round((E - E.mean()).abs().mean().item(), 6), "energy units (always answering the mean energy)"),
        "forceMAD": (round(mad_f / cf, 6), "force units (always answering 0)"),
        "turnE": (turn_e, "largest |E(turned) - E|"),
        "turnF": (turn_f, "largest |f(turned) - turned f|"),
        "mirrorE": (mir_e, "largest |E(mirrored) - E|"),
        "mirrorF": (mir_f, "largest |f(mirrored) - mirrored f|"),
    }
