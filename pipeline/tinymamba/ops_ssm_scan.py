"""Python mirror of the tinymamba local op 'ssm_scan' (site/src/lib/explainers/tinymamba/ops/scan.ts).

    inputs  u [T, D], dt [T, D], B [T, N], C [T, N];  weights A_log [D, N], D [D];  params {"state": N}
    delta = softplus(dt);  h_t = exp(delta_t A) h_{t-1} + delta_t B_t u_t  (A = -exp(A_log), h_0 = 0)
    y_t = sum_n C_t[n] h_t[:, n] + D u_t;   sub-steps 'delta' [T, D] and 'h' [T, D, N]
Used from export.py's describe() as b.op("ssm_scan", id, [u, dt, B, C], params=..., weights=..., ref=ssm_scan).
"""
import torch
import torch.nn.functional as F

OPS = {"ssm_scan": {"version": 1, "category": "recurrent", "label": ("Selective scan", "选择性扫描"),
                    "attrs": {"state": {"type": "int", "required": True}}, "weights": [{"slot": "A_log"}, {"slot": "D"}]}}


def ssm_scan(xs, params, weights):
    u, dt, B, C = xs
    A = -torch.exp(weights["A_log"])
    delta = F.softplus(dt)
    h = torch.zeros_like(A)
    hs, ys = [], []
    for t in range(u.shape[0]):
        h = torch.exp(delta[t, :, None] * A) * h + delta[t, :, None] * B[t][None, :] * u[t, :, None]
        hs.append(h)
        ys.append((h * C[t][None, :]).sum(-1) + weights["D"] * u[t])
    return torch.stack(ys), {"delta": delta, "h": torch.stack(hs)}
