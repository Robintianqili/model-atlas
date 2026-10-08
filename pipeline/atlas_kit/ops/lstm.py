"""Mirror of ops/lstm.ts: one LSTM layer over a sequence [T, In] -> h [T, H], PyTorch gate order i, f, g, o:
    z = W_ih x_t + b_ih + W_hh h_{t-1} + b_hh;  i, f, o = sigmoid(.), g = tanh(.)
    c_t = f * c_{t-1} + i * g;  h_t = o * tanh(c_t);  h_0 = c_0 = 0
Both biases are kept (as nn.LSTM stores them). Sub-steps i f g o c h, each [T, H], are recorded as "<id>#<sub>".
SpecBuilder.lstm() checks this loop against the nn.LSTM module itself (AK104).
"""
import torch

OP = "lstm"
VERSION = 1
CATEGORY = "recurrent"
LABEL = ("LSTM", "LSTM（长短期记忆）")
ATTRS = {"hidden": {"type": "int", "required": True}, "inputSize": {"type": "int"}}
ALIASES: dict = {}
WEIGHTS = [{"slot": "weight_ih"}, {"slot": "weight_hh"}, {"slot": "bias_ih", "optional": True},
           {"slot": "bias_hh", "optional": True}]
SUBS = ["i", "f", "g", "o", "c", "h"]


def reference(xs, params, weights):
    x = xs[0]
    H = int(params["hidden"])
    Wih, Whh = weights["weight_ih"], weights["weight_hh"]
    bih, bhh = weights.get("bias_ih"), weights.get("bias_hh")
    h = torch.zeros(H, dtype=x.dtype)
    c = torch.zeros(H, dtype=x.dtype)
    rows = {k: [] for k in SUBS}
    for t in range(x.shape[0]):
        z = Wih @ x[t] + Whh @ h
        if bih is not None:
            z = z + bih
        if bhh is not None:
            z = z + bhh
        i, f, g, o = z[:H].sigmoid(), z[H:2 * H].sigmoid(), z[2 * H:3 * H].tanh(), z[3 * H:].sigmoid()
        c = f * c + i * g
        h = o * c.tanh()
        for k, v in zip(SUBS, (i, f, g, o, c, h)):
            rows[k].append(v)
    subs = {k: torch.stack(v) if v else torch.zeros(0, H, dtype=x.dtype) for k, v in rows.items()}
    return subs["h"], subs
