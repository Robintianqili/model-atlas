"""Feature-visualisation icons: for each conv channel, an input patch optimised to excite the channel's centre cell.
Adds 'icon' (base64 uint8, patchSize x patchSize, same crop as the top patches) and 'iconAct' to every channel
in site/static/models/tinycnn/channels.json. Run after export.py:  python channel_icons.py"""
import base64, json, os
import numpy as np, torch, torch.nn.functional as F
from common import TinyVGG
from classes import BASE

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "..", "..", "site", "static", "models", "tinycnn", "channels.json")
m = TinyVGG((8, 8, 16, 16), len(BASE))
m.load_state_dict(torch.load(os.path.join(HERE, "model_final.pt"), map_location="cpu"))
m.eval()
for p in m.parameters():
    p.requires_grad_(False)
chan = json.load(open(PATH))
PAD = chan["pad"]
_k = torch.tensor([1.0, 2.0, 1.0])
BLUR = (_k[:, None] * _k[None, :] / 16).reshape(1, 1, 3, 3)


def act(x, layer):
    t = m.trace(x)
    return t[layer]


for layer, info in chan["layers"].items():
    C = len(info["channels"])
    H = act(torch.zeros(1, 1, 28, 28), layer).shape[-1]
    cy = cx = H // 2
    jump, rs, P = info["jump"], info["rfSize"], info["patchSize"]
    g = torch.Generator().manual_seed(0)
    rng = np.random.default_rng(0)
    p = (-1.5 + 0.1 * torch.randn(C, 1, 28, 28, generator=g)).requires_grad_(True)
    opt = torch.optim.Adam([p], lr=0.05)
    idx = torch.arange(C)
    for step in range(400):
        x = torch.sigmoid(F.conv2d(p, BLUR, padding=1))
        dy, dx = (int(v) for v in rng.integers(-1, 2, 2))
        a = act(torch.roll(x, (dy, dx), (2, 3)), layer)[idx, idx, cy, cx]  # channel c's centre cell for batch item c
        tv = (x[..., 1:, :] - x[..., :-1, :]).abs().mean((1, 2, 3)) + (x[..., :, 1:] - x[..., :, :-1]).abs().mean((1, 2, 3))
        loss = (-a + 3.0 * tv + 1.0 * x.mean((1, 2, 3))).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        x = torch.sigmoid(F.conv2d(p, BLUR, padding=1))
        a = act(x, layer)[idx, idx, cy, cx]
    y0, x0 = cy * jump, cx * jump
    info["iconCell"] = [cy, cx]
    info["iconRf"] = [y0, x0, y0 + rs - 1, x0 + rs - 1]
    for c in range(C):
        img = np.zeros((28 + 2 * PAD, 28 + 2 * PAD), np.uint8)
        img[PAD:PAD + 28, PAD:PAD + 28] = np.round(x[c, 0].numpy() * 255).astype(np.uint8)
        patch = img[y0:y0 + P, x0:x0 + P]
        ch = info["channels"][c]
        ch["icon"] = base64.b64encode(patch.tobytes()).decode()
        ch["iconAct"] = round(float(a[c]), 4)
        if c < 1:
            print(layer, c, "icon act", ch["iconAct"], "best dataset act", ch["top"][0]["act"])
            print("\n".join("".join(" .:-=+*#%@"[min(9, v // 26)] for v in row) for row in patch))
chan["iconNote"] = ("icon = input patch (same crop as the top patches) optimised by gradient ascent to maximise the channel's "
                    "centre-cell pre-ReLU activation: input = sigmoid(3x3 binomial blur of a free 28x28 map), random +-1 px jitter, "
                    "Adam lr 0.05, 400 steps, penalties 3*total variation + 1*mean ink. iconAct = the activation it reaches. Precomputed by pipeline/tinycnn/channel_icons.py.")
json.dump(chan, open(PATH, "w"))
print("channels.json", os.path.getsize(PATH))
