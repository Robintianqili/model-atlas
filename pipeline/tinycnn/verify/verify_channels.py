"""channels.json structure + recomputed top activations (verification stream).
usage: python pipeline/tinycnn/verify/verify_channels.py"""
import base64, json, os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "pipeline", "tinycnn"))
from common import TinyVGG, load_class  # noqa: E402
from classes import BASE  # noqa: E402
D = os.path.join(ROOT, "site", "static", "models", "tinycnn")
spec = json.load(open(os.path.join(D, "spec.json"), encoding="utf-8"))
W = np.fromfile(os.path.join(D, "weights.bin"), dtype="<f4")
m = TinyVGG((8, 8, 16, 16), 10)
m.load_state_dict({t["name"]: torch.tensor(W[t["offset"]:t["offset"] + t["length"]].reshape(t["shape"]).copy()) for t in spec["tensors"]}); m.eval()
ch = json.load(open(os.path.join(D, "channels.json")))
bad = 0; n = 0; maxd = 0.0
from PIL import Image
sheet = []
for lid, L in ch["layers"].items():
    P = L["patchSize"]
    assert len(L["channels"]) == spec["layers"][[l["id"] for l in spec["layers"]].index(lid)]["outShape"][0], lid
    for c in L["channels"]:
        assert len(c["top"]) == 6 and len(c["classMeanMax"]) == 10 and "icon" in c, (lid, c["channel"])
        acts = [t["act"] for t in c["top"]]
        if acts != sorted(acts, reverse=True): bad += 1
        for t in c["top"][:2]:
            img = load_class(BASE[t["class"]])[t["fileIndex"]].reshape(1, 1, 28, 28)
            with torch.no_grad():
                a = m.trace(torch.tensor(img).float() / 255)[lid][0, c["channel"]]
            d = abs(float(a.max()) - t["act"]); maxd = max(maxd, d); n += 1
            if d > 1e-3 or float(a[t["y"], t["x"]]) < float(a.max()) - 1e-4: bad += 1
            patch = np.frombuffer(base64.b64decode(t["patch"]), np.uint8)
            if patch.size != P * P: bad += 1
        if c["channel"] < 2:
            sheet.append([np.frombuffer(base64.b64decode(t["patch"]), np.uint8).reshape(P, P) for t in c["top"]] +
                         [np.frombuffer(base64.b64decode(c["icon"]), np.uint8).reshape(L.get("iconCell", P) if False else -1, int(np.sqrt(len(base64.b64decode(c["icon"])))))])
print(f"channels.json: {n} top activations recomputed, max |diff| {maxd:.2e}, problems {bad}")
# render conv2_2 channel 0/1 patches + icon
cell = 64
img = np.zeros((len(sheet) * (cell + 4), 7 * (cell + 4)), np.uint8) + 60
for r, row in enumerate(sheet):
    for c, p in enumerate(row):
        im = np.array(Image.fromarray(p).resize((cell, cell), Image.NEAREST))
        img[r * (cell + 4):r * (cell + 4) + cell, c * (cell + 4):c * (cell + 4) + cell] = im
Image.fromarray(img).save(os.path.join(HERE, "out", "channels_sample.png"))
print("wrote out/channels_sample.png (rows: conv1_1 c0,c1, conv1_2 c0,c1, conv2_1 c0,c1, conv2_2 c0,c1; last column = icon)")
