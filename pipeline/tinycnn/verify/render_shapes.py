"""Render out/shapes.json (28x28 preprocess outputs from verify_ts.ts) into labelled PNG sheets."""
import json, os, sys
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
items = json.load(open(os.path.join(HERE, "out", "shapes.json")))
flt = sys.argv[1] if len(sys.argv) > 1 else ""
items = [it for it in items if flt in it["name"]]
S, COLS = 6, 6
cw, ch = 28 * S + 8, 28 * S + 40
rows = (len(items) + COLS - 1) // COLS
sheet = Image.new("RGB", (COLS * cw, rows * ch), (70, 70, 70))
d = ImageDraw.Draw(sheet)
for i, it in enumerate(items):
    a = (np.clip(np.array(it["img"], np.float32).reshape(28, 28), 0, 1) * 255).astype(np.uint8)
    im = Image.fromarray(a).resize((28 * S, 28 * S), Image.NEAREST).convert("RGB")
    r, c = divmod(i, COLS)
    x, y = c * cw + 4, r * ch + 4
    sheet.paste(im, (x, y))
    name = it["name"].replace(", brush", "\nbrush").replace(" | ", "\n", 1)
    lines = name.split("\n")
    d.text((x, y + 28 * S + 2), lines[0][:30] + " -> " + f"{it['pred']} {it['p']}", fill=(255, 255, 160))
    d.text((x, y + 28 * S + 14), " ".join(lines[1:])[:32], fill=(200, 200, 200))
    d.text((x, y + 28 * S + 26), " ".join(lines[1:])[32:64], fill=(200, 200, 200))
out = os.path.join(HERE, "out", f"shapes{('_' + flt.replace(' ', '_')) if flt else ''}.png")
sheet.save(out)
print(out, sheet.size)
