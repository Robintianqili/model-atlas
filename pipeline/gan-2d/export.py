"""gan-2d: write site/static/models/gan-2d/presets.json, the preset list of the input box (kind B: no weights ship).

The points themselves are made in TypeScript (site/src/lib/explainers/gan-2d/data.ts, seeded), so the twin and the page
read the same points; this file only names the presets the input box offers, in its order.

    npm run atlas -- py pipeline/gan-2d/export.py      # writes into the site of ATLAS_SITE (a lane or the main tree)
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    import atlas_kit as ak
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import atlas_kit as ak
from atlas_kit._io import json_bytes, write_bytes

PRESETS = [
    {"id": "ring", "en": "Ring", "zh": "圆环"},
    {"id": "blob", "en": "One blob", "zh": "一团点"},
    {"id": "two", "en": "Two blobs", "zh": "两团点"},
    {"id": "eight", "en": "Eight blobs", "zh": "八团点"},
    {"id": "smiley", "en": "Smiley", "zh": "笑脸"},
]

if __name__ == "__main__":
    out = ak.site_path("static", "models", "gan-2d", "presets.json")
    write_bytes(out, json_bytes({"format": "gan-2d-presets/1", "presets": PRESETS}, where=str(out)))
    print(f"wrote {out} ({len(PRESETS)} presets)")
