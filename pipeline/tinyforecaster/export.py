"""tinyforecaster: write site/static/models/tinyforecaster/presets.json, the series presets of the input widget.

Nothing else is exported: the forecaster has no shipped weights (it trains live in the browser from a seeded start).
The presets are generators of the series adapter (site/src/lib/atlas/data/series.ts), named with their seed; the
package's adapter (explainers/tinyforecaster/adapters/series/adapter.ts) builds each series from that in TS.

    npm run atlas -- py pipeline/tinyforecaster/export.py      # writes into the site of ATLAS_SITE (a lane or the main tree)
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

# the order of the input widget: the lesson's start series first
PRESETS = [{"id": name, "seed": 1} for name in ("seasonal", "sine", "square", "trend", "randomWalk")]

if __name__ == "__main__":
    out = ak.site_path("static", "models", "tinyforecaster", "presets.json")
    write_bytes(out, json_bytes({"format": "tinyforecaster-presets/1", "presets": PRESETS}, where=str(out)))
    print(f"wrote {out} ({len(PRESETS)} presets)")
