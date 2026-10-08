"""node2vec: writes static/models/node2vec/facts.json, the one measured number the lesson quotes that it does not
compute live: how many of the 32 unlabelled members the GCN page's network sorts right from the same two labelled
leaders (members 0 and 33). It is read from the gcn package's facts.json (trainAcc, measured by pipeline/gcn/train.py)
in the same site folder, so the two pages cannot disagree; claims.test.ts checks it again.

    npm run atlas -- py pipeline/node2vec/facts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    import atlas_kit as ak
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import atlas_kit as ak

gcn = ak.read_json(ak.site_path("static", "models", "gcn", "facts.json"))["facts"]["trainAcc"]
right = round(gcn["value"] * 32)
assert abs(right - gcn["value"] * 32) < 1e-9, gcn
doc = {"format": "atlas-facts/1", "facts": {"gcnRight": {
    "value": right, "unit": "of the 32 unlabelled members, sorted right by the GCN trained on the labels of members 0 and 33",
    "source": f"static/models/gcn/facts.json trainAcc {gcn['value']} x 32 ({gcn['source']})", "measured": gcn["measured"]}}}
out = ak.site_path("static", "models", "node2vec", "facts.json")
out.parent.mkdir(parents=True, exist_ok=True)
ak.write_json(out, doc, indent=1)
print(f"wrote {out}: gcnRight = {right}")
