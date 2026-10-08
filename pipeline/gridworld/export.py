"""gridworld: write site/static/models/gridworld/presets.json, the maze presets of the editor (kind C has no weights).

The boards are the same as BOARDS in site/src/lib/explainers/gridworld/world.ts ('classic' is also its DEFAULTS.rows):
'.' open, '#' wall, 'G' goal (+1), 'P' pit (-1), 'S' start. 'walled' is the classic maze with the goal walled off,
the fixture whose lesson must say that the goal cannot be reached.

    npm run atlas -- py pipeline/gridworld/export.py      # writes into the site of ATLAS_SITE (a lane or the main tree)
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

BOARDS = [
    {"id": "classic", "en": "Classic", "zh": "经典", "rows": [".....G", ".###..", "...#P.", ".#....", "S#.#.."]},
    {"id": "cliff", "en": "Cliff", "zh": "悬崖", "rows": [".......", ".......", ".......", "SPPPPPG"]},
    {"id": "maze", "en": "Maze", "zh": "迷宫", "rows": ["S.#...", "..#.#.", ".##.#.", "....#.", ".####.", ".....G"]},
    {"id": "walled", "en": "Walled-off goal", "zh": "被墙围住的终点", "rows": ["....#G", ".###.#", "...#P.", ".#....", "S#.#.."]},
]

if __name__ == "__main__":
    out = ak.site_path("static", "models", "gridworld", "presets.json")
    write_bytes(out, json_bytes({"format": "gridworld-presets/1", "boards": BOARDS}, where=str(out)))
    print(f"wrote {out} ({len(BOARDS)} boards)")
