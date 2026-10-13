"""ATLAS_SITE and ATLAS_ROOT (design §3.1). The same rules as site/scripts/atlas/root.ts.

- ATLAS_SITE: the site folder a command runs in. In a lane that is the lane's copy (<lanes>/<name>/site; the lanes
  folder is site/.work unless ATLAS_LANES_DIR or "lanesDir" in atlas.config(.local).json moves it, e.g. to another
  drive).
- ATLAS_ROOT: for a lane, the `root` its lane.json records (<lanes>/<name>/lane.json, next to the lane's site/): a lane
  outside the repo has no repo above it. Otherwise the nearest ancestor of ATLAS_SITE that contains CONTRACTS.md and
  pipeline/. From a lane in site/.work/<name>/site, site/.work/<name>/.. has no pipeline/, so the search continues up to
  the repo root.

Resolution order:
  ATLAS_SITE  1. the ATLAS_SITE environment variable (the CLI always sets it)
              2. walking up from the current directory: the first folder that is a site
                 (svelte.config.js + package.json + src/lib), or the site/ child of a repo root
              3. <ATLAS_ROOT>/site
  ATLAS_ROOT  1. the ATLAS_ROOT environment variable (the CLI always sets it)
              2. the root recorded in ATLAS_SITE's lane.json (a lane)
              3. the nearest ancestor of ATLAS_SITE with CONTRACTS.md and pipeline/
              4. the repo this file lives in (pipeline/atlas_kit/paths.py -> two levels up)
No script needs a '../../site' relative path: use site_path(...) and root_path(...).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from .errors import fail

_HERE_ROOT = Path(__file__).resolve().parents[2]  # <root>/pipeline/atlas_kit/paths.py


def is_site(d: Path) -> bool:
    return (d / "svelte.config.js").is_file() and (d / "package.json").is_file() and (d / "src" / "lib").is_dir()


def is_root(d: Path) -> bool:
    return (d / "CONTRACTS.md").is_file() and (d / "pipeline").is_dir()


def lane_info(site: Path) -> Optional[dict]:
    """What the lane.json next to a lane's site/ says ({"name", "dir", "root"?, "main"?}); for a lane without one in
    site/.work/<name>/site, its name and folder; None for the main tree."""
    s = site.resolve()
    if s.name != "site":
        return None
    lj = s.parent / "lane.json"
    if lj.is_file():
        try:
            j = json.loads(lj.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            j = None
        if isinstance(j, dict) and j.get("format") == "atlas-lane/1" and isinstance(j.get("name"), str) and j["name"]:
            out = {"name": j["name"], "dir": s.parent}
            for key in ("root", "main"):
                if isinstance(j.get(key), str) and j[key]:
                    out[key] = Path(j[key]).resolve()
            return out
    if s.parent.parent.name == ".work":
        return {"name": s.parent.name, "dir": s.parent}
    return None


def find_root(site: Path) -> Optional[Path]:
    """The repo root for `site`: its lane.json `root` (a lane, possibly outside the repo), else the nearest ancestor of
    `site` (the site itself excluded) that contains CONTRACTS.md and pipeline/."""
    lane = lane_info(site)
    if lane and lane.get("root") is not None and is_root(lane["root"]):
        return lane["root"]
    for d in site.resolve().parents:
        if is_root(d):
            return d
    return None


def _site_from(start: Path) -> Optional[Path]:
    start = start.resolve()
    for d in (start, *start.parents):
        if is_site(d):
            return d
        if is_root(d) and is_site(d / "site"):
            return d / "site"
    return None


def _env_dir(name: str) -> Optional[Path]:
    v = os.environ.get(name, "").strip()
    if not v:
        return None
    p = Path(v)
    if not p.is_dir():
        fail("AK002", f"{name} points to a folder that does not exist: {v}",
             fix=f"unset {name}, or run through `npm run atlas -- ...`, which sets it")
    return p.resolve()


def atlas_root() -> Path:
    env = _env_dir("ATLAS_ROOT")
    if env is not None:
        if not is_root(env):
            fail("AK002", f"ATLAS_ROOT has no CONTRACTS.md and pipeline/: {env}",
                 fix="point ATLAS_ROOT at the repo root (the folder with CONTRACTS.md), or unset it")
        return env
    site = _env_dir("ATLAS_SITE") or _site_from(Path.cwd())
    if site is not None:
        r = find_root(site)
        if r is not None:
            return r
    if is_root(_HERE_ROOT):
        return _HERE_ROOT
    fail("AK002", "cannot find the repo root (a folder with CONTRACTS.md and pipeline/)",
         fix="run from inside the repo or a lane, or set ATLAS_ROOT")


def atlas_site() -> Path:
    env = _env_dir("ATLAS_SITE")
    if env is not None:
        if not is_site(env):
            fail("AK002", f"ATLAS_SITE is not a site folder (no svelte.config.js, package.json, src/lib): {env}",
                 fix="point ATLAS_SITE at site/ or at a lane's <lanes>/<name>/site, or unset it")
        return env
    site = _site_from(Path.cwd())
    if site is not None:
        return site
    root = atlas_root()
    if is_site(root / "site"):
        return root / "site"
    fail("AK002", "cannot find the site folder", fix="run from inside the repo, or set ATLAS_SITE")


def site_path(*parts: str) -> Path:
    return atlas_site().joinpath(*parts)


def root_path(*parts: str) -> Path:
    return atlas_root().joinpath(*parts)


def lane_name(site: Optional[Path] = None) -> Optional[str]:
    """'w1-e1' for <root>/site/.work/w1-e1/site or for any <lanes>/w1-e1/site with a lane.json; None for the main tree."""
    info = lane_info(site or atlas_site())
    return info["name"] if info else None


def rel(p: os.PathLike | str) -> str:
    """A path relative to ATLAS_ROOT with forward slashes (for messages and spec `code` entries)."""
    p = Path(p).resolve()
    try:
        return p.relative_to(atlas_root()).as_posix()
    except ValueError:
        return p.as_posix()
