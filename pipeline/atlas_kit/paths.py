"""ATLAS_SITE and ATLAS_ROOT (design §3.1). The same rules as site/scripts/atlas/root.ts.

- ATLAS_SITE: the site folder a command runs in. In a lane that is the lane's copy (site/.work/<name>/site).
- ATLAS_ROOT: the nearest ancestor of ATLAS_SITE that contains CONTRACTS.md and pipeline/. From a lane,
  site/.work/<name>/site/.. has no pipeline/, so the search continues up to the repo root.

Resolution order:
  ATLAS_SITE  1. the ATLAS_SITE environment variable (the CLI always sets it)
              2. walking up from the current directory: the first folder that is a site
                 (svelte.config.js + package.json + src/lib), or the site/ child of a repo root
              3. <ATLAS_ROOT>/site
  ATLAS_ROOT  1. the ATLAS_ROOT environment variable (the CLI always sets it)
              2. the nearest ancestor of ATLAS_SITE with CONTRACTS.md and pipeline/
              3. the repo this file lives in (pipeline/atlas_kit/paths.py -> two levels up)
No script needs a '../../site' relative path: use site_path(...) and root_path(...).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .errors import fail

_HERE_ROOT = Path(__file__).resolve().parents[2]  # <root>/pipeline/atlas_kit/paths.py


def is_site(d: Path) -> bool:
    return (d / "svelte.config.js").is_file() and (d / "package.json").is_file() and (d / "src" / "lib").is_dir()


def is_root(d: Path) -> bool:
    return (d / "CONTRACTS.md").is_file() and (d / "pipeline").is_dir()


def find_root(site: Path) -> Optional[Path]:
    """Nearest ancestor of `site` (the site itself excluded) that contains CONTRACTS.md and pipeline/."""
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
         fix="run from inside the repo, or set ATLAS_ROOT")


def atlas_site() -> Path:
    env = _env_dir("ATLAS_SITE")
    if env is not None:
        if not is_site(env):
            fail("AK002", f"ATLAS_SITE is not a site folder (no svelte.config.js, package.json, src/lib): {env}",
                 fix="point ATLAS_SITE at site/ or at a lane's site/.work/<name>/site, or unset it")
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
    """'w1-e1' for <root>/site/.work/w1-e1/site, None for the main tree."""
    s = (site or atlas_site()).resolve()
    if s.name == "site" and s.parent.parent.name == ".work":
        return s.parent.name
    return None


def rel(p: os.PathLike | str) -> str:
    """A path relative to ATLAS_ROOT with forward slashes (for messages and spec `code` entries)."""
    p = Path(p).resolve()
    try:
        return p.relative_to(atlas_root()).as_posix()
    except ValueError:
        return p.as_posix()
