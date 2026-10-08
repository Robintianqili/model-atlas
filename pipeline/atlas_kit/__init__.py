"""atlas_kit: the Python side of the Model Atlas DevKit (research/devkit/DEVKIT_DESIGN.md §4.11).

    import atlas_kit as ak
    ak.export_model(describe, cases, slug=..., name=(en, zh))   # kind A: spec + weights + reference
    ak.Rng(seed)                                                 # mulberry32, bit-identical to core/rng.ts
    ak.TapeWriter(process, schema, seed, opt)                    # atlas-tape/1 reference tapes (kind C)
    ak.json_arg(a.input)                                         # --input <json> or --input @<file> (reference.py)
    ak.json_bytes(obj), ak.write_json(path, obj)                 # UTF-8 JSON without NaN, LF (extra assets)
    ak.validate(dir)                                             # structural checks of a model folder
    ak.data.quickdraw(...), ak.data.tinystories_chars(...), ak.data.karate()

Run with PYTHONPATH=<repo>/pipeline (the CLI sets it, plus ATLAS_SITE and ATLAS_ROOT). Python >= 3.10.
Importing atlas_kit switches TF32 off (numerics.enforce). torch is needed only for the export side: Rng, TapeWriter,
validate and paths work without it.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .errors import CODES, AtlasError, fail  # noqa: E402
from ._io import json_arg, json_bytes, read_json, write_bytes, write_json  # noqa: E402
from .numerics import enforce as _enforce  # noqa: E402
from .paths import atlas_root, atlas_site, root_path, site_path  # noqa: E402
from .rng import Rng, fnv1a32  # noqa: E402
from .tape import TapeWriter  # noqa: E402
from .validate import validate  # noqa: E402
from . import data  # noqa: E402

_enforce()

_TORCH_NAMES = {"SpecBuilder": "spec", "Node": "spec", "Case": "spec", "export_model": "export", "ExportReport": "export"}


def __getattr__(name: str):
    """The export side imports torch lazily, so kind-C references can use ak.Rng and ak.TapeWriter without it."""
    mod = _TORCH_NAMES.get(name)
    if mod is None:
        raise AttributeError(f"module 'atlas_kit' has no attribute {name!r}")
    try:
        import importlib
        m = importlib.import_module(f".{mod}", __name__)
    except ImportError as e:  # torch missing
        fail("AK002", f"atlas_kit.{name} needs torch ({e})",
             fix="pip install -r pipeline/requirements.txt (a CPU wheel of torch is enough)")
    value = getattr(m, name)
    globals()[name] = value
    return value


__all__ = ["AtlasError", "CODES", "Case", "ExportReport", "Node", "Rng", "SpecBuilder", "TapeWriter", "atlas_root",
           "atlas_site", "data", "export_model", "fail", "fnv1a32", "json_arg", "json_bytes", "read_json", "root_path",
           "site_path", "validate", "write_bytes", "write_json"]
